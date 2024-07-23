#include "CrnnNet.h"
#include "OcrUtils.h"
#include <numeric>

CrnnNet::~CrnnNet() {
    net.clear();
}

void CrnnNet::setNumThread(int numOfThread) {
    numThread = numOfThread;
}

char *readKeysFromAssets(AAssetManager *mgr) {
    //LOGI("readKeysFromAssets start...");
    if (mgr == NULL) {
        LOGE(" %s", "AAssetManager==NULL");
        return NULL;
    }
    char *buffer;
    /*获取文件名并打开*/
    AAsset *asset = AAssetManager_open(mgr, "keys.txt", AASSET_MODE_UNKNOWN);
    if (asset == NULL) {
        LOGE(" %s", "asset==NULL");
        return NULL;
    }
    /*获取文件大小*/
    off_t bufferSize = AAsset_getLength(asset);
    //LOGI("file size : %d", bufferSize);
    buffer = (char *) malloc(bufferSize + 1);
    buffer[bufferSize] = 0;
    int numBytesRead = AAsset_read(asset, buffer, bufferSize);
    //LOGI("readKeysFromAssets: %d", numBytesRead);
    /*关闭文件*/
    AAsset_close(asset);
    //LOGI("readKeysFromAssets exit...");
    return buffer;
}

bool CrnnNet::initModel(AAssetManager *mgr) {
    int ret_param = net.load_param(mgr, "crnn_lite_op.param");
    int ret_bin = net.load_model(mgr, "crnn_lite_op.bin");
    if (ret_param != 0 || ret_bin != 0) {
        LOGE("# %d  %d", ret_param, ret_bin);
        return false;
    }

    char *buffer = readKeysFromAssets(mgr);
    if (buffer != NULL) {
        std::istringstream inStr(buffer);
        std::string line;
        int size = 0;
        while (getline(inStr, line)) {
            keys.emplace_back(line);
            size++;
        }
        free(buffer);
        LOGI("keys size(%d)", size);
    } else {
        LOGE(" txt file not found");
        return false;
    }

    return true;
}

TextLine CrnnNet::scoreToTextLine(const float *outputData, int h, int w) {
    int keySize = keys.size();
    std::string strRes;
    std::vector<float> scores;
    std::vector<int> indices;
    std::vector<int> columns;
    int lastIndex = 0;
    int maxIndex;
    float maxValue;

    auto emptyCount = 0;
    auto lastEmptyCount = 0;
    auto curCharNum = 0;
    for (int i = 0; i < h; i++) {
        maxIndex = 0;
        maxValue = -1000.f;
        //do softmax
        std::vector<float> exps(w);
        for (int j = 0; j < w; j++) {
            float expSingle = exp(outputData[i * w + j]);
            exps.at(j) = expSingle;
        }
        float partition = accumulate(exps.begin(), exps.end(), 0.0);//row sum
        for (int j = 0; j < w; j++) {
            float softmax = exps[j] / partition;
            if (softmax > maxValue) {
                maxValue = softmax;
                maxIndex = j;
            }
        }

        if (maxIndex == 0) {
            emptyCount++;
        }

        if (maxIndex > 0 && maxIndex == lastIndex) {
            emptyCount++;
        }
        //no softmax
        /*for (int j = 0; j < w; j++) {
            if (srcData[i * w + j] > maxValue) {
                maxValue = srcData[i * w + j];
                maxIndex = j;
            }
        }*/
        if (maxIndex > 0 && maxIndex < keySize && (!(i > 0 && maxIndex == lastIndex))) {
            if (scores.size() <= 0) {
                int count = 1 + emptyCount;
                columns.emplace_back(count);
            } else {
                int count = columns.at(curCharNum - 1) + emptyCount / 2;
                columns.at(curCharNum - 1) = count;
                count = emptyCount / 2 + 1;
                columns.emplace_back(count);
            }
            scores.emplace_back(maxValue);
            indices.emplace_back(i);
            strRes.append(keys[maxIndex - 1]);
            curCharNum++;
            emptyCount = 0;
        }
        if (i == h - 1 && columns.size() >= 1) {
            columns.at(columns.size() - 1) += emptyCount;
        }
        lastIndex = maxIndex;
    }
    return {strRes, scores, indices, columns, h};
}

TextLine CrnnNet::getTextLine(const cv::Mat &src) {
    float scale = (float) dstHeight / (float) src.rows;
    int dstWidth = int((float) src.cols * scale);

    cv::Mat srcResize;
    cv::resize(src, srcResize, cv::Size(dstWidth, dstHeight));

    ncnn::Mat input = ncnn::Mat::from_pixels(
            srcResize.data, ncnn::Mat::PIXEL_RGB,
            srcResize.cols, srcResize.rows);

    input.substract_mean_normalize(meanValues, normValues);

    ncnn::Extractor extractor = net.create_extractor();
    extractor.set_num_threads(numThread);
    extractor.input("input", input);

    ncnn::Mat out;
    extractor.extract("out", out);

    return scoreToTextLine((float *) out.data, out.h, out.w);
}

std::vector<TextLine> CrnnNet::getTextLines(std::vector<cv::Mat> &partImg) {
    int size = partImg.size();
    std::vector<TextLine> textLines(size);
    for (int i = 0; i < size; ++i) {
        //getTextLine
        double startCrnnTime = getCurrentTime();
        TextLine textLine = getTextLine(partImg[i]);
        double endCrnnTime = getCurrentTime();
        textLine.time = endCrnnTime - startCrnnTime;
        textLines[i] = textLine;
    }
    return textLines;
}
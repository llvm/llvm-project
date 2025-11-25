/*
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "hipBin.h"

int main(int argc, char* argv[]){
    HipBin hipbin;
    vector<HipBinBase*>& platformPtrs = hipbin.getHipBinPtrs();
    for (unsigned int j = 0; j < platformPtrs.size(); j++) {
        if (argc == 1) {
        platformPtrs.at(j)->printFull();
        }
        for (int i = 1; i < argc; ++i) {
        HipBinCommand cmd;
        cmd = platformPtrs.at(j)->gethipconfigCmd(argv[i]);
        switch (cmd) {
        case help: platformPtrs.at(j)->printUsage();
            break;
        case path: cout << platformPtrs.at(j)->getHipPath();
            break;
        case roccmpath: cout << platformPtrs.at(j)->getRoccmPath();
            break;
        case cpp_config: cout << platformPtrs.at(j)->getCppConfig();
            break;
        case compiler: cout << CompilerTypeStr((
                                platformPtrs.at(j)->getPlatformInfo()).compiler);
            break;
        case platform: cout << PlatformTypeStr((
                                platformPtrs.at(j)->getPlatformInfo()).platform);
            break;
        case runtime: cout << RuntimeTypeStr((
                                platformPtrs.at(j)->getPlatformInfo()).runtime);
            break;
        case hipclangpath: cout << platformPtrs.at(j)->getCompilerPath();
            break;
        case full: platformPtrs.at(j)->printFull();
            break;
        case version: cout << platformPtrs.at(j)->getHipVersion();
            break;
        case check: platformPtrs.at(j)->checkHipconfig();
            break;
        case newline: cout << endl;
            break;
        default:
            platformPtrs.at(j)->printUsage();
            break;
        }
        }
    }
}

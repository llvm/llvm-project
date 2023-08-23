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

#include "hipBin_util.h"
#include "hipBin_amd.h"
#include "hipBin_nvidia.h"
#include <iostream>
#include <vector>
#include <string>

class HipBinUtil;
class HipBinBase;
class HipBinAmd;
class HipBinNvidia;
class HipBin;


class HipBin {
 private:
  HipBinUtil* hipBinUtilPtr_;
  vector<HipBinBase*> hipBinBasePtrs_;
  vector<PlatformInfo> platformVec_;
  HipBinBase* hipBinNVPtr_;
  HipBinBase* hipBinAMDPtr_;

 public:
  HipBin();
  ~HipBin();
  vector<HipBinBase*>& getHipBinPtrs();
  vector<PlatformInfo>& getPlaformInfo();
  void executeHipBin(string filename, int argc, char* argv[]);
  void executeHipConfig(int argc, char* argv[]);
  void executeHipCC(int argc, char* argv[]);
};


// Implementation ================================================
//===========================================================================

HipBin::HipBin() {
  hipBinUtilPtr_ = hipBinUtilPtr_->getInstance();
  hipBinNVPtr_ = new HipBinNvidia();
  hipBinAMDPtr_ = new HipBinAmd();
  bool platformDetected = false;
  if (hipBinAMDPtr_->detectPlatform()) {
    // populates the struct with AMD info
    const PlatformInfo& platformInfo = hipBinAMDPtr_->getPlatformInfo();
    platformVec_.push_back(platformInfo);
    hipBinBasePtrs_.push_back(hipBinAMDPtr_);
    platformDetected = true;
  } else if (hipBinNVPtr_->detectPlatform()) {
    // populates the struct with Nvidia info
    const PlatformInfo& platformInfo = hipBinNVPtr_->getPlatformInfo();
    platformVec_.push_back(platformInfo);
    hipBinBasePtrs_.push_back(hipBinNVPtr_);
    platformDetected = true;
  }
  // if no device is detected, then it is defaulted to AMD
  if (!platformDetected) {
    std::cerr << "Device not supported - Defaulting to AMD" << endl;
    // populates the struct with AMD info
    const PlatformInfo& platformInfo = hipBinAMDPtr_->getPlatformInfo();
    platformVec_.push_back(platformInfo);
    hipBinBasePtrs_.push_back(hipBinAMDPtr_);
  }
}

HipBin::~HipBin() {
  delete hipBinNVPtr_;
  delete hipBinAMDPtr_;
  // clearing the vector so no one accesses the pointers
  hipBinBasePtrs_.clear();
  // clearing the platform vector as the pointers are deleted
  platformVec_.clear();
  delete hipBinUtilPtr_;
}

vector<PlatformInfo>& HipBin::getPlaformInfo() {
  return platformVec_;  // Return the populated platform info.
}


vector<HipBinBase*>& HipBin::getHipBinPtrs() {
  return hipBinBasePtrs_;  // Return the populated device pointers.
}


void HipBin::executeHipBin(string filename, int argc, char* argv[]) {
  if (hipBinUtilPtr_->substringPresent(filename, "hipconfig")) {
    executeHipConfig(argc, argv);
  } else if (hipBinUtilPtr_->substringPresent(filename, "hipcc")) {
    executeHipCC(argc, argv);
  } else {
    std::cerr << "Command " << filename
    << " not supported. Name the exe as hipconfig"
    << " or hipcc and then try again ..." << endl;
    exit(-1);
  }
}


void HipBin::executeHipCC(int argc, char* argv[]) {
  vector<HipBinBase*>& platformPtrs = getHipBinPtrs();
  vector<string> argvcc;
  for (int i = 0; i < argc; i++) {
    argvcc.push_back(argv[i]);
  }
  // 0th index points to the first platform detected.
  // In the near future this vector will contain mulitple devices
  platformPtrs.at(0)->executeHipCCCmd(argvcc);
}


void HipBin::executeHipConfig(int argc, char* argv[]) {
  vector<HipBinBase*>& platformPtrs = getHipBinPtrs();
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
      case newline: platformPtrs.at(j)->printFull();
        cout << endl;
        break;
      default:
        platformPtrs.at(j)->printUsage();
        break;
      }
    }
  }
}

//===========================================================================
//===========================================================================

int main(int argc, char* argv[]) {
  fs::path filename(argv[0]);
  filename = filename.filename();

  HipBin hipBin;
  hipBin.executeHipBin(filename.string(), argc, argv);
}

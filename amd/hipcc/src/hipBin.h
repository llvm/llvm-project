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
  vector<HipBinBase *> hipBinBasePtrs_;
  HipBinBase* hipBinNVPtr_;
  HipBinBase* hipBinAMDPtr_;

 public:
  HipBin();
  ~HipBin();
  vector<HipBinBase *> &getHipBinPtrs();
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
    hipBinBasePtrs_.push_back(hipBinAMDPtr_);
    platformDetected = true;
  } else if (hipBinNVPtr_->detectPlatform()) {
    // populates the struct with Nvidia info
    hipBinBasePtrs_.push_back(hipBinNVPtr_);
    platformDetected = true;
  }
  // if no device is detected, then it is defaulted to AMD
  if (!platformDetected) {
    std::cerr << "Device not supported - Defaulting to AMD" << endl;
    // populates the struct with AMD info
    hipBinBasePtrs_.push_back(hipBinAMDPtr_);
  }
}

HipBin::~HipBin() {
  delete hipBinNVPtr_;
  delete hipBinAMDPtr_;
  // clearing the vector so no one accesses the pointers
  hipBinBasePtrs_.clear();
  delete hipBinUtilPtr_;
}

vector<HipBinBase*>& HipBin::getHipBinPtrs() {
  return hipBinBasePtrs_;  // Return the populated device pointers.
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
      case newline: cout << endl;
        break;
      default:
        platformPtrs.at(j)->printUsage();
        break;
      }
    }
  }
}
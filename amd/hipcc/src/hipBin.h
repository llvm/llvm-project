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
  vector<HipBinBase *> hipBinBasePtrs_;
  HipBinBase* hipBinNVPtr_;
  HipBinBase* hipBinAMDPtr_;

 public:
  HipBin();
  ~HipBin();
  vector<HipBinBase *> &getHipBinPtrs();
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



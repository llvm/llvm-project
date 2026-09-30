//===--- OffloadTarget.h - Declare generic offload target -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares a generic "offload" TargetInfo object.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_BASIC_TARGETS_OFFLOADTARGET_H
#define LLVM_CLANG_LIB_BASIC_TARGETS_OFFLOADTARGET_H

#include "clang/Basic/OffloadArch.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "llvm/Support/Compiler.h"
#include "llvm/TargetParser/Triple.h"

namespace clang {
namespace targets {

class LLVM_LIBRARY_VISIBILITY OffloadTargetInfo : public TargetInfo {
protected:
  OffloadArch DeviceArch = OffloadArch::getUnused();

public:
  OffloadTargetInfo(const llvm::Triple &Triple) : TargetInfo(Triple) {}

  // Clang driver emits -target-cpu to indicate offload device architecture for
  // both CUDA and SYCL. We override setCPU to capture this offload device
  // architecture information.
  bool setCPU(StringRef Name) override {
    DeviceArch = StringToOffloadArch(Name);
    return !DeviceArch.isUnknownOrUnused();
  }

  bool isValidCPUName(StringRef Name) const override {
    return !StringToOffloadArch(Name).isUnknown();
  }

  void fillValidCPUList(SmallVectorImpl<StringRef> &Values) const override {
    fillValidOffloadArchList(Values);
  }

  OffloadArch getOffloadArch() const override { return DeviceArch; }
};

} // namespace targets
} // namespace clang

#endif // LLVM_CLANG_LIB_BASIC_TARGETS_OFFLOADTARGET_H
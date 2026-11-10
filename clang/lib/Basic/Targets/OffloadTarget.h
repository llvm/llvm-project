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

  /// \return true if \p Arch is an offload device architecture this target
  /// should accept.
  virtual bool isSupportedOffloadArch(OffloadArch Arch) const { return false; }

public:
  OffloadTargetInfo(const llvm::Triple &Triple) : TargetInfo(Triple) {}

  bool isValidCPUName(StringRef Name) const override {
    return isSupportedOffloadArch(StringToOffloadArch(Name));
  }

  bool setCPU(StringRef Name) override {
    OffloadArch Arch = StringToOffloadArch(Name);
    if (!isSupportedOffloadArch(Arch))
      return false;
    DeviceArch = Arch;
    return true;
  }

  OffloadArch getOffloadArch() const override { return DeviceArch; }
};

} // namespace targets
} // namespace clang

#endif // LLVM_CLANG_LIB_BASIC_TARGETS_OFFLOADTARGET_H

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
#include "llvm/Support/Compiler.h"
#include "llvm/TargetParser/Triple.h"

namespace clang {
namespace targets {

/// Base TargetInfo class for offload device targets. Targets inheriting this
/// class obtain an offload architecture from the `-target-cpu` option, which
/// is then accessible via getOffloadArch().
class LLVM_LIBRARY_VISIBILITY OffloadTargetInfo : public TargetInfo {
  OffloadArch SelectedOffloadArch = OffloadArch::getUnused();

protected:
  /// \return true if \p Arch is an offload device architecture this target
  /// should accept.
  virtual bool isSupportedOffloadArch(OffloadArch Arch) const { return false; }

  /// Sets offload arch directly. Does not check isSupportedOffloadArch().
  void setOffloadArch(OffloadArch Arch) { SelectedOffloadArch = Arch; }

public:
  OffloadTargetInfo(const llvm::Triple &Triple) : TargetInfo(Triple) {}

  bool isValidCPUName(StringRef Name) const override {
    return isSupportedOffloadArch(StringToOffloadArch(Name));
  }

  bool setCPU(StringRef Name) override {
    OffloadArch Arch = StringToOffloadArch(Name);
    if (!isSupportedOffloadArch(Arch))
      return false;
    setOffloadArch(Arch);
    return true;
  }

  OffloadArch getOffloadArch() const override { return SelectedOffloadArch; }
};

} // namespace targets
} // namespace clang

#endif // LLVM_CLANG_LIB_BASIC_TARGETS_OFFLOADTARGET_H

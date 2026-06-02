//===--- EZH.cpp - Implement EZH target feature support -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements EZH TargetInfo objects.
//
//===----------------------------------------------------------------------===//

#include "EZH.h"
#include "clang/Basic/MacroBuilder.h"

using namespace clang;
using namespace clang::targets;

const char *const EZHTargetInfo::GCCRegNames[] = {
    "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7",
    "gpo", "gpd", "cfs", "cfm", "sp", "pc", "gpi", "ra"
};

ArrayRef<const char *> EZHTargetInfo::getGCCRegNames() const {
  return llvm::ArrayRef(GCCRegNames);
}

ArrayRef<TargetInfo::GCCRegAlias> EZHTargetInfo::getGCCRegAliases() const {
  return {};
}

void EZHTargetInfo::getTargetDefines(const LangOptions &Opts,
                                       MacroBuilder &Builder) const {
  Builder.defineMacro("__ezh__");
  Builder.defineMacro("__SOFTFP__");
  if (HasBitsliceInterrupts) {
    Builder.defineMacro("__EZH_BITSLICE_INTERRUPTS__");
  }
}

//===--- P2.cpp - Implement P2 target feature support -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements P2 TargetInfo objects.
//
//===----------------------------------------------------------------------===//

#include "P2.h"
#include "clang/Basic/MacroBuilder.h"
#include "llvm/ADT/StringSwitch.h"

using namespace clang;
using namespace clang::targets;

namespace clang {
namespace targets {

} // namespace targets
} // namespace clang

// bool P2TargetInfo::isValidCPUName(StringRef Name) const {

// }

// void P2TargetInfo::fillValidCPUList(SmallVectorImpl<StringRef> &Values) const {

// }

void P2TargetInfo::getTargetDefines(const LangOptions &Opts,
                                     MacroBuilder &Builder) const {
}

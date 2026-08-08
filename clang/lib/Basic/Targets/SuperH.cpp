//===--- SuperH.cpp - Declare SuperH target feature support -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares SuperH TargetInfo objects.
//
//===----------------------------------------------------------------------===//

#include "SuperH.h"
#include "clang/Basic/MacroBuilder.h"
#include "llvm/ADT/StringRef.h"

using namespace clang;
using namespace clang::targets;

struct LLVM_LIBRARY_VISIBILITY SHCPUInfo {
  llvm::StringLiteral Name;
};

static constexpr SHCPUInfo CPUInfo[] = {
  {{"sh1"}},
  {{"sh2"}},
  {{"sh2a"}},
  {{"sh3"}},
  {{"sh4"}},
  {{"sh4a"}},
};

bool SuperHTargetInfo::isValidCPUName(StringRef Name) const {
  return llvm::any_of(
      CPUInfo, [&](const SHCPUInfo &Info) { return Info.Name == Name; });
}

void SuperHTargetInfo::fillValidCPUList(SmallVectorImpl<StringRef> &Values) const {
  for (const SHCPUInfo &Info : CPUInfo)
    Values.push_back(Info.Name);
}

bool SuperHTargetInfo::setCPU(const std::string &Name) {
  // Set the ABI field based on the device or family name.
  const auto *It = llvm::find_if(
      CPUInfo, [&](const SHCPUInfo &Info) { return Info.Name == Name; });
  if (It != std::end(CPUInfo)) {
    CPU = Name;
    ABI = "sh";
    return true;
  }

  // Parameter Name is neither valid family name nor valid device name.
  return false;
}

std::optional<std::string>
SuperHTargetInfo::handleAsmEscapedChar(char EscChar) const {
  return std::nullopt;
}

void SuperHTargetInfo::getTargetDefines(const LangOptions &Opts,
                                     MacroBuilder &Builder) const {
  Builder.defineMacro("SH");
  Builder.defineMacro("__SH");
  Builder.defineMacro("__SH__");
}
//===--- OpenCLOptions.cpp---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/OpenCLOptions.h"

namespace clang {

bool OpenCLOptions::isKnown(llvm::StringRef Ext) const {
  return OptMap.find(Ext) != OptMap.end();
}

bool OpenCLOptions::isAvailableOption(llvm::StringRef Ext,
                                      const LangOptions &LO) const {
  if (!isKnown(Ext))
    return false;

  auto &OptInfo = OptMap.find(Ext)->getValue();
  if (OptInfo.isCoreIn(LO) || OptInfo.isOptionalCoreIn(LO))
    return isSupported(Ext, LO);

  return isEnabled(Ext);
}

bool OpenCLOptions::isEnabled(llvm::StringRef Ext) const {
  auto I = OptMap.find(Ext);
  return I != OptMap.end() && I->getValue().Enabled;
}

bool OpenCLOptions::isWithPragma(llvm::StringRef Ext) const {
  auto E = OptMap.find(Ext);
  return E != OptMap.end() && E->second.WithPragma;
}

bool OpenCLOptions::isSupported(llvm::StringRef Ext,
                                const LangOptions &LO) const {
  auto I = OptMap.find(Ext);
  return I != OptMap.end() && I->getValue().Supported &&
         I->getValue().isAvailableIn(LO);
}

bool OpenCLOptions::isSupportedCore(llvm::StringRef Ext,
                                    const LangOptions &LO) const {
  auto I = OptMap.find(Ext);
  return I != OptMap.end() && I->getValue().Supported &&
         I->getValue().isCoreIn(LO);
}

bool OpenCLOptions::isSupportedOptionalCore(llvm::StringRef Ext,
                                            const LangOptions &LO) const {
  auto I = OptMap.find(Ext);
  return I != OptMap.end() && I->getValue().Supported &&
         I->getValue().isOptionalCoreIn(LO);
}

bool OpenCLOptions::isSupportedCoreOrOptionalCore(llvm::StringRef Ext,
                                                  const LangOptions &LO) const {
  return isSupportedCore(Ext, LO) || isSupportedOptionalCore(Ext, LO);
}

bool OpenCLOptions::isSupportedExtension(llvm::StringRef Ext,
                                         const LangOptions &LO) const {
  auto I = OptMap.find(Ext);
  return I != OptMap.end() && I->getValue().Supported &&
         I->getValue().isAvailableIn(LO) &&
         !isSupportedCoreOrOptionalCore(Ext, LO);
}

void OpenCLOptions::enable(llvm::StringRef Ext, bool V) {
  OptMap[Ext].Enabled = V;
}

void OpenCLOptions::acceptsPragma(llvm::StringRef Ext, bool V) {
  OptMap[Ext].WithPragma = V;
}

void OpenCLOptions::support(llvm::StringRef Ext, bool V) {
  assert(!Ext.empty() && "Extension is empty.");
  assert(Ext[0] != '+' && Ext[0] != '-');
  OptMap[Ext].Supported = V;
}

OpenCLOptions::OpenCLOptions() {
#define OPENCL_GENERIC_EXTENSION(Ext, WithPragma, AvailVer, CoreVer, OptVer)   \
  OptMap.insert_or_assign(                                                     \
      #Ext, OpenCLOptionInfo{WithPragma, AvailVer, CoreVer, OptVer});
#include "clang/Basic/OpenCLExtensions.def"
}

void OpenCLOptions::addSupport(const llvm::StringMap<bool> &FeaturesMap,
                               const LangOptions &Opts) {
  for (const auto &F : FeaturesMap) {
    const auto &Name = F.getKey();
    if (F.getValue() && isKnown(Name) && OptMap[Name].isAvailableIn(Opts))
      support(Name);
  }
}

void OpenCLOptions::disableAll() {
  for (auto &Opt : OptMap)
    Opt.getValue().Enabled = false;
}

} // end namespace clang

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

bool OpenCLOptions::isEnabled(llvm::StringRef Ext) const {
  auto E = OptMap.find(Ext);
  return E != OptMap.end() && E->second.Enabled;
}

bool OpenCLOptions::isWithPragma(llvm::StringRef Ext) const {
  auto E = OptMap.find(Ext);
  return E != OptMap.end() && E->second.WithPragma;
}

bool OpenCLOptions::isSupported(llvm::StringRef Ext,
                                const LangOptions &LO) const {
  auto E = OptMap.find(Ext);
  if (E == OptMap.end()) {
    return false;
  }
  auto I = OptMap.find(Ext)->getValue();
  return I.Supported && I.isAvailableIn(LO);
}

bool OpenCLOptions::isSupportedCore(llvm::StringRef Ext,
                                    const LangOptions &LO) const {
  auto E = OptMap.find(Ext);
  if (E == OptMap.end()) {
    return false;
  }
  auto I = OptMap.find(Ext)->getValue();
  return I.Supported && I.isCoreIn(LO);
}

bool OpenCLOptions::isSupportedOptionalCore(llvm::StringRef Ext,
                                            const LangOptions &LO) const {
  auto E = OptMap.find(Ext);
  if (E == OptMap.end()) {
    return false;
  }
  auto I = OptMap.find(Ext)->getValue();
  return I.Supported && I.isOptionalCoreIn(LO);
}

bool OpenCLOptions::isSupportedCoreOrOptionalCore(llvm::StringRef Ext,
                                                  const LangOptions &LO) const {
  return isSupportedCore(Ext, LO) || isSupportedOptionalCore(Ext, LO);
}

bool OpenCLOptions::isSupportedExtension(llvm::StringRef Ext,
                                         const LangOptions &LO) const {
  auto E = OptMap.find(Ext);
  if (E == OptMap.end()) {
    return false;
  }
  auto I = OptMap.find(Ext)->getValue();
  return I.Supported && I.isAvailableIn(LO) &&
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

void OpenCLOptions::enableSupportedCore(const LangOptions &LO) {
  for (auto &Opt : OptMap)
    if (isSupportedCoreOrOptionalCore(Opt.getKey(), LO))
      Opt.getValue().Enabled = true;
}

} // end namespace clang

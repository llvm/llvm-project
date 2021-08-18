//===- LangOptions.cpp - C Language Family Language Options ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the LangOptions class.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/LangOptions.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Path.h"

using namespace clang;

LangOptions::LangOptions() : LangStd(LangStandard::lang_unspecified) {
#define LANGOPT(Name, Bits, Default, Description) Name = Default;
#define ENUM_LANGOPT(Name, Type, Bits, Default, Description) set##Name(Default);
#include "clang/Basic/LangOptions.def"
}

void LangOptions::resetNonModularOptions() {
#define LANGOPT(Name, Bits, Default, Description)
#define BENIGN_LANGOPT(Name, Bits, Default, Description) Name = Default;
#define BENIGN_ENUM_LANGOPT(Name, Type, Bits, Default, Description) \
  Name = static_cast<unsigned>(Default);
#include "clang/Basic/LangOptions.def"

  // These options do not affect AST generation.
  NoSanitizeFiles.clear();
  XRayAlwaysInstrumentFiles.clear();
  XRayNeverInstrumentFiles.clear();

  CurrentModule.clear();
  IsHeaderFile = false;
}

bool LangOptions::isNoBuiltinFunc(StringRef FuncName) const {
  for (unsigned i = 0, e = NoBuiltinFuncs.size(); i != e; ++i)
    if (FuncName.equals(NoBuiltinFuncs[i]))
      return true;
  return false;
}

VersionTuple LangOptions::getOpenCLVersionTuple() const {
  const int Ver = OpenCLCPlusPlus ? OpenCLCPlusPlusVersion : OpenCLVersion;
  return VersionTuple(Ver / 100, (Ver % 100) / 10);
}

void LangOptions::remapPathPrefix(SmallString<256> &Path) const {
  for (const auto &Entry : MacroPrefixMap)
    if (llvm::sys::path::replace_path_prefix(Path, Entry.first, Entry.second))
      break;
}

std::string LangOptions::getOpenCLVersionString() const {
  std::string Result;
  {
    llvm::raw_string_ostream Out(Result);
    Out << (OpenCLCPlusPlus ? "C++ for OpenCL" : "OpenCL C") << " version "
        << getOpenCLVersionTuple().getAsString();
  }
  return Result;
}

FPOptions FPOptions::defaultWithoutTrailingStorage(const LangOptions &LO) {
  FPOptions result(LO);
  return result;
}

LLVM_DUMP_METHOD void FPOptions::dump() {
#define OPTION(NAME, TYPE, WIDTH, PREVIOUS)                                    \
  llvm::errs() << "\n " #NAME " " << get##NAME();
#include "clang/Basic/FPOptions.def"
  llvm::errs() << "\n";
}

LLVM_DUMP_METHOD void FPOptionsOverride::dump() {
#define OPTION(NAME, TYPE, WIDTH, PREVIOUS)                                    \
  if (has##NAME##Override())                                                   \
    llvm::errs() << "\n " #NAME " Override is " << get##NAME##Override();
#include "clang/Basic/FPOptions.def"
  llvm::errs() << "\n";
}

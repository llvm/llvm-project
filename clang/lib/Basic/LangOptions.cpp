//===--- LangOptions.cpp - C Language Family Language Options ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the LangOptions class.
//
//===----------------------------------------------------------------------===//
#include "clang/Basic/LangOptions.h"
#include "llvm/ADT/StringRef.h"

using namespace clang;

LangOptions::LangOptions()
  : IsHeaderFile(false) {
#define LANGOPT(Name, Bits, Default, Description) Name = Default;
#define ENUM_LANGOPT(Name, Type, Bits, Default, Description) set##Name(Default);
#include "clang/Basic/LangOptions.def"
}

void LangOptions::resetNonModularOptions() {
#define LANGOPT(Name, Bits, Default, Description)
#define BENIGN_LANGOPT(Name, Bits, Default, Description) Name = Default;
#define BENIGN_ENUM_LANGOPT(Name, Type, Bits, Default, Description) \
  Name = Default;
#include "clang/Basic/LangOptions.def"

  // These options do not affect AST generation.
  SanitizerBlacklistFiles.clear();
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

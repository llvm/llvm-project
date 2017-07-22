//===--- Nios2.cpp - Implement Nios2 target feature support ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements Nios2 TargetInfo objects.
//
//===----------------------------------------------------------------------===//

#include "Nios2.h"
#include "Targets.h"
#include "clang/Basic/MacroBuilder.h"
#include "clang/Basic/TargetBuiltins.h"
#include "llvm/ADT/StringSwitch.h"

using namespace clang;
using namespace clang::targets;

const Builtin::Info Nios2TargetInfo::BuiltinInfo[] = {
#define BUILTIN(ID, TYPE, ATTRS)                                               \
  {#ID, TYPE, ATTRS, nullptr, ALL_LANGUAGES, nullptr},
#define TARGET_BUILTIN(ID, TYPE, ATTRS, FEATURE)                               \
  {#ID, TYPE, ATTRS, nullptr, ALL_LANGUAGES, FEATURE},
#include "clang/Basic/BuiltinsNios2.def"
};

bool Nios2TargetInfo::isFeatureSupportedByCPU(StringRef Feature,
                                              StringRef CPU) const {
  const bool isR2 = CPU == "nios2r2";
  return llvm::StringSwitch<bool>(Feature)
      .Case("nios2r2mandatory", isR2)
      .Case("nios2r2bmx", isR2)
      .Case("nios2r2mpx", isR2)
      .Case("nios2r2cdx", isR2)
      .Default(false);
}

void Nios2TargetInfo::getTargetDefines(const LangOptions &Opts,
                                       MacroBuilder &Builder) const {
  DefineStd(Builder, "nios2", Opts);
  DefineStd(Builder, "NIOS2", Opts);

  Builder.defineMacro("__nios2");
  Builder.defineMacro("__NIOS2");
  Builder.defineMacro("__nios2__");
  Builder.defineMacro("__NIOS2__");
}

ArrayRef<Builtin::Info> Nios2TargetInfo::getTargetBuiltins() const {
  return llvm::makeArrayRef(BuiltinInfo, clang::Nios2::LastTSBuiltin -
                                             Builtin::FirstTSBuiltin);
}

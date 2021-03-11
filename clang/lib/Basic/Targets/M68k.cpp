//===--- M68k.cpp - Implement M68k targets feature support-------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements M68k TargetInfo objects.
//
//===----------------------------------------------------------------------===//

#include "M68k.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/TargetBuiltins.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/TargetParser.h"
#include <cstring>

namespace clang {
namespace targets {

M68kTargetInfo::M68kTargetInfo(const llvm::Triple &Triple,
                               const TargetOptions &)
    : TargetInfo(Triple) {

  std::string Layout = "";

  // M68k is Big Endian
  Layout += "E";

  // FIXME how to wire it with the used object format?
  Layout += "-m:e";

  // M68k pointers are always 32 bit wide even for 16 bit cpus
  Layout += "-p:32:32";

  // M68k integer data types
  Layout += "-i8:8:8-i16:16:16-i32:16:32";

  // FIXME no floats at the moment

  // The registers can hold 8, 16, 32 bits
  Layout += "-n8:16:32";

  // 16 bit alignment for both stack and aggregate
  // in order to conform to ABI used by GCC
  Layout += "-a:0:16-S16";

  resetDataLayout(Layout);

  SizeType = UnsignedInt;
  PtrDiffType = SignedInt;
  IntPtrType = SignedInt;
}

bool M68kTargetInfo::setCPU(const std::string &Name) {
  StringRef N = Name;
  CPU = llvm::StringSwitch<CPUKind>(N)
            .Case("generic", CK_68000)
            .Case("M68000", CK_68000)
            .Case("M68010", CK_68010)
            .Case("M68020", CK_68020)
            .Case("M68030", CK_68030)
            .Case("M68040", CK_68040)
            .Case("M68060", CK_68060)
            .Default(CK_Unknown);
  return CPU != CK_Unknown;
}

void M68kTargetInfo::getTargetDefines(const LangOptions &Opts,
                                      MacroBuilder &Builder) const {
  using llvm::Twine;

  Builder.defineMacro("__m68k__");

  Builder.defineMacro("mc68000");
  Builder.defineMacro("__mc68000");
  Builder.defineMacro("__mc68000__");

  // For sub-architecture
  switch (CPU) {
  case CK_68010:
    Builder.defineMacro("mc68010");
    Builder.defineMacro("__mc68010");
    Builder.defineMacro("__mc68010__");
    break;
  case CK_68020:
    Builder.defineMacro("mc68020");
    Builder.defineMacro("__mc68020");
    Builder.defineMacro("__mc68020__");
    break;
  case CK_68030:
    Builder.defineMacro("mc68030");
    Builder.defineMacro("__mc68030");
    Builder.defineMacro("__mc68030__");
    break;
  case CK_68040:
    Builder.defineMacro("mc68040");
    Builder.defineMacro("__mc68040");
    Builder.defineMacro("__mc68040__");
    break;
  case CK_68060:
    Builder.defineMacro("mc68060");
    Builder.defineMacro("__mc68060");
    Builder.defineMacro("__mc68060__");
    break;
  default:
    break;
  }
}

ArrayRef<Builtin::Info> M68kTargetInfo::getTargetBuiltins() const {
  // FIXME: Implement.
  return None;
}

bool M68kTargetInfo::hasFeature(StringRef Feature) const {
  // FIXME elaborate moar
  return Feature == "M68000";
}

const char *const M68kTargetInfo::GCCRegNames[] = {
    "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
    "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7",
    "pc"};

ArrayRef<const char *> M68kTargetInfo::getGCCRegNames() const {
  return llvm::makeArrayRef(GCCRegNames);
}

ArrayRef<TargetInfo::GCCRegAlias> M68kTargetInfo::getGCCRegAliases() const {
  // No aliases.
  return None;
}

bool M68kTargetInfo::validateAsmConstraint(
    const char *&Name, TargetInfo::ConstraintInfo &info) const {
  switch (*Name) {
  case 'a': // address register
  case 'd': // data register
  case 'f': // floating point register
    info.setAllowsRegister();
    return true;
  case 'K': // the constant 1
  case 'L': // constant -1^20 .. 1^19
  case 'M': // constant 1-4:
    return true;
  }
  // FIXME: Support all constraints like 'N', 'O', 'P', 'R'
  return false;
}

const char *M68kTargetInfo::getClobbers() const {
  // FIXME: Is this really right?
  return "";
}

M68kTargetInfo::BuiltinVaListKind M68kTargetInfo::getBuiltinVaListKind() const {
  // FIXME: implement
  llvm_unreachable("Not implemented yet");
}

} // namespace targets
} // namespace clang

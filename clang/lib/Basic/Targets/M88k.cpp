//===--- M88k.cpp - Implement M88k targets feature support ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements M88k TargetInfo objects.
//
//===----------------------------------------------------------------------===//

#include "M88k.h"
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

M88kTargetInfo::M88kTargetInfo(const llvm::Triple &Triple,
                               const TargetOptions &)
    : TargetInfo(Triple) {

  std::string Layout = "";

  // M68k is Big Endian
  Layout += "E";

  // FIXME how to wire it with the used object format?
  Layout += "-m:e";

  // Pointers are 32 bit.
  Layout += "-p:32:8:32";

  // Make sure that global data has at least 16 bits of alignment by
  // default, so that we can refer to it using LARL.  We don't have any
  // special requirements for stack variables though.
  Layout += "-i1:8:16-i8:8:16";

  // 64-bit integers are naturally aligned.
  Layout += "-i64:64";

  // 128-bit floats are aligned only to 64 bits.
  Layout += "-f128:64";

  // We prefer 16 bits of aligned for all globals; see above.
  Layout += "-a:8:16";

  // Integer registers are 32bits.
  Layout += "-n32";

  resetDataLayout(Layout);

  SizeType = UnsignedInt;
  PtrDiffType = SignedInt;
  IntPtrType = SignedInt;
}

bool M88kTargetInfo::setCPU(const std::string &Name) {
  StringRef N = Name;
  CPU = llvm::StringSwitch<CPUKind>(N)
            .Case("generic", CK_88000)
            .Case("M88000", CK_88000)
            .Case("M88100", CK_88100)
            .Case("M88110", CK_88110)
            .Default(CK_Unknown);
  return CPU != CK_Unknown;
}

void M88kTargetInfo::getTargetDefines(const LangOptions &Opts,
                                      MacroBuilder &Builder) const {
  using llvm::Twine;

  Builder.defineMacro("__m88k__");
  Builder.defineMacro("__m88k");

  // For sub-architecture
  switch (CPU) {
  case CK_88000:
    Builder.defineMacro("__mc88000__");
    break;
  case CK_88100:
    Builder.defineMacro("__mc88100__");
    break;
  case CK_88110:
    Builder.defineMacro("__mc88110__");
    break;
  default:
    break;
  }
}

ArrayRef<Builtin::Info> M88kTargetInfo::getTargetBuiltins() const {
  // TODO Implement.
  return None;
}

bool M88kTargetInfo::hasFeature(StringRef Feature) const {
  // TODO Implement.
  return Feature == "M88000";
}

const char *const M88kTargetInfo::GCCRegNames[] = {
    // TODO Extended registers, control registers.
    "r0",  "r1",  "r2",  "r3",  "r4",  "r5",  "r6",  "r7",  "r8",  "r9",  "r10",
    "r11", "r12", "r13", "r14", "r15", "r16", "r17", "r18", "r19", "r20", "r21",
    "r22", "r23", "r24", "r25", "r26", "r27", "r28", "r29", "r39", "r31"};

ArrayRef<const char *> M88kTargetInfo::getGCCRegNames() const {
  return llvm::makeArrayRef(GCCRegNames);
}

ArrayRef<TargetInfo::GCCRegAlias> M88kTargetInfo::getGCCRegAliases() const {
  // No aliases.
  return None;
}

bool M88kTargetInfo::validateAsmConstraint(
    const char *&Name, TargetInfo::ConstraintInfo &info) const {
  // TODO Implement.
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
  return false;
}

const char *M88kTargetInfo::getClobbers() const {
  // TODO Implement.
  return "";
}

} // namespace targets
} // namespace clang

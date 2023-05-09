//===--- LoongArch.cpp - Implement LoongArch target feature support -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements LoongArch TargetInfo objects.
//
//===----------------------------------------------------------------------===//

#include "LoongArch.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/MacroBuilder.h"
#include "clang/Basic/TargetBuiltins.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/TargetParser.h"

using namespace clang;
using namespace clang::targets;

ArrayRef<const char *> LoongArchTargetInfo::getGCCRegNames() const {
  static const char *const GCCRegNames[] = {
      // General purpose registers.
      "$r0", "$r1", "$r2", "$r3", "$r4", "$r5", "$r6", "$r7", "$r8", "$r9",
      "$r10", "$r11", "$r12", "$r13", "$r14", "$r15", "$r16", "$r17", "$r18",
      "$r19", "$r20", "$r21", "$r22", "$r23", "$r24", "$r25", "$r26", "$r27",
      "$r28", "$r29", "$r30", "$r31",
      // Floating point registers.
      "$f0", "$f1", "$f2", "$f3", "$f4", "$f5", "$f6", "$f7", "$f8", "$f9",
      "$f10", "$f11", "$f12", "$f13", "$f14", "$f15", "$f16", "$f17", "$f18",
      "$f19", "$f20", "$f21", "$f22", "$f23", "$f24", "$f25", "$f26", "$f27",
      "$f28", "$f29", "$f30", "$f31",
      // Condition flag registers.
      "$fcc0", "$fcc1", "$fcc2", "$fcc3", "$fcc4", "$fcc5", "$fcc6", "$fcc7"};
  return llvm::ArrayRef(GCCRegNames);
}

ArrayRef<TargetInfo::GCCRegAlias>
LoongArchTargetInfo::getGCCRegAliases() const {
  static const TargetInfo::GCCRegAlias GCCRegAliases[] = {
      {{"$zero"}, "$r0"},       {{"$ra"}, "$r1"},    {{"$tp"}, "$r2"},
      {{"$sp"}, "$r3"},         {{"$a0"}, "$r4"},    {{"$a1"}, "$r5"},
      {{"$a2"}, "$r6"},         {{"$a3"}, "$r7"},    {{"$a4"}, "$r8"},
      {{"$a5"}, "$r9"},         {{"$a6"}, "$r10"},   {{"$a7"}, "$r11"},
      {{"$t0"}, "$r12"},        {{"$t1"}, "$r13"},   {{"$t2"}, "$r14"},
      {{"$t3"}, "$r15"},        {{"$t4"}, "$r16"},   {{"$t5"}, "$r17"},
      {{"$t6"}, "$r18"},        {{"$t7"}, "$r19"},   {{"$t8"}, "$r20"},
      {{"$fp", "$s9"}, "$r22"}, {{"$s0"}, "$r23"},   {{"$s1"}, "$r24"},
      {{"$s2"}, "$r25"},        {{"$s3"}, "$r26"},   {{"$s4"}, "$r27"},
      {{"$s5"}, "$r28"},        {{"$s6"}, "$r29"},   {{"$s7"}, "$r30"},
      {{"$s8"}, "$r31"},        {{"$fa0"}, "$f0"},   {{"$fa1"}, "$f1"},
      {{"$fa2"}, "$f2"},        {{"$fa3"}, "$f3"},   {{"$fa4"}, "$f4"},
      {{"$fa5"}, "$f5"},        {{"$fa6"}, "$f6"},   {{"$fa7"}, "$f7"},
      {{"$ft0"}, "$f8"},        {{"$ft1"}, "$f9"},   {{"$ft2"}, "$f10"},
      {{"$ft3"}, "$f11"},       {{"$ft4"}, "$f12"},  {{"$ft5"}, "$f13"},
      {{"$ft6"}, "$f14"},       {{"$ft7"}, "$f15"},  {{"$ft8"}, "$f16"},
      {{"$ft9"}, "$f17"},       {{"$ft10"}, "$f18"}, {{"$ft11"}, "$f19"},
      {{"$ft12"}, "$f20"},      {{"$ft13"}, "$f21"}, {{"$ft14"}, "$f22"},
      {{"$ft15"}, "$f23"},      {{"$fs0"}, "$f24"},  {{"$fs1"}, "$f25"},
      {{"$fs2"}, "$f26"},       {{"$fs3"}, "$f27"},  {{"$fs4"}, "$f28"},
      {{"$fs5"}, "$f29"},       {{"$fs6"}, "$f30"},  {{"$fs7"}, "$f31"},
  };
  return llvm::ArrayRef(GCCRegAliases);
}

bool LoongArchTargetInfo::validateAsmConstraint(
    const char *&Name, TargetInfo::ConstraintInfo &Info) const {
  // See the GCC definitions here:
  // https://gcc.gnu.org/onlinedocs/gccint/Machine-Constraints.html
  // Note that the 'm' constraint is handled in TargetInfo.
  switch (*Name) {
  default:
    return false;
  case 'f':
    // A floating-point register (if available).
    Info.setAllowsRegister();
    return true;
  case 'k':
    // A memory operand whose address is formed by a base register and
    // (optionally scaled) index register.
    Info.setAllowsMemory();
    return true;
  case 'l':
    // A signed 16-bit constant.
    Info.setRequiresImmediate(-32768, 32767);
    return true;
  case 'I':
    // A signed 12-bit constant (for arithmetic instructions).
    Info.setRequiresImmediate(-2048, 2047);
    return true;
  case 'J':
    // Integer zero.
    Info.setRequiresImmediate(0);
    return true;
  case 'K':
    // An unsigned 12-bit constant (for logic instructions).
    Info.setRequiresImmediate(0, 4095);
    return true;
  case 'Z':
    // ZB: An address that is held in a general-purpose register. The offset is
    //     zero.
    // ZC: A memory operand whose address is formed by a base register
    //     and offset that is suitable for use in instructions with the same
    //     addressing mode as ll.w and sc.w.
    if (Name[1] == 'C' || Name[1] == 'B') {
      Info.setAllowsMemory();
      ++Name; // Skip over 'Z'.
      return true;
    }
    return false;
  }
}

std::string
LoongArchTargetInfo::convertConstraint(const char *&Constraint) const {
  std::string R;
  switch (*Constraint) {
  case 'Z':
    // "ZC"/"ZB" are two-character constraints; add "^" hint for later
    // parsing.
    R = "^" + std::string(Constraint, 2);
    ++Constraint;
    break;
  default:
    R = TargetInfo::convertConstraint(Constraint);
    break;
  }
  return R;
}

void LoongArchTargetInfo::getTargetDefines(const LangOptions &Opts,
                                           MacroBuilder &Builder) const {
  Builder.defineMacro("__loongarch__");
  unsigned GRLen = getRegisterWidth();
  Builder.defineMacro("__loongarch_grlen", Twine(GRLen));
  if (GRLen == 64)
    Builder.defineMacro("__loongarch64");

  if (HasFeatureD)
    Builder.defineMacro("__loongarch_frlen", "64");
  else if (HasFeatureF)
    Builder.defineMacro("__loongarch_frlen", "32");
  else
    Builder.defineMacro("__loongarch_frlen", "0");

  // TODO: define __loongarch_arch and __loongarch_tune.

  StringRef ABI = getABI();
  if (ABI == "lp64d" || ABI == "lp64f" || ABI == "lp64s")
    Builder.defineMacro("__loongarch_lp64");

  if (ABI == "lp64d" || ABI == "ilp32d") {
    Builder.defineMacro("__loongarch_hard_float");
    Builder.defineMacro("__loongarch_double_float");
  } else if (ABI == "lp64f" || ABI == "ilp32f") {
    Builder.defineMacro("__loongarch_hard_float");
    Builder.defineMacro("__loongarch_single_float");
  } else if (ABI == "lp64s" || ABI == "ilp32s") {
    Builder.defineMacro("__loongarch_soft_float");
  }

  Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_1");
  Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_2");
  Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_4");
  if (GRLen == 64)
    Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_8");
}

static constexpr Builtin::Info BuiltinInfo[] = {
#define BUILTIN(ID, TYPE, ATTRS)                                               \
  {#ID, TYPE, ATTRS, nullptr, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
#define TARGET_BUILTIN(ID, TYPE, ATTRS, FEATURE)                               \
  {#ID, TYPE, ATTRS, FEATURE, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
#include "clang/Basic/BuiltinsLoongArch.def"
};

bool LoongArchTargetInfo::initFeatureMap(
    llvm::StringMap<bool> &Features, DiagnosticsEngine &Diags, StringRef CPU,
    const std::vector<std::string> &FeaturesVec) const {
  if (getTriple().getArch() == llvm::Triple::loongarch64)
    Features["64bit"] = true;
  if (getTriple().getArch() == llvm::Triple::loongarch32)
    Features["32bit"] = true;

  return TargetInfo::initFeatureMap(Features, Diags, CPU, FeaturesVec);
}

/// Return true if has this feature.
bool LoongArchTargetInfo::hasFeature(StringRef Feature) const {
  bool Is64Bit = getTriple().getArch() == llvm::Triple::loongarch64;
  // TODO: Handle more features.
  return llvm::StringSwitch<bool>(Feature)
      .Case("loongarch32", !Is64Bit)
      .Case("loongarch64", Is64Bit)
      .Case("32bit", !Is64Bit)
      .Case("64bit", Is64Bit)
      .Default(false);
}

ArrayRef<Builtin::Info> LoongArchTargetInfo::getTargetBuiltins() const {
  return llvm::ArrayRef(BuiltinInfo, clang::LoongArch::LastTSBuiltin -
                                         Builtin::FirstTSBuiltin);
}

bool LoongArchTargetInfo::handleTargetFeatures(
    std::vector<std::string> &Features, DiagnosticsEngine &Diags) {
  for (const auto &Feature : Features) {
    if (Feature == "+d" || Feature == "+f") {
      // "d" implies "f".
      HasFeatureF = true;
      if (Feature == "+d") {
        HasFeatureD = true;
      }
    }
  }
  return true;
}

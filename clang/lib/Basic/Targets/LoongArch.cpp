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
#include "llvm/Support/TargetParser.h"
#include "llvm/Support/raw_ostream.h"

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
      "$f28", "$f29", "$f30", "$f31"};
  return llvm::makeArrayRef(GCCRegNames);
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
  return llvm::makeArrayRef(GCCRegAliases);
}

bool LoongArchTargetInfo::validateAsmConstraint(
    const char *&Name, TargetInfo::ConstraintInfo &Info) const {
  // See the GCC definitions here:
  // https://gcc.gnu.org/onlinedocs/gccint/Machine-Constraints.html
  switch (*Name) {
  // TODO: handle 'k', 'm', "ZB", "ZC".
  default:
    return false;
  case 'f':
    // A floating-point register (if available).
    Info.setAllowsRegister();
    return true;
  case 'l':
    // A signed 16-bit constant.
    Info.setRequiresImmediate(-32768, 32767);
    return true;
  case 'I':
    // A signed 12-bit constant (for arithmetic instructions).
    Info.setRequiresImmediate(-2048, 2047);
    return true;
  case 'K':
    // An unsigned 12-bit constant (for logic instructions).
    Info.setRequiresImmediate(0, 4095);
    return true;
  }
}

void LoongArchTargetInfo::getTargetDefines(const LangOptions &Opts,
                                           MacroBuilder &Builder) const {
  Builder.defineMacro("__loongarch__");
  // TODO: Define more macros.
}

ArrayRef<Builtin::Info> LoongArchTargetInfo::getTargetBuiltins() const {
  // TODO: To be implemented in future.
  return {};
}

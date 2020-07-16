//===--- RISCV.cpp - Implement RISCV target feature support ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements RISCV TargetInfo objects.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "clang/Basic/MacroBuilder.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/TargetParser.h"

using namespace clang;
using namespace clang::targets;

ArrayRef<const char *> RISCVTargetInfo::getGCCRegNames() const {
  static const char *const GCCRegNames[] = {
      // Integer registers
      "x0",  "x1",  "x2",  "x3",  "x4",  "x5",  "x6",  "x7",
      "x8",  "x9",  "x10", "x11", "x12", "x13", "x14", "x15",
      "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23",
      "x24", "x25", "x26", "x27", "x28", "x29", "x30", "x31",

      // Floating point registers
      "f0",  "f1",  "f2",  "f3",  "f4",  "f5",  "f6",  "f7",
      "f8",  "f9",  "f10", "f11", "f12", "f13", "f14", "f15",
      "f16", "f17", "f18", "f19", "f20", "f21", "f22", "f23",
      "f24", "f25", "f26", "f27", "f28", "f29", "f30", "f31"};
  return llvm::makeArrayRef(GCCRegNames);
}

ArrayRef<TargetInfo::GCCRegAlias> RISCVTargetInfo::getGCCRegAliases() const {
  static const TargetInfo::GCCRegAlias GCCRegAliases[] = {
      {{"zero"}, "x0"}, {{"ra"}, "x1"},   {{"sp"}, "x2"},    {{"gp"}, "x3"},
      {{"tp"}, "x4"},   {{"t0"}, "x5"},   {{"t1"}, "x6"},    {{"t2"}, "x7"},
      {{"s0"}, "x8"},   {{"s1"}, "x9"},   {{"a0"}, "x10"},   {{"a1"}, "x11"},
      {{"a2"}, "x12"},  {{"a3"}, "x13"},  {{"a4"}, "x14"},   {{"a5"}, "x15"},
      {{"a6"}, "x16"},  {{"a7"}, "x17"},  {{"s2"}, "x18"},   {{"s3"}, "x19"},
      {{"s4"}, "x20"},  {{"s5"}, "x21"},  {{"s6"}, "x22"},   {{"s7"}, "x23"},
      {{"s8"}, "x24"},  {{"s9"}, "x25"},  {{"s10"}, "x26"},  {{"s11"}, "x27"},
      {{"t3"}, "x28"},  {{"t4"}, "x29"},  {{"t5"}, "x30"},   {{"t6"}, "x31"},
      {{"ft0"}, "f0"},  {{"ft1"}, "f1"},  {{"ft2"}, "f2"},   {{"ft3"}, "f3"},
      {{"ft4"}, "f4"},  {{"ft5"}, "f5"},  {{"ft6"}, "f6"},   {{"ft7"}, "f7"},
      {{"fs0"}, "f8"},  {{"fs1"}, "f9"},  {{"fa0"}, "f10"},  {{"fa1"}, "f11"},
      {{"fa2"}, "f12"}, {{"fa3"}, "f13"}, {{"fa4"}, "f14"},  {{"fa5"}, "f15"},
      {{"fa6"}, "f16"}, {{"fa7"}, "f17"}, {{"fs2"}, "f18"},  {{"fs3"}, "f19"},
      {{"fs4"}, "f20"}, {{"fs5"}, "f21"}, {{"fs6"}, "f22"},  {{"fs7"}, "f23"},
      {{"fs8"}, "f24"}, {{"fs9"}, "f25"}, {{"fs10"}, "f26"}, {{"fs11"}, "f27"},
      {{"ft8"}, "f28"}, {{"ft9"}, "f29"}, {{"ft10"}, "f30"}, {{"ft11"}, "f31"}};
  return llvm::makeArrayRef(GCCRegAliases);
}

bool RISCVTargetInfo::validateAsmConstraint(
    const char *&Name, TargetInfo::ConstraintInfo &Info) const {
  switch (*Name) {
  default:
    return false;
  case 'I':
    // A 12-bit signed immediate.
    Info.setRequiresImmediate(-2048, 2047);
    return true;
  case 'J':
    // Integer zero.
    Info.setRequiresImmediate(0);
    return true;
  case 'K':
    // A 5-bit unsigned immediate for CSR access instructions.
    Info.setRequiresImmediate(0, 31);
    return true;
  case 'f':
    // A floating-point register.
    Info.setAllowsRegister();
    return true;
  case 'A':
    // An address that is held in a general-purpose register.
    Info.setAllowsMemory();
    return true;
  }
}

void RISCVTargetInfo::getTargetDefines(const LangOptions &Opts,
                                       MacroBuilder &Builder) const {
  Builder.defineMacro("__ELF__");
  Builder.defineMacro("__riscv");
  bool Is64Bit = getTriple().getArch() == llvm::Triple::riscv64;
  Builder.defineMacro("__riscv_xlen", Is64Bit ? "64" : "32");
  StringRef CodeModel = getTargetOpts().CodeModel;
  if (CodeModel == "default")
    CodeModel = "small";

  if (CodeModel == "small")
    Builder.defineMacro("__riscv_cmodel_medlow");
  else if (CodeModel == "medium")
    Builder.defineMacro("__riscv_cmodel_medany");

  StringRef ABIName = getABI();
  if (ABIName == "ilp32f" || ABIName == "lp64f")
    Builder.defineMacro("__riscv_float_abi_single");
  else if (ABIName == "ilp32d" || ABIName == "lp64d")
    Builder.defineMacro("__riscv_float_abi_double");
  else
    Builder.defineMacro("__riscv_float_abi_soft");

  if (ABIName == "ilp32e")
    Builder.defineMacro("__riscv_abi_rve");

  if (HasM) {
    Builder.defineMacro("__riscv_mul");
    Builder.defineMacro("__riscv_div");
    Builder.defineMacro("__riscv_muldiv");
  }

  if (HasA)
    Builder.defineMacro("__riscv_atomic");

  if (HasF || HasD) {
    Builder.defineMacro("__riscv_flen", HasD ? "64" : "32");
    Builder.defineMacro("__riscv_fdiv");
    Builder.defineMacro("__riscv_fsqrt");
  }

  if (HasC)
    Builder.defineMacro("__riscv_compressed");

  if (HasB)
    Builder.defineMacro("__riscv_bitmanip");
}

/// Return true if has this feature, need to sync with handleTargetFeatures.
bool RISCVTargetInfo::hasFeature(StringRef Feature) const {
  bool Is64Bit = getTriple().getArch() == llvm::Triple::riscv64;
  return llvm::StringSwitch<bool>(Feature)
      .Case("riscv", true)
      .Case("riscv32", !Is64Bit)
      .Case("riscv64", Is64Bit)
      .Case("m", HasM)
      .Case("a", HasA)
      .Case("f", HasF)
      .Case("d", HasD)
      .Case("c", HasC)
      .Case("experimental-b", HasB)
      .Default(false);
}

/// Perform initialization based on the user configured set of features.
bool RISCVTargetInfo::handleTargetFeatures(std::vector<std::string> &Features,
                                           DiagnosticsEngine &Diags) {
  for (const auto &Feature : Features) {
    if (Feature == "+m")
      HasM = true;
    else if (Feature == "+a")
      HasA = true;
    else if (Feature == "+f")
      HasF = true;
    else if (Feature == "+d")
      HasD = true;
    else if (Feature == "+c")
      HasC = true;
    else if (Feature == "+experimental-b")
      HasB = true;
  }

  return true;
}

bool RISCV32TargetInfo::isValidCPUName(StringRef Name) const {
  return llvm::RISCV::checkCPUKind(llvm::RISCV::parseCPUKind(Name),
                                   /*Is64Bit=*/false);
}

void RISCV32TargetInfo::fillValidCPUList(
    SmallVectorImpl<StringRef> &Values) const {
  llvm::RISCV::fillValidCPUArchList(Values, false);
}

bool RISCV64TargetInfo::isValidCPUName(StringRef Name) const {
  return llvm::RISCV::checkCPUKind(llvm::RISCV::parseCPUKind(Name),
                                   /*Is64Bit=*/true);
}

void RISCV64TargetInfo::fillValidCPUList(
    SmallVectorImpl<StringRef> &Values) const {
  llvm::RISCV::fillValidCPUArchList(Values, true);
}

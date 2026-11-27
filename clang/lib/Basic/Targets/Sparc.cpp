//===--- Sparc.cpp - Implement Sparc target feature support ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements Sparc TargetInfo objects.
//
//===----------------------------------------------------------------------===//

#include "Sparc.h"
#include "Targets.h"
#include "clang/Basic/MacroBuilder.h"
#include "llvm/ADT/StringSwitch.h"

using namespace clang;
using namespace clang::targets;

const char *const SparcTargetInfo::GCCRegNames[] = {
    // clang-format off
    // Integer registers
    "r0",  "r1",  "r2",  "r3",  "r4",  "r5",  "r6",  "r7",  "r8",  "r9",  "r10",
    "r11", "r12", "r13", "r14", "r15", "r16", "r17", "r18", "r19", "r20", "r21",
    "r22", "r23", "r24", "r25", "r26", "r27", "r28", "r29", "r30", "r31",

    // Floating-point registers
    "f0",  "f1",  "f2",  "f3",  "f4",  "f5",  "f6",  "f7",  "f8",  "f9",  "f10",
    "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f19", "f20", "f21",
    "f22", "f23", "f24", "f25", "f26", "f27", "f28", "f29", "f30", "f31", "f32",
    "f34", "f36", "f38", "f40", "f42", "f44", "f46", "f48", "f50", "f52", "f54",
    "f56", "f58", "f60", "f62",

    // Condition code registers
    "icc", "fcc0", "fcc1", "fcc2", "fcc3",
    // clang-format on
};

ArrayRef<const char *> SparcTargetInfo::getGCCRegNames() const {
  return llvm::ArrayRef(GCCRegNames);
}

const TargetInfo::GCCRegAlias SparcTargetInfo::GCCRegAliases[] = {
    {{"g0"}, "r0"},  {{"g1"}, "r1"},  {{"g2"}, "r2"},        {{"g3"}, "r3"},
    {{"g4"}, "r4"},  {{"g5"}, "r5"},  {{"g6"}, "r6"},        {{"g7"}, "r7"},
    {{"o0"}, "r8"},  {{"o1"}, "r9"},  {{"o2"}, "r10"},       {{"o3"}, "r11"},
    {{"o4"}, "r12"}, {{"o5"}, "r13"}, {{"o6", "sp"}, "r14"}, {{"o7"}, "r15"},
    {{"l0"}, "r16"}, {{"l1"}, "r17"}, {{"l2"}, "r18"},       {{"l3"}, "r19"},
    {{"l4"}, "r20"}, {{"l5"}, "r21"}, {{"l6"}, "r22"},       {{"l7"}, "r23"},
    {{"i0"}, "r24"}, {{"i1"}, "r25"}, {{"i2"}, "r26"},       {{"i3"}, "r27"},
    {{"i4"}, "r28"}, {{"i5"}, "r29"}, {{"i6", "fp"}, "r30"}, {{"i7"}, "r31"},
};

ArrayRef<TargetInfo::GCCRegAlias> SparcTargetInfo::getGCCRegAliases() const {
  return llvm::ArrayRef(GCCRegAliases);
}

bool SparcTargetInfo::hasFeature(StringRef Feature) const {
  return llvm::StringSwitch<bool>(Feature)
      .Case("softfloat", SoftFloat)
      .Case("sparc", true)
      .Default(false);
}

bool SparcTargetInfo::initFeatureMap(
    llvm::StringMap<bool> &Features, DiagnosticsEngine &Diags, StringRef CPU,
    const std::vector<std::string> &FeaturesVec) const {
  if (std::optional<llvm::StringMap<bool>> CPUFeatures =
          llvm::Sparc::getSparcDefaultTargetFeatures(CPU))
    Features = std::move(*CPUFeatures);
  return TargetInfo::initFeatureMap(Features, Diags, CPU, FeaturesVec);
}

bool SparcTargetInfo::handleTargetFeatures(std::vector<std::string> &Features,
                                           DiagnosticsEngine &Diags) {
  for (const std::string &Feature : Features) {
    if (Feature == "+soft-float")
      SoftFloat = true;
    else if (Feature == "+v9")
      IsV9 = true;
    else if (Feature == "+leon")
      IsLeon = true;
    else if (Feature == "+hasleoncasa")
      HasLeonCASA = true;
    else if (Feature == "+soft-mul-div")
      SoftMulDiv = true;
  }
  return true;
}

bool SparcTargetInfo::isValidCPUName(StringRef Name) const {
  return llvm::Sparc::isValidCPU(Name);
}

void SparcTargetInfo::fillValidCPUList(
    SmallVectorImpl<StringRef> &Values) const {
  llvm::Sparc::fillValidCPUList(Values);
}

void SparcTargetInfo::getTargetDefines(const LangOptions &Opts,
                                       MacroBuilder &Builder) const {
  DefineStd(Builder, "sparc", Opts);
  Builder.defineMacro("__REGISTER_PREFIX__", "");

  if (SoftFloat)
    Builder.defineMacro("_SOFT_FLOAT");
}

void SparcV8TargetInfo::getTargetDefines(const LangOptions &Opts,
                                         MacroBuilder &Builder) const {
  SparcTargetInfo::getTargetDefines(Opts, Builder);
  // As with GCC, V7 gets none of the V8 macros, since it lacks the integer
  // multiply and divide instructions.
  if (getTriple().isOSSolaris()) {
    if (!SoftMulDiv)
      Builder.defineMacro("__sparcv8");
  } else if (IsV9) {
    Builder.defineMacro("__sparc_v9__");
  } else if (!SoftMulDiv) {
    Builder.defineMacro("__sparcv8");
    Builder.defineMacro("__sparcv8__");
    Builder.defineMacro("__sparc_v8__");
  }
  if (IsLeon)
    Builder.defineMacro("__leon__");
  if (IsV9 || HasLeonCASA) {
    Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_1");
    Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_2");
    Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_4");
    // As with GCC, 64-bit compare-and-swap is only inlined in V8+ mode.
    if (IsV9 && V8Plus)
      Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_8");
  }
  if (LongDoubleWidth == 128)
    Builder.defineMacro("__LONG_DOUBLE_128__");
}

void SparcV9TargetInfo::getTargetDefines(const LangOptions &Opts,
                                         MacroBuilder &Builder) const {
  SparcTargetInfo::getTargetDefines(Opts, Builder);
  Builder.defineMacro("__sparcv9");
  Builder.defineMacro("__arch64__");
  // Solaris doesn't need these variants, but the BSDs do.
  if (!getTriple().isOSSolaris()) {
    Builder.defineMacro("__sparc64__");
    Builder.defineMacro("__sparc_v9__");
    Builder.defineMacro("__sparcv9__");
  }

  Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_1");
  Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_2");
  Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_4");
  Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_8");
}

bool SparcV9TargetInfo::isValidCPUName(StringRef Name) const {
  std::optional<llvm::StringMap<bool>> Features =
      llvm::Sparc::getSparcDefaultTargetFeatures(Name);
  return Features && Features->lookup("v9");
}

void SparcV9TargetInfo::fillValidCPUList(
    SmallVectorImpl<StringRef> &Values) const {
  SmallVector<StringRef> CPUs;
  llvm::Sparc::fillValidCPUList(CPUs);
  llvm::copy_if(CPUs, std::back_inserter(Values),
                [this](StringRef CPU) { return isValidCPUName(CPU); });
}

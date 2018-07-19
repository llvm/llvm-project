//===--- Sparc.cpp - Implement Sparc target feature support ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
    "r0",  "r1",  "r2",  "r3",  "r4",  "r5",  "r6",  "r7",  "r8",  "r9",  "r10",
    "r11", "r12", "r13", "r14", "r15", "r16", "r17", "r18", "r19", "r20", "r21",
    "r22", "r23", "r24", "r25", "r26", "r27", "r28", "r29", "r30", "r31"
};

ArrayRef<const char *> SparcTargetInfo::getGCCRegNames() const {
  return llvm::makeArrayRef(GCCRegNames);
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
  return llvm::makeArrayRef(GCCRegAliases);
}

bool SparcTargetInfo::hasFeature(StringRef Feature) const {
  return llvm::StringSwitch<bool>(Feature)
      .Case("softfloat", SoftFloat)
      .Case("sparc", true)
      .Default(false);
}

SparcTargetInfo::CPUKind SparcTargetInfo::getCPUKind(StringRef Name) const {
  return llvm::StringSwitch<CPUKind>(Name)
      .Case("v8", CK_V8)
      .Case("supersparc", CK_SUPERSPARC)
      .Case("sparclite", CK_SPARCLITE)
      .Case("f934", CK_F934)
      .Case("hypersparc", CK_HYPERSPARC)
      .Case("sparclite86x", CK_SPARCLITE86X)
      .Case("sparclet", CK_SPARCLET)
      .Case("tsc701", CK_TSC701)
      .Case("v9", CK_V9)
      .Case("ultrasparc", CK_ULTRASPARC)
      .Case("ultrasparc3", CK_ULTRASPARC3)
      .Case("niagara", CK_NIAGARA)
      .Case("niagara2", CK_NIAGARA2)
      .Case("niagara3", CK_NIAGARA3)
      .Case("niagara4", CK_NIAGARA4)
      .Case("ma2100", CK_MYRIAD2100)
      .Case("ma2150", CK_MYRIAD2150)
      .Case("ma2155", CK_MYRIAD2155)
      .Case("ma2450", CK_MYRIAD2450)
      .Case("ma2455", CK_MYRIAD2455)
      .Case("ma2x5x", CK_MYRIAD2x5x)
      .Case("ma2080", CK_MYRIAD2080)
      .Case("ma2085", CK_MYRIAD2085)
      .Case("ma2480", CK_MYRIAD2480)
      .Case("ma2485", CK_MYRIAD2485)
      .Case("ma2x8x", CK_MYRIAD2x8x)
      // FIXME: the myriad2[.n] spellings are obsolete,
      // but a grace period is needed to allow updating dependent builds.
      .Case("myriad2", CK_MYRIAD2x5x)
      .Case("myriad2.1", CK_MYRIAD2100)
      .Case("myriad2.2", CK_MYRIAD2x5x)
      .Case("myriad2.3", CK_MYRIAD2x8x)
      .Case("leon2", CK_LEON2)
      .Case("at697e", CK_LEON2_AT697E)
      .Case("at697f", CK_LEON2_AT697F)
      .Case("leon3", CK_LEON3)
      .Case("ut699", CK_LEON3_UT699)
      .Case("gr712rc", CK_LEON3_GR712RC)
      .Case("leon4", CK_LEON4)
      .Case("gr740", CK_LEON4_GR740)
      .Default(CK_GENERIC);
}

void SparcTargetInfo::getTargetDefines(const LangOptions &Opts,
                                       MacroBuilder &Builder) const {
  DefineStd(Builder, "sparc", Opts);
  Builder.defineMacro("__REGISTER_PREFIX__", "");

  if (SoftFloat)
    Builder.defineMacro("SOFT_FLOAT", "1");
}

void SparcV8TargetInfo::getTargetDefines(const LangOptions &Opts,
                                         MacroBuilder &Builder) const {
  SparcTargetInfo::getTargetDefines(Opts, Builder);
  switch (getCPUGeneration(CPU)) {
  case CG_V8:
    Builder.defineMacro("__sparcv8");
    if (getTriple().getOS() != llvm::Triple::Solaris)
      Builder.defineMacro("__sparcv8__");
    break;
  case CG_V9:
    Builder.defineMacro("__sparcv9");
    if (getTriple().getOS() != llvm::Triple::Solaris) {
      Builder.defineMacro("__sparcv9__");
      Builder.defineMacro("__sparc_v9__");
    }
    break;
  }
  if (getTriple().getVendor() == llvm::Triple::Myriad) {
    std::string MyriadArchValue, Myriad2Value;
    Builder.defineMacro("__sparc_v8__");
    Builder.defineMacro("__leon__");
    switch (CPU) {
    case CK_MYRIAD2100:
      MyriadArchValue = "__ma2100";
      Myriad2Value = "1";
      break;
    case CK_MYRIAD2150:
      MyriadArchValue = "__ma2150";
      Myriad2Value = "2";
      break;
    case CK_MYRIAD2155:
      MyriadArchValue = "__ma2155";
      Myriad2Value = "2";
      break;
    case CK_MYRIAD2450:
      MyriadArchValue = "__ma2450";
      Myriad2Value = "2";
      break;
    case CK_MYRIAD2455:
      MyriadArchValue = "__ma2455";
      Myriad2Value = "2";
      break;
    case CK_MYRIAD2x5x:
      Myriad2Value = "2";
      break;
    case CK_MYRIAD2080:
      MyriadArchValue = "__ma2080";
      Myriad2Value = "3";
      break;
    case CK_MYRIAD2085:
      MyriadArchValue = "__ma2085";
      Myriad2Value = "3";
      break;
    case CK_MYRIAD2480:
      MyriadArchValue = "__ma2480";
      Myriad2Value = "3";
      break;
    case CK_MYRIAD2485:
      MyriadArchValue = "__ma2485";
      Myriad2Value = "3";
      break;
    case CK_MYRIAD2x8x:
      Myriad2Value = "3";
      break;
    default:
      MyriadArchValue = "__ma2100";
      Myriad2Value = "1";
      break;
    }
    if (!MyriadArchValue.empty()) {
      Builder.defineMacro(MyriadArchValue, "1");
      Builder.defineMacro(MyriadArchValue + "__", "1");
    }
    Builder.defineMacro("__myriad2__", Myriad2Value);
    Builder.defineMacro("__myriad2", Myriad2Value);
  }
}

void SparcV9TargetInfo::getTargetDefines(const LangOptions &Opts,
                                         MacroBuilder &Builder) const {
  SparcTargetInfo::getTargetDefines(Opts, Builder);
  Builder.defineMacro("__sparcv9");
  Builder.defineMacro("__arch64__");
  // Solaris doesn't need these variants, but the BSDs do.
  if (getTriple().getOS() != llvm::Triple::Solaris) {
    Builder.defineMacro("__sparc64__");
    Builder.defineMacro("__sparc_v9__");
    Builder.defineMacro("__sparcv9__");
  }
}

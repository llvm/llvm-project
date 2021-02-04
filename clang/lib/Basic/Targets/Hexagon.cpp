//===--- Hexagon.cpp - Implement Hexagon target feature support -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements Hexagon TargetInfo objects.
//
//===----------------------------------------------------------------------===//

#include "Hexagon.h"
#include "Targets.h"
#include "clang/Basic/MacroBuilder.h"
#include "clang/Basic/TargetBuiltins.h"
#include "llvm/ADT/StringSwitch.h"

using namespace clang;
using namespace clang::targets;

void HexagonTargetInfo::getTargetDefines(const LangOptions &Opts,
                                         MacroBuilder &Builder) const {
  Builder.defineMacro("__qdsp6__", "1");
  Builder.defineMacro("__hexagon__", "1");

  Builder.defineMacro("__ELF__");

  // The macro __HVXDBL__ is deprecated.
  bool DefineHvxDbl = false;

  if (CPU == "hexagonv5") {
    Builder.defineMacro("__HEXAGON_V5__");
    Builder.defineMacro("__HEXAGON_ARCH__", "5");
    if (Opts.HexagonQdsp6Compat) {
      Builder.defineMacro("__QDSP6_V5__");
      Builder.defineMacro("__QDSP6_ARCH__", "5");
    }
  } else if (CPU == "hexagonv55") {
    Builder.defineMacro("__HEXAGON_V55__");
    Builder.defineMacro("__HEXAGON_ARCH__", "55");
    Builder.defineMacro("__QDSP6_V55__");
    Builder.defineMacro("__QDSP6_ARCH__", "55");
  } else if (CPU == "hexagonv60") {
    DefineHvxDbl = true;
    Builder.defineMacro("__HEXAGON_V60__");
    Builder.defineMacro("__HEXAGON_ARCH__", "60");
    Builder.defineMacro("__QDSP6_V60__");
    Builder.defineMacro("__QDSP6_ARCH__", "60");
  } else if (CPU == "hexagonv62") {
    DefineHvxDbl = true;
    Builder.defineMacro("__HEXAGON_V62__");
    Builder.defineMacro("__HEXAGON_ARCH__", "62");
  } else if (CPU == "hexagonv65") {
    DefineHvxDbl = true;
    Builder.defineMacro("__HEXAGON_V65__");
    Builder.defineMacro("__HEXAGON_ARCH__", "65");
  } else if (CPU == "hexagonv66") {
    DefineHvxDbl = true;
    Builder.defineMacro("__HEXAGON_V66__");
    Builder.defineMacro("__HEXAGON_ARCH__", "66");
  } else if (CPU == "hexagonv67") {
    Builder.defineMacro("__HEXAGON_V67__");
    Builder.defineMacro("__HEXAGON_ARCH__", "67");
  } else if (CPU == "hexagonv67t") {
    Builder.defineMacro("__HEXAGON_V67T__");
    Builder.defineMacro("__HEXAGON_ARCH__", "67");
  } else if (CPU == "hexagonv68") {
    Builder.defineMacro("__HEXAGON_V68__");
    Builder.defineMacro("__HEXAGON_ARCH__", "68");
  }

  if (hasFeature("hvx-length64b")) {
    Builder.defineMacro("__HVX__");
    Builder.defineMacro("__HVX_ARCH__", HVXVersion);
    Builder.defineMacro("__HVX_LENGTH__", "64");
  }

  if (hasFeature("hvx-length128b")) {
    Builder.defineMacro("__HVX__");
    Builder.defineMacro("__HVX_ARCH__", HVXVersion);
    Builder.defineMacro("__HVX_LENGTH__", "128");
    if (DefineHvxDbl)
      Builder.defineMacro("__HVXDBL__");
  }

  if (hasFeature("audio")) {
    Builder.defineMacro("__HEXAGON_AUDIO__");
  }

  std::string NumPhySlots = isTinyCore() ? "3" : "4";
  Builder.defineMacro("__HEXAGON_PHYSICAL_SLOTS__", NumPhySlots);
}

bool HexagonTargetInfo::initFeatureMap(
    llvm::StringMap<bool> &Features, DiagnosticsEngine &Diags, StringRef CPU,
    const std::vector<std::string> &FeaturesVec) const {
  if (isTinyCore())
    Features["audio"] = true;

  StringRef CPUFeature = CPU;
  CPUFeature.consume_front("hexagon");
  CPUFeature.consume_back("t");
  if (!CPUFeature.empty())
    Features[CPUFeature] = true;

  Features["long-calls"] = false;

  return TargetInfo::initFeatureMap(Features, Diags, CPU, FeaturesVec);
}

bool HexagonTargetInfo::handleTargetFeatures(std::vector<std::string> &Features,
                                             DiagnosticsEngine &Diags) {
  for (auto &F : Features) {
    if (F == "+hvx-length64b")
      HasHVX = HasHVX64B = true;
    else if (F == "+hvx-length128b")
      HasHVX = HasHVX128B = true;
    else if (F.find("+hvxv") != std::string::npos) {
      HasHVX = true;
      HVXVersion = F.substr(std::string("+hvxv").length());
    } else if (F == "-hvx")
      HasHVX = HasHVX64B = HasHVX128B = false;
    else if (F == "+long-calls")
      UseLongCalls = true;
    else if (F == "-long-calls")
      UseLongCalls = false;
    else if (F == "+audio")
      HasAudio = true;
  }
  return true;
}

const char *const HexagonTargetInfo::GCCRegNames[] = {
    "r0",  "r1",  "r2",  "r3",  "r4",  "r5",  "r6",  "r7",  "r8",
    "r9",  "r10", "r11", "r12", "r13", "r14", "r15", "r16", "r17",
    "r18", "r19", "r20", "r21", "r22", "r23", "r24", "r25", "r26",
    "r27", "r28", "r29", "r30", "r31", "p0",  "p1",  "p2",  "p3",
    "sa0", "lc0", "sa1", "lc1", "m0",  "m1",  "usr", "ugp",
    "r1:0", "r3:2", "r5:4", "r7:6", "r9:8", "r11:10", "r13:12", "r15:14",
    "r17:16", "r19:18", "r21:20", "r23:22", "r25:24", "r27:26", "r29:28",
    "r31:30"
};

ArrayRef<const char *> HexagonTargetInfo::getGCCRegNames() const {
  return llvm::makeArrayRef(GCCRegNames);
}

const TargetInfo::GCCRegAlias HexagonTargetInfo::GCCRegAliases[] = {
    {{"sp"}, "r29"},
    {{"fp"}, "r30"},
    {{"lr"}, "r31"},
};

ArrayRef<TargetInfo::GCCRegAlias> HexagonTargetInfo::getGCCRegAliases() const {
  return llvm::makeArrayRef(GCCRegAliases);
}

const Builtin::Info HexagonTargetInfo::BuiltinInfo[] = {
#define BUILTIN(ID, TYPE, ATTRS)                                               \
  {#ID, TYPE, ATTRS, nullptr, ALL_LANGUAGES, nullptr},
#define LIBBUILTIN(ID, TYPE, ATTRS, HEADER)                                    \
  {#ID, TYPE, ATTRS, HEADER, ALL_LANGUAGES, nullptr},
#define TARGET_BUILTIN(ID, TYPE, ATTRS, FEATURE)                               \
  {#ID, TYPE, ATTRS, nullptr, ALL_LANGUAGES, FEATURE},
#include "clang/Basic/BuiltinsHexagon.def"
};

bool HexagonTargetInfo::hasFeature(StringRef Feature) const {
  std::string VS = "hvxv" + HVXVersion;
  if (Feature == VS)
    return true;

  return llvm::StringSwitch<bool>(Feature)
      .Case("hexagon", true)
      .Case("hvx", HasHVX)
      .Case("hvx-length64b", HasHVX64B)
      .Case("hvx-length128b", HasHVX128B)
      .Case("long-calls", UseLongCalls)
      .Case("audio", HasAudio)
      .Default(false);
}

struct CPUSuffix {
  llvm::StringLiteral Name;
  llvm::StringLiteral Suffix;
};

static constexpr CPUSuffix Suffixes[] = {
    {{"hexagonv5"},  {"5"}},  {{"hexagonv55"},  {"55"}},
    {{"hexagonv60"}, {"60"}}, {{"hexagonv62"},  {"62"}},
    {{"hexagonv65"}, {"65"}}, {{"hexagonv66"},  {"66"}},
    {{"hexagonv67"}, {"67"}}, {{"hexagonv67t"}, {"67t"}},
    {{"hexagonv68"}, {"68"}},
};

const char *HexagonTargetInfo::getHexagonCPUSuffix(StringRef Name) {
  const CPUSuffix *Item = llvm::find_if(
      Suffixes, [Name](const CPUSuffix &S) { return S.Name == Name; });
  if (Item == std::end(Suffixes))
    return nullptr;
  return Item->Suffix.data();
}

void HexagonTargetInfo::fillValidCPUList(
    SmallVectorImpl<StringRef> &Values) const {
  for (const CPUSuffix &Suffix : Suffixes)
    Values.push_back(Suffix.Name);
}

ArrayRef<Builtin::Info> HexagonTargetInfo::getTargetBuiltins() const {
  return llvm::makeArrayRef(BuiltinInfo, clang::Hexagon::LastTSBuiltin -
                                             Builtin::FirstTSBuiltin);
}

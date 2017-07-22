//===--- Hexagon.cpp - Implement Hexagon target feature support -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

  if (CPU == "hexagonv4") {
    Builder.defineMacro("__HEXAGON_V4__");
    Builder.defineMacro("__HEXAGON_ARCH__", "4");
    if (Opts.HexagonQdsp6Compat) {
      Builder.defineMacro("__QDSP6_V4__");
      Builder.defineMacro("__QDSP6_ARCH__", "4");
    }
  } else if (CPU == "hexagonv5") {
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
    Builder.defineMacro("__HEXAGON_V60__");
    Builder.defineMacro("__HEXAGON_ARCH__", "60");
    Builder.defineMacro("__QDSP6_V60__");
    Builder.defineMacro("__QDSP6_ARCH__", "60");
  } else if (CPU == "hexagonv62") {
    Builder.defineMacro("__HEXAGON_V62__");
    Builder.defineMacro("__HEXAGON_ARCH__", "62");
  }

  if (hasFeature("hvx")) {
    Builder.defineMacro("__HVX__");
    if (hasFeature("hvx-double"))
      Builder.defineMacro("__HVXDBL__");
  }
}

bool HexagonTargetInfo::initFeatureMap(
    llvm::StringMap<bool> &Features, DiagnosticsEngine &Diags, StringRef CPU,
    const std::vector<std::string> &FeaturesVec) const {
  // Default for v60: -hvx, -hvx-double.
  Features["hvx"] = false;
  Features["hvx-double"] = false;
  Features["long-calls"] = false;

  return TargetInfo::initFeatureMap(Features, Diags, CPU, FeaturesVec);
}

bool HexagonTargetInfo::handleTargetFeatures(std::vector<std::string> &Features,
                                             DiagnosticsEngine &Diags) {
  for (auto &F : Features) {
    if (F == "+hvx")
      HasHVX = true;
    else if (F == "-hvx")
      HasHVX = HasHVXDouble = false;
    else if (F == "+hvx-double")
      HasHVX = HasHVXDouble = true;
    else if (F == "-hvx-double")
      HasHVXDouble = false;

    if (F == "+long-calls")
      UseLongCalls = true;
    else if (F == "-long-calls")
      UseLongCalls = false;
  }
  return true;
}

void HexagonTargetInfo::setFeatureEnabled(llvm::StringMap<bool> &Features,
                                          StringRef Name, bool Enabled) const {
  if (Enabled) {
    if (Name == "hvx-double")
      Features["hvx"] = true;
  } else {
    if (Name == "hvx")
      Features["hvx-double"] = false;
  }
  Features[Name] = Enabled;
}

const char *const HexagonTargetInfo::GCCRegNames[] = {
    "r0",  "r1",  "r2",  "r3",  "r4",  "r5",  "r6",  "r7",  "r8",
    "r9",  "r10", "r11", "r12", "r13", "r14", "r15", "r16", "r17",
    "r18", "r19", "r20", "r21", "r22", "r23", "r24", "r25", "r26",
    "r27", "r28", "r29", "r30", "r31", "p0",  "p1",  "p2",  "p3",
    "sa0", "lc0", "sa1", "lc1", "m0",  "m1",  "usr", "ugp"
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
#include "clang/Basic/BuiltinsHexagon.def"
};

bool HexagonTargetInfo::hasFeature(StringRef Feature) const {
  return llvm::StringSwitch<bool>(Feature)
      .Case("hexagon", true)
      .Case("hvx", HasHVX)
      .Case("hvx-double", HasHVXDouble)
      .Case("long-calls", UseLongCalls)
      .Default(false);
}

const char *HexagonTargetInfo::getHexagonCPUSuffix(StringRef Name) {
  return llvm::StringSwitch<const char *>(Name)
      .Case("hexagonv4", "4")
      .Case("hexagonv5", "5")
      .Case("hexagonv55", "55")
      .Case("hexagonv60", "60")
      .Case("hexagonv62", "62")
      .Default(nullptr);
}

ArrayRef<Builtin::Info> HexagonTargetInfo::getTargetBuiltins() const {
  return llvm::makeArrayRef(BuiltinInfo, clang::Hexagon::LastTSBuiltin -
                                             Builtin::FirstTSBuiltin);
}

//===- Sanitizers.cpp - C Language Family Language Options ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the classes from Sanitizers.h
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/Sanitizers.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cmath>
#include <optional>

using namespace clang;

static const double SanitizerMaskCutoffsEps = 0.000000001f;

void SanitizerMaskCutoffs::set(SanitizerMask K, double V) {
  if (V < SanitizerMaskCutoffsEps && Cutoffs.empty())
    return;
  for (unsigned int i = 0; i < SanitizerKind::SO_Count; ++i)
    if (K & SanitizerMask::bitPosToMask(i)) {
      Cutoffs.resize(SanitizerKind::SO_Count);
      Cutoffs[i] = V;
    }
}

std::optional<double> SanitizerMaskCutoffs::operator[](unsigned Kind) const {
  if (Cutoffs.empty() || Cutoffs[Kind] < SanitizerMaskCutoffsEps)
    return std::nullopt;

  return Cutoffs[Kind];
}

void SanitizerMaskCutoffs::clear(SanitizerMask K) { set(K, 0); }

std::optional<std::vector<unsigned>>
SanitizerMaskCutoffs::getAllScaled(unsigned ScalingFactor) const {
  std::vector<unsigned> ScaledCutoffs;

  bool AnyCutoff = false;
  for (unsigned int i = 0; i < SanitizerKind::SO_Count; ++i) {
    auto C = (*this)[i];
    if (C.has_value()) {
      ScaledCutoffs.push_back(lround(std::clamp(*C, 0.0, 1.0) * ScalingFactor));
      AnyCutoff = true;
    } else {
      ScaledCutoffs.push_back(0);
    }
  }

  if (AnyCutoff)
    return ScaledCutoffs;

  return std::nullopt;
}

// Once LLVM switches to C++17, the constexpr variables can be inline and we
// won't need this.
#define SANITIZER(NAME, ID) constexpr SanitizerMask SanitizerKind::ID;
#define SANITIZER_GROUP(NAME, ID, ALIAS)                                       \
  constexpr SanitizerMask SanitizerKind::ID;                                   \
  constexpr SanitizerMask SanitizerKind::ID##Group;
#include "clang/Basic/Sanitizers.def"

SanitizerMask clang::parseSanitizerValue(StringRef Value, bool AllowGroups) {
  SanitizerMask ParsedKind = llvm::StringSwitch<SanitizerMask>(Value)
#define SANITIZER(NAME, ID) .Case(NAME, SanitizerKind::ID)
#define SANITIZER_GROUP(NAME, ID, ALIAS)                                       \
  .Case(NAME, AllowGroups ? SanitizerKind::ID##Group : SanitizerMask())
#include "clang/Basic/Sanitizers.def"
    .Default(SanitizerMask());
  return ParsedKind;
}

bool clang::parseSanitizerWeightedValue(StringRef Value, bool AllowGroups,
                                        SanitizerMaskCutoffs &Cutoffs) {
  SanitizerMask ParsedKind = llvm::StringSwitch<SanitizerMask>(Value)
#define SANITIZER(NAME, ID) .StartsWith(NAME "=", SanitizerKind::ID)
#define SANITIZER_GROUP(NAME, ID, ALIAS)                                       \
  .StartsWith(NAME "=",                                                        \
              AllowGroups ? SanitizerKind::ID##Group : SanitizerMask())
#include "clang/Basic/Sanitizers.def"
                                 .Default(SanitizerMask());

  if (!ParsedKind)
    return false;
  auto [N, W] = Value.split('=');
  double A;
  if (W.getAsDouble(A))
    return false;
  A = std::clamp(A, 0.0, 1.0);
  // AllowGroups is already taken into account for ParsedKind,
  // hence we unconditionally expandSanitizerGroups.
  Cutoffs.set(expandSanitizerGroups(ParsedKind), A);
  return true;
}

void clang::serializeSanitizerSet(SanitizerSet Set,
                                  SmallVectorImpl<StringRef> &Values) {
#define SANITIZER(NAME, ID)                                                    \
  if (Set.has(SanitizerKind::ID))                                              \
    Values.push_back(NAME);
#include "clang/Basic/Sanitizers.def"
}

void clang::serializeSanitizerMaskCutoffs(
    const SanitizerMaskCutoffs &Cutoffs, SmallVectorImpl<std::string> &Values) {
#define SANITIZER(NAME, ID)                                                    \
  if (auto C = Cutoffs[SanitizerKind::SO_##ID]) {                              \
    std::string Str;                                                           \
    llvm::raw_string_ostream OS(Str);                                          \
    OS << NAME "=" << llvm::format("%.8f", *C);                                \
    Values.emplace_back(StringRef(Str).rtrim('0'));                            \
  }
#include "clang/Basic/Sanitizers.def"
}

SanitizerMask clang::expandSanitizerGroups(SanitizerMask Kinds) {
#define SANITIZER(NAME, ID)
#define SANITIZER_GROUP(NAME, ID, ALIAS)                                       \
  if (Kinds & SanitizerKind::ID##Group)                                        \
    Kinds |= SanitizerKind::ID;
#include "clang/Basic/Sanitizers.def"
  return Kinds;
}

llvm::hash_code SanitizerMask::hash_value() const {
  return llvm::hash_combine_range(&maskLoToHigh[0], &maskLoToHigh[kNumElem]);
}

namespace clang {
unsigned SanitizerMask::countPopulation() const {
  unsigned total = 0;
  for (const auto &Val : maskLoToHigh)
    total += llvm::popcount(Val);
  return total;
}

llvm::hash_code hash_value(const clang::SanitizerMask &Arg) {
  return Arg.hash_value();
}

StringRef AsanDtorKindToString(llvm::AsanDtorKind kind) {
  switch (kind) {
  case llvm::AsanDtorKind::None:
    return "none";
  case llvm::AsanDtorKind::Global:
    return "global";
  case llvm::AsanDtorKind::Invalid:
    return "invalid";
  }
  return "invalid";
}

llvm::AsanDtorKind AsanDtorKindFromString(StringRef kindStr) {
  return llvm::StringSwitch<llvm::AsanDtorKind>(kindStr)
      .Case("none", llvm::AsanDtorKind::None)
      .Case("global", llvm::AsanDtorKind::Global)
      .Default(llvm::AsanDtorKind::Invalid);
}

StringRef AsanDetectStackUseAfterReturnModeToString(
    llvm::AsanDetectStackUseAfterReturnMode mode) {
  switch (mode) {
  case llvm::AsanDetectStackUseAfterReturnMode::Always:
    return "always";
  case llvm::AsanDetectStackUseAfterReturnMode::Runtime:
    return "runtime";
  case llvm::AsanDetectStackUseAfterReturnMode::Never:
    return "never";
  case llvm::AsanDetectStackUseAfterReturnMode::Invalid:
    return "invalid";
  }
  return "invalid";
}

llvm::AsanDetectStackUseAfterReturnMode
AsanDetectStackUseAfterReturnModeFromString(StringRef modeStr) {
  return llvm::StringSwitch<llvm::AsanDetectStackUseAfterReturnMode>(modeStr)
      .Case("always", llvm::AsanDetectStackUseAfterReturnMode::Always)
      .Case("runtime", llvm::AsanDetectStackUseAfterReturnMode::Runtime)
      .Case("never", llvm::AsanDetectStackUseAfterReturnMode::Never)
      .Default(llvm::AsanDetectStackUseAfterReturnMode::Invalid);
}

} // namespace clang

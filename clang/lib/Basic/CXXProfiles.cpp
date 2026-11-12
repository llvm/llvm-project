//===--- CXXProfiles.cpp - C++ profiles registry --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/CXXProfiles.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"

using namespace clang;

StringRef clang::getCXXProfileName(CXXProfileID Profile) {
  switch (Profile) {
#define CXX_PROFILE(ID, NAME)                                                  \
  case CXXProfileID::ID:                                                       \
    return NAME;
#include "clang/Basic/CXXProfiles.def"
  }
  llvm_unreachable("unknown profile");
}

StringRef clang::getCXXProfileRuleName(CXXProfileRuleID Rule) {
  switch (Rule) {
#define CXX_PROFILE_RULE(PROFILE, ID, NAME)                                    \
  case CXXProfileRuleID::ID:                                                   \
    return NAME;
#include "clang/Basic/CXXProfiles.def"
  }
  llvm_unreachable("unknown profile rule");
}

CXXProfileID clang::getCXXProfileOfRule(CXXProfileRuleID Rule) {
  switch (Rule) {
#define CXX_PROFILE_RULE(PROFILE, ID, NAME)                                    \
  case CXXProfileRuleID::ID:                                                   \
    return CXXProfileID::PROFILE;
#include "clang/Basic/CXXProfiles.def"
  }
  llvm_unreachable("unknown profile rule");
}

std::optional<CXXProfileID> clang::lookupCXXProfile(StringRef Name) {
  return llvm::StringSwitch<std::optional<CXXProfileID>>(Name)
#define CXX_PROFILE(ID, NAME) .Case(NAME, CXXProfileID::ID)
#include "clang/Basic/CXXProfiles.def"
      .Default(std::nullopt);
}

std::optional<CXXProfileRuleID>
clang::lookupCXXProfileRule(CXXProfileID Profile, StringRef Name) {
#define CXX_PROFILE_RULE(PROFILE, ID, NAME)                                    \
  if (Profile == CXXProfileID::PROFILE && Name == NAME)                        \
    return CXXProfileRuleID::ID;
#include "clang/Basic/CXXProfiles.def"
  return std::nullopt;
}

StringRef clang::getCXXProfileDesignatorName(StringRef Designator) {
  return Designator.take_until([](char C) { return C == '('; }).rtrim();
}

//===--- CXXProfiles.h - C++ profiles registry ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines the registry of C++ profiles (P3589) known to Clang.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_CXXPROFILES_H
#define LLVM_CLANG_BASIC_CXXPROFILES_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include <optional>

namespace clang {

/// A C++ profile known to Clang.
enum class CXXProfileID {
#define CXX_PROFILE(ID, NAME) ID,
#include "clang/Basic/CXXProfiles.def"
};

/// A rule of a C++ profile known to Clang.
enum class CXXProfileRuleID {
#define CXX_PROFILE_RULE(PROFILE, ID, NAME) ID,
#include "clang/Basic/CXXProfiles.def"
};

/// Returns the name of \p Profile as written in a profile designator.
StringRef getCXXProfileName(CXXProfileID Profile);

/// Returns the name of \p Rule as written in a 'rule:' argument.
StringRef getCXXProfileRuleName(CXXProfileRuleID Rule);

/// Returns the profile that \p Rule belongs to.
CXXProfileID getCXXProfileOfRule(CXXProfileRuleID Rule);

/// Looks up a profile by name, e.g. "std::casting".
std::optional<CXXProfileID> lookupCXXProfile(StringRef Name);

/// Looks up a rule of \p Profile by name, e.g. "reinterpret_cast".
std::optional<CXXProfileRuleID> lookupCXXProfileRule(CXXProfileID Profile,
                                                     StringRef Name);

/// Returns the profile name of a profile designator, i.e. the designator
/// without its argument list. For example, returns "acme::hardened" for
/// "acme::hardened(level: 3)".
StringRef getCXXProfileDesignatorName(StringRef Designator);

} // namespace clang

#endif // LLVM_CLANG_BASIC_CXXPROFILES_H

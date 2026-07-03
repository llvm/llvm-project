//===--- Profiles.h - C++ profiles framework helpers -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_PROFILES_H
#define LLVM_CLANG_BASIC_PROFILES_H

#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include <string>

namespace clang::profiles {

enum class ProfileArgumentKind : unsigned {
  Positional = 0,
  Named = 1,
};

struct ProfileArgument {
  llvm::StringRef Key;
  llvm::StringRef Value;
  ProfileArgumentKind Kind = ProfileArgumentKind::Positional;
  SourceRange Range;

  bool isNamed() const { return Kind == ProfileArgumentKind::Named; }
};

/// A profile enforced by [[profiles::enforce]]: the profile name plus the
/// canonical spelling of the designator that enforced it (P3589R2 [decl.attr
/// .enforce]p3 compares repeated enforcements by their spelling). Shared by
/// Sema's enforcement list, Module's exported enforcement set, and the
/// serialized PCH record.
struct EnforcedProfile {
  std::string ProfileName;
  std::string Designator;
};

/// The canonical spelling of a profile argument: the value token for a
/// positional argument, "key : value" for a named one. Enforcement identity
/// (P3589R2 [decl.attr.enforce]p3) compares designators by this spelling.
/// A template over the argument representation so it serves both
/// ProfileArgument and the parser's owning-string argument type.
template <typename ArgumentT>
std::string getCanonicalProfileArgumentSpelling(const ArgumentT &Argument) {
  if (!Argument.isNamed())
    return std::string(Argument.Value);
  return (llvm::Twine(Argument.Key) + " : " + Argument.Value).str();
}

} // namespace clang::profiles

#endif // LLVM_CLANG_BASIC_PROFILES_H

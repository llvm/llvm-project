//===- bolt/Utils/NameResolver.h - Names deduplication helper ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helper class for names deduplication.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_UTILS_NAME_RESOLVER_H
#define BOLT_UTILS_NAME_RESOLVER_H

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"

namespace llvm {
namespace bolt {

class NameResolver {
  /// Track the number of duplicate names.
  StringMap<uint64_t> Counters;

  /// Character guaranteed not to be used by any "native" name passed to
  /// uniquify() function.
  static constexpr char Sep = '/';

public:
  /// Return the number of uniquified versions of a given \p Name.
  uint64_t getUniquifiedNameCount(StringRef Name) const {
    if (Counters.contains(Name))
      return Counters.at(Name);
    return 0;
  }

  /// Return unique version of the \p Name in the form "Name<Sep><ID>".
  std::string getUniqueName(StringRef Name, const uint64_t ID) const {
    return (Name + Twine(Sep) + Twine(ID)).str();
  }

  /// Register new version of \p Name and return unique version in the form
  /// "Name<Sep><Number>".
  std::string uniquify(StringRef Name) {
    const uint64_t ID = ++Counters[Name];
    return getUniqueName(Name, ID);
  }

  /// For uniquified \p Name, return the original form (that may no longer be
  /// unique).
  static StringRef restore(StringRef Name) {
    return Name.substr(0, Name.find_first_of(Sep));
  }

  /// Append \p Suffix to the original string in \p UniqueName  preserving the
  /// deduplication form. E.g. append("Name<Sep>42", "Suffix") will return
  /// "NameSuffix<Sep>42".
  static std::string append(StringRef UniqueName, StringRef Suffix) {
    StringRef LHS, RHS;
    std::tie(LHS, RHS) = UniqueName.split(Sep);
    return (LHS + Suffix + Twine(Sep) + RHS).str();
  }
};

} // namespace bolt
} // namespace llvm

#endif

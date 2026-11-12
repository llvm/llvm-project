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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/xxhash.h"

namespace llvm {
namespace bolt {

class NameResolver {
  /// Track the number of duplicate names, keyed by a 128-bit hash of the name
  /// rather than by the name itself. Storing hashes instead of the full strings
  /// avoids duplicating potentially large (mangled) symbol names, which is a
  /// significant source of memory use while processing the symbol table. Using
  /// a 128-bit hash makes collisions effectively impossible, so the counts (and
  /// therefore the generated unique names) are identical to a string-keyed map
  /// and remain reproducible to match profile (fdata) names.
  DenseMap<std::pair<uint64_t, uint64_t>, uint64_t> Counters;

  /// Character guaranteed not to be used by any "native" name passed to
  /// uniquify() function.
  static constexpr char Sep = '/';

  /// Return the map key used to track occurrences of \p Name.
  static std::pair<uint64_t, uint64_t> getKey(StringRef Name) {
    const XXH128_hash_t Hash = llvm::xxh3_128bits(
        reinterpret_cast<const uint8_t *>(Name.data()), Name.size());
    return {Hash.low64, Hash.high64};
  }

public:
  /// Return the number of uniquified versions of a given \p Name.
  uint64_t getUniquifiedNameCount(StringRef Name) const {
    return Counters.lookup(getKey(Name));
  }

  /// Return unique version of the \p Name in the form "Name<Sep><ID>".
  std::string getUniqueName(StringRef Name, const uint64_t ID) const {
    return (Name + Twine(Sep) + Twine(ID)).str();
  }

  /// Register new version of \p Name and return unique version in the form
  /// "Name<Sep><Number>".
  std::string uniquify(StringRef Name) {
    const uint64_t ID = ++Counters[getKey(Name)];
    return getUniqueName(Name, ID);
  }

  /// Release the memory used to track name occurrences. Call once no more names
  /// need to be uniquified (e.g. after file object discovery is complete).
  void clear() { Counters.clear(); }

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

  // Drops the suffix that describes the function's number of names.
  static StringRef dropNumNames(StringRef Name) {
    const size_t Pos = Name.find("(*");
    return Pos != StringRef::npos ? Name.substr(0, Pos) : Name;
  }
};

} // namespace bolt
} // namespace llvm

#endif

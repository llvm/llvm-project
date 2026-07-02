//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ALIASES_CLANGTIDYALIASES_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ALIASES_CLANGTIDYALIASES_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h" // IWYU pragma: keep
#include "llvm/ADT/StringRef.h"   // IWYU pragma: keep

namespace clang::tidy {

class ClangTidyAliases {
public:
  /// Alias never expires.
  static constexpr int NoExpiration = -1;

  /// An entry in the alias table.
  /// ExpirationVersion is the LLVM major release at which the alias expires.
  /// Use NoExpiration for permanent aliases.
  struct Entry {
    StringRef Alias;
    StringRef Canonical;
    int ExpirationVersion;
  };

  /// Look up the canonical name for an alias.
  static StringRef getCanonicalForAlias(StringRef Alias);

  /// Look up aliases for a canonical name (may return multiple).
  static const SmallVector<StringRef> &
  getAliasesForCanonical(StringRef Canonical);

  /// Get a reference to the full alias table for iteration.
  static ArrayRef<Entry> getReference();

  /// Returns true if the given entry is active (not expired) in the current
  /// LLVM version.
  static bool isActive(const Entry &E);

  /// Filtered range type for iterating only active (non-expired) entries.
  using ActiveRange = llvm::iterator_range<
      llvm::filter_iterator<const Entry *, bool (*)(const Entry &)>>;

  /// Get only active (non-expired) alias entries.
  static ActiveRange activeEntries();
};

} // namespace clang::tidy

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ALIASES_CLANGTIDYALIASES_H

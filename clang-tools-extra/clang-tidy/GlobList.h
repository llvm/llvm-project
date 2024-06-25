//===--- GlobList.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GLOBLIST_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GLOBLIST_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Regex.h"

namespace clang::tidy {

/// Read-only set of strings represented as a list of positive and negative
/// globs.
///
/// Positive globs add all matched strings to the set, negative globs remove
/// them in the order of appearance in the list.
class GlobList {
public:
  virtual ~GlobList() = default;

  /// \p Globs is a comma-separated list of globs (only the '*' metacharacter is
  /// supported) with an optional '-' prefix to denote exclusion.
  ///
  /// An empty \p Globs string is interpreted as one glob that matches an empty
  /// string.
  ///
  /// \p KeepNegativeGlobs a bool flag indicating whether to keep negative
  /// globs from \p Globs or not. When false, negative globs are simply ignored.
  GlobList(StringRef Globs, bool KeepNegativeGlobs = true);

  /// Returns \c true if the pattern matches \p S. The result is the last
  /// matching glob's Positive flag.
  virtual bool contains(StringRef S) const;

private:
  struct GlobListItem {
    bool IsPositive;
    llvm::Regex Regex;
    llvm::StringRef Text;
  };
  SmallVector<GlobListItem, 0> Items;

public:
  const SmallVectorImpl<GlobListItem> &getItems() const { return Items; };
};

/// A \p GlobList that caches search results, so that search is performed only
/// once for the same query.
class CachedGlobList final : public GlobList {
public:
  using GlobList::GlobList;

  /// \see GlobList::contains
  bool contains(StringRef S) const override;

private:
  mutable llvm::StringMap<bool> Cache;
};

} // namespace clang::tidy

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GLOBLIST_H

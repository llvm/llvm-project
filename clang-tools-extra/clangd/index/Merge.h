//===--- Merge.h -------------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_MERGE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_MERGE_H

#include "Index.h"

namespace clang {
namespace clangd {

// Merge symbols L and R, preferring data from L in case of conflict.
// The two symbols must have the same ID.
// Returned symbol may contain data owned by either source.
Symbol mergeSymbol(const Symbol &L, const Symbol &R);

// MergedIndex is a composite index based on two provided Indexes:
//  - the Dynamic index covers few files, but is relatively up-to-date.
//  - the Static index covers a bigger set of files, but is relatively stale.
// The returned index attempts to combine results, and avoid duplicates.
//
// FIXME: We don't have a mechanism in Index to track deleted symbols and
// refs in dirty files, so the merged index may return stale symbols
// and refs from Static index.
class MergedIndex : public SymbolIndex {
  const SymbolIndex *Dynamic, *Static;

public:
  // The constructor does not access the symbols.
  // It's safe to inherit from this class and pass pointers to derived members.
  MergedIndex(const SymbolIndex *Dynamic, const SymbolIndex *Static)
      : Dynamic(Dynamic), Static(Static) {}

  bool fuzzyFind(const FuzzyFindRequest &,
                 llvm::function_ref<void(const Symbol &)>) const override;
  void lookup(const LookupRequest &,
              llvm::function_ref<void(const Symbol &)>) const override;
  void refs(const RefsRequest &,
            llvm::function_ref<void(const Ref &)>) const override;
  size_t estimateMemoryUsage() const override {
    return Dynamic->estimateMemoryUsage() + Static->estimateMemoryUsage();
  }
};

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_MERGE_H

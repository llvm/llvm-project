//===--- Merge.h -------------------------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

// mergeIndex returns a composite index based on two provided Indexes:
//  - the Dynamic index covers few files, but is relatively up-to-date.
//  - the Static index covers a bigger set of files, but is relatively stale.
// The returned index attempts to combine results, and avoid duplicates.
//
// FIXME: We don't have a mechanism in Index to track deleted symbols and
// refs in dirty files, so the merged index may return stale symbols
// and refs from Static index.
std::unique_ptr<SymbolIndex> mergeIndex(const SymbolIndex *Dynamic,
                                        const SymbolIndex *Static);

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_MERGE_H

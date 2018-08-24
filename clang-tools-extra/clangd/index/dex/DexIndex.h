//===--- DexIndex.h - Dex Symbol Index Implementation -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines Dex - a symbol index implementation based on query iterators
// over symbol tokens, such as fuzzy matching trigrams, scopes, types, etc.
// While consuming more memory and having longer build stage due to
// preprocessing, Dex will have substantially lower latency. It will also allow
// efficient symbol searching which is crucial for operations like code
// completion, and can be very important for a number of different code
// transformations which will be eventually supported by Clangd.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_DEX_DEXINDEX_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_DEX_DEXINDEX_H

#include "../Index.h"
#include "../MemIndex.h"
#include "Iterator.h"
#include "Token.h"
#include "Trigram.h"
#include <mutex>

namespace clang {
namespace clangd {
namespace dex {

/// In-memory Dex trigram-based index implementation.
// FIXME(kbobyrev): Introduce serialization and deserialization of the symbol
// index so that it can be loaded from the disk. Since static index is not
// changed frequently, it's safe to assume that it has to be built only once
// (when the clangd process starts). Therefore, it can be easier to store built
// index on disk and then load it if available.
class DexIndex : public SymbolIndex {
public:
  /// \brief (Re-)Build index for `Symbols`. All symbol pointers must remain
  /// accessible as long as `Symbols` is kept alive.
  void build(std::shared_ptr<std::vector<const Symbol *>> Symbols);

  /// \brief Build index from a symbol slab.
  static std::unique_ptr<SymbolIndex> build(SymbolSlab Slab);

  bool
  fuzzyFind(const FuzzyFindRequest &Req,
            llvm::function_ref<void(const Symbol &)> Callback) const override;

  void lookup(const LookupRequest &Req,
              llvm::function_ref<void(const Symbol &)> Callback) const override;

  void findOccurrences(const OccurrencesRequest &Req,
                       llvm::function_ref<void(const SymbolOccurrence &)>
                           Callback) const override;

  size_t estimateMemoryUsage() const override;

private:

  mutable std::mutex Mutex;

  std::shared_ptr<std::vector<const Symbol *>> Symbols /*GUARDED_BY(Mutex)*/;
  llvm::DenseMap<SymbolID, const Symbol *> LookupTable /*GUARDED_BY(Mutex)*/;
  llvm::DenseMap<const Symbol *, float> SymbolQuality /*GUARDED_BY(Mutex)*/;
  // Inverted index is a mapping from the search token to the posting list,
  // which contains all items which can be characterized by such search token.
  // For example, if the search token is scope "std::", the corresponding
  // posting list would contain all indices of symbols defined in namespace std.
  // Inverted index is used to retrieve posting lists which are processed during
  // the fuzzyFind process.
  llvm::DenseMap<Token, PostingList> InvertedIndex /*GUARDED_BY(Mutex)*/;
};

} // namespace dex
} // namespace clangd
} // namespace clang

#endif

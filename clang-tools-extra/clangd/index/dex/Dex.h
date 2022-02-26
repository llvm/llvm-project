//===--- Dex.h - Dex Symbol Index Implementation ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This defines Dex - a symbol index implementation based on query iterators
/// over symbol tokens, such as fuzzy matching trigrams, scopes, types, etc.
/// While consuming more memory and having longer build stage due to
/// preprocessing, Dex will have substantially lower latency. It will also allow
/// efficient symbol searching which is crucial for operations like code
/// completion, and can be very important for a number of different code
/// transformations which will be eventually supported by Clangd.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_DEX_DEX_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_DEX_DEX_H

#include "index/dex/Iterator.h"
#include "index/Index.h"
#include "index/Relation.h"
#include "index/dex/PostingList.h"
#include "index/dex/Token.h"
#include "llvm/ADT/StringSet.h"

namespace clang {
namespace clangd {
namespace dex {

/// In-memory Dex trigram-based index implementation.
class Dex : public SymbolIndex {
public:
  // All data must outlive this index.
  template <typename SymbolRange, typename RefsRange, typename RelationsRange>
  Dex(SymbolRange &&Symbols, RefsRange &&Refs, RelationsRange &&Relations)
      : Corpus(0) {
    for (auto &&Sym : Symbols)
      this->Symbols.push_back(&Sym);
    for (auto &&Ref : Refs)
      this->Refs.try_emplace(Ref.first, Ref.second);
    for (auto &&Rel : Relations)
      this->Relations[std::make_pair(Rel.Subject,
                                     static_cast<uint8_t>(Rel.Predicate))]
          .push_back(Rel.Object);
    buildIndex();
  }
  // Symbols and Refs are owned by BackingData, Index takes ownership.
  template <typename SymbolRange, typename RefsRange, typename RelationsRange,
            typename Payload>
  Dex(SymbolRange &&Symbols, RefsRange &&Refs, RelationsRange &&Relations,
      Payload &&BackingData, size_t BackingDataSize)
      : Dex(std::forward<SymbolRange>(Symbols), std::forward<RefsRange>(Refs),
            std::forward<RelationsRange>(Relations)) {
    KeepAlive = std::shared_ptr<void>(
        std::make_shared<Payload>(std::move(BackingData)), nullptr);
    this->BackingDataSize = BackingDataSize;
  }

  template <typename SymbolRange, typename RefsRange, typename RelationsRange,
            typename FileRange, typename Payload>
  Dex(SymbolRange &&Symbols, RefsRange &&Refs, RelationsRange &&Relations,
      FileRange &&Files, IndexContents IdxContents, Payload &&BackingData,
      size_t BackingDataSize)
      : Dex(std::forward<SymbolRange>(Symbols), std::forward<RefsRange>(Refs),
            std::forward<RelationsRange>(Relations),
            std::forward<Payload>(BackingData), BackingDataSize) {
    this->Files = std::forward<FileRange>(Files);
    this->IdxContents = IdxContents;
  }

  /// Builds an index from slabs. The index takes ownership of the slab.
  static std::unique_ptr<SymbolIndex> build(SymbolSlab, RefSlab, RelationSlab);

  bool
  fuzzyFind(const FuzzyFindRequest &Req,
            llvm::function_ref<void(const Symbol &)> Callback) const override;

  void lookup(const LookupRequest &Req,
              llvm::function_ref<void(const Symbol &)> Callback) const override;

  bool refs(const RefsRequest &Req,
            llvm::function_ref<void(const Ref &)> Callback) const override;

  void relations(const RelationsRequest &Req,
                 llvm::function_ref<void(const SymbolID &, const Symbol &)>
                     Callback) const override;

  llvm::unique_function<IndexContents(llvm::StringRef) const>
  indexedFiles() const override;

  size_t estimateMemoryUsage() const override;

private:
  void buildIndex();
  std::unique_ptr<Iterator> iterator(const Token &Tok) const;
  std::unique_ptr<Iterator>
  createFileProximityIterator(llvm::ArrayRef<std::string> ProximityPaths) const;
  std::unique_ptr<Iterator>
  createTypeBoostingIterator(llvm::ArrayRef<std::string> Types) const;

  /// Stores symbols sorted in the descending order of symbol quality..
  std::vector<const Symbol *> Symbols;
  /// SymbolQuality[I] is the quality of Symbols[I].
  std::vector<float> SymbolQuality;
  llvm::DenseMap<SymbolID, const Symbol *> LookupTable;
  /// Inverted index is a mapping from the search token to the posting list,
  /// which contains all items which can be characterized by such search token.
  /// For example, if the search token is scope "std::", the corresponding
  /// posting list would contain all indices of symbols defined in namespace
  /// std. Inverted index is used to retrieve posting lists which are processed
  /// during the fuzzyFind process.
  llvm::DenseMap<Token, PostingList> InvertedIndex;
  dex::Corpus Corpus;
  llvm::DenseMap<SymbolID, llvm::ArrayRef<Ref>> Refs;
  static_assert(sizeof(RelationKind) == sizeof(uint8_t),
                "RelationKind should be of same size as a uint8_t");
  llvm::DenseMap<std::pair<SymbolID, uint8_t>, std::vector<SymbolID>> Relations;
  std::shared_ptr<void> KeepAlive; // poor man's move-only std::any
  // Set of files which were used during this index build.
  llvm::StringSet<> Files;
  // Contents of the index (symbols, references, etc.)
  IndexContents IdxContents;
  // Size of memory retained by KeepAlive.
  size_t BackingDataSize = 0;
};

/// Returns Search Token for a number of parent directories of given Path.
/// Should be used within the index build process.
///
/// This function is exposed for testing only.
std::vector<std::string> generateProximityURIs(llvm::StringRef URIPath);

} // namespace dex
} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_DEX_DEX_H

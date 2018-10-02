//===--- Dex.h - Dex Symbol Index Implementation ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

#include "Iterator.h"
#include "PostingList.h"
#include "Token.h"
#include "Trigram.h"
#include "index/Index.h"
#include "index/MemIndex.h"
#include "index/SymbolCollector.h"

namespace clang {
namespace clangd {
namespace dex {

/// In-memory Dex trigram-based index implementation.
// FIXME(kbobyrev): Introduce serialization and deserialization of the symbol
// index so that it can be loaded from the disk. Since static index is not
// changed frequently, it's safe to assume that it has to be built only once
// (when the clangd process starts). Therefore, it can be easier to store built
// index on disk and then load it if available.
class Dex : public SymbolIndex {
public:
  // All symbols must outlive this index.
  template <typename Range>
  Dex(Range &&Symbols, llvm::ArrayRef<std::string> Schemes)
      : Corpus(0), URISchemes(Schemes) {
    // If Schemes don't contain any items, fall back to SymbolCollector's
    // default URI schemes.
    if (URISchemes.empty()) {
      SymbolCollector::Options Opts;
      URISchemes = Opts.URISchemes;
    }
    for (auto &&Sym : Symbols)
      this->Symbols.push_back(&Sym);
    buildIndex();
  }
  // Symbols are owned by BackingData, Index takes ownership.
  template <typename Range, typename Payload>
  Dex(Range &&Symbols, Payload &&BackingData, size_t BackingDataSize,
      llvm::ArrayRef<std::string> URISchemes)
      : Dex(std::forward<Range>(Symbols), URISchemes) {
    KeepAlive = std::shared_ptr<void>(
        std::make_shared<Payload>(std::move(BackingData)), nullptr);
    this->BackingDataSize = BackingDataSize;
  }

  /// Builds an index from a slab. The index takes ownership of the slab.
  static std::unique_ptr<SymbolIndex>
  build(SymbolSlab Slab, llvm::ArrayRef<std::string> URISchemes) {
    // Store Slab size before it is moved.
    const auto BackingDataSize = Slab.bytes();
    return llvm::make_unique<Dex>(Slab, std::move(Slab), BackingDataSize,
                                  URISchemes);
  }

  bool
  fuzzyFind(const FuzzyFindRequest &Req,
            llvm::function_ref<void(const Symbol &)> Callback) const override;

  void lookup(const LookupRequest &Req,
              llvm::function_ref<void(const Symbol &)> Callback) const override;

  void refs(const RefsRequest &Req,
            llvm::function_ref<void(const Ref &)> Callback) const override;

  size_t estimateMemoryUsage() const override;

private:
  void buildIndex();

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
  std::shared_ptr<void> KeepAlive; // poor man's move-only std::any
  // Size of memory retained by KeepAlive.
  size_t BackingDataSize = 0;

  std::vector<std::string> URISchemes;
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

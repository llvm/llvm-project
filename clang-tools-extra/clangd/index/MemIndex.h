//===--- MemIndex.h - Dynamic in-memory symbol index. -------------- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_MEMINDEX_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_MEMINDEX_H

#include "Index.h"
#include <mutex>

namespace clang {
namespace clangd {

/// MemIndex is a naive in-memory index suitable for a small set of symbols.
class MemIndex : public SymbolIndex {
public:
  /// Maps from a symbol ID to all corresponding symbol occurrences.
  /// The map doesn't own occurrence objects.
  using OccurrenceMap =
      llvm::DenseMap<SymbolID, std::vector<const SymbolOccurrence *>>;

  MemIndex() = default;
  // All symbols and occurrences must outlive this index.
  // TODO: find a better type for Occurrences here.
  template <typename SymbolRange>
  MemIndex(SymbolRange &&Symbols, OccurrenceMap Occurrences)
      : Occurrences(std::move(Occurrences)) {
    for (const Symbol &S : Symbols)
      Index[S.ID] = &S;
  }
  // Symbols are owned by BackingData, Index takes ownership.
  template <typename Range, typename Payload>
  MemIndex(Range &&Symbols, OccurrenceMap Occurrences, Payload &&BackingData)
      : MemIndex(std::forward<Range>(Symbols), std::move(Occurrences)) {
    KeepAlive = std::shared_ptr<void>(
        std::make_shared<Payload>(std::move(BackingData)), nullptr);
  }

  /// Builds an index from a slab. The index takes ownership of the data.
  static std::unique_ptr<SymbolIndex> build(SymbolSlab Slab,
                                            SymbolOccurrenceSlab Occurrences);

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
  // Index is a set of symbols that are deduplicated by symbol IDs.
  llvm::DenseMap<SymbolID, const Symbol *> Index;
  // A map from symbol ID to symbol occurrences, support query by IDs.
  OccurrenceMap Occurrences;
  std::shared_ptr<void> KeepAlive; // poor man's move-only std::any
};

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_MEMINDEX_H

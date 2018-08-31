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

/// \brief This implements an index for a (relatively small) set of symbols (or
/// symbol occurrences) that can be easily managed in memory.
class MemIndex : public SymbolIndex {
public:
  /// Maps from a symbol ID to all corresponding symbol occurrences.
  /// The map doesn't own occurrence objects.
  using OccurrenceMap =
      llvm::DenseMap<SymbolID, std::vector<const SymbolOccurrence *>>;

  /// \brief (Re-)Build index for `Symbols` and update `Occurrences`.
  /// All symbol pointers and symbol occurrence pointers must remain accessible
  /// as long as `Symbols` and `Occurrences` are kept alive.
  void build(std::shared_ptr<std::vector<const Symbol *>> Symbols,
             std::shared_ptr<OccurrenceMap> Occurrences);

  /// \brief Build index from a symbol slab and a symbol occurrence slab.
  static std::unique_ptr<SymbolIndex> build(SymbolSlab Symbols,
                                            SymbolOccurrenceSlab Occurrences);

  bool
  fuzzyFind(const FuzzyFindRequest &Req,
            llvm::function_ref<void(const Symbol &)> Callback) const override;

  void
  lookup(const LookupRequest &Req,
         llvm::function_ref<void(const Symbol &)> Callback) const override;

  void findOccurrences(const OccurrencesRequest &Req,
                       llvm::function_ref<void(const SymbolOccurrence &)>
                           Callback) const override;

  size_t estimateMemoryUsage() const override;

private:

  std::shared_ptr<std::vector<const Symbol *>> Symbols;
  // Index is a set of symbols that are deduplicated by symbol IDs.
  // FIXME: build smarter index structure.
  llvm::DenseMap<SymbolID, const Symbol *> Index;
  // A map from symbol ID to symbol occurrences, support query by IDs.
  std::shared_ptr<OccurrenceMap> Occurrences;
  mutable std::mutex Mutex;
};

// Returns pointers to the symbols in given slab and bundles slab lifetime with
// returned symbol pointers so that the pointers are never invalid.
std::shared_ptr<std::vector<const Symbol *>>
getSymbolsFromSlab(SymbolSlab Slab);

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_MEMINDEX_H

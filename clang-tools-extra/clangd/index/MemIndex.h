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

/// \brief This implements an index for a (relatively small) set of symbols that
/// can be easily managed in memory.
class MemIndex : public SymbolIndex {
public:
  /// \brief (Re-)Build index for `Symbols`. All symbol pointers must remain
  /// accessible as long as `Symbols` is kept alive.
  void build(std::shared_ptr<std::vector<const Symbol *>> Symbols);

  bool fuzzyFind(Context &Ctx, const FuzzyFindRequest &Req,
                 std::function<void(const Symbol &)> Callback) const override;

private:
  std::shared_ptr<std::vector<const Symbol *>> Symbols;
  // Index is a set of symbols that are deduplicated by symbol IDs.
  // FIXME: build smarter index structure.
  llvm::DenseMap<SymbolID, const Symbol *> Index;
  mutable std::mutex Mutex;
};

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_MEMINDEX_H

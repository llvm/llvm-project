//===--- FileSymbols.h - Symbols from files. ---------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_FILESYMBOLS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_FILESYMBOLS_H

#include "../Path.h"
#include "Index.h"
#include "llvm/ADT/StringMap.h"
#include <mutex>

namespace clang {
namespace clangd {

/// \brief A container of Symbols from several source files. It can be updated
/// at source-file granularity, replacing all symbols from one file with a new
/// set.
///
/// This implements a snapshot semantics for symbols in a file. Each update to a
/// file will create a new snapshot for all symbols in the file. Snapshots are
/// managed with shared pointers that are shared between this class and the
/// users. For each file, this class only stores a pointer pointing to the
/// newest snapshot, and an outdated snapshot is deleted by the last owner of
/// the snapshot, either this class or the symbol index.
///
/// The snapshot semantics keeps critical sections minimal since we only need
/// locking when we swap or obtain refereces to snapshots.
class FileSymbols {
public:
  /// \brief Updates all symbols in a file. If \p Slab is nullptr, symbols for
  /// \p Path will be removed.
  void update(PathRef Path, std::unique_ptr<SymbolSlab> Slab);

  // The shared_ptr keeps the symbols alive
  std::shared_ptr<std::vector<const Symbol *>> allSymbols();

private:
  mutable std::mutex Mutex;

  /// \brief Stores the latest snapshots for all active files.
  llvm::StringMap<std::shared_ptr<SymbolSlab>> FileToSlabs;
};

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_FILESYMBOLS_H

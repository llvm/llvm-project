//===--- FileIndex.h - Index for files. ---------------------------- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// FileIndex implements SymbolIndex for symbols from a set of files. Symbols are
// maintained at source-file granuality (e.g. with ASTs), and files can be
// updated dynamically.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_FILEINDEX_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_FILEINDEX_H

#include "ClangdUnit.h"
#include "Index.h"
#include "MemIndex.h"
#include "Merge.h"
#include "clang/Lex/Preprocessor.h"
#include <memory>

namespace clang {
namespace clangd {

/// Select between in-memory index implementations, which have tradeoffs.
enum class IndexType {
  // MemIndex is trivially cheap to build, but has poor query performance.
  Light,
  // Dex is relatively expensive to build and uses more memory, but is fast.
  Heavy,
};

/// A container of Symbols from several source files. It can be updated
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
/// locking when we swap or obtain references to snapshots.
class FileSymbols {
public:
  /// Updates all symbols and refs in a file.
  /// If either is nullptr, corresponding data for \p Path will be removed.
  void update(PathRef Path, std::unique_ptr<SymbolSlab> Slab,
              std::unique_ptr<RefSlab> Refs);

  // The index keeps the symbols alive.
  std::unique_ptr<SymbolIndex>
  buildIndex(IndexType, ArrayRef<std::string> URISchemes = {});

private:
  mutable std::mutex Mutex;

  /// Stores the latest symbol snapshots for all active files.
  llvm::StringMap<std::shared_ptr<SymbolSlab>> FileToSymbols;
  /// Stores the latest ref snapshots for all active files.
  llvm::StringMap<std::shared_ptr<RefSlab>> FileToRefs;
};

/// This manages symbols from files and an in-memory index on all symbols.
/// FIXME: Expose an interface to remove files that are closed.
class FileIndex : public MergedIndex {
public:
  /// If URISchemes is empty, the default schemes in SymbolCollector will be
  /// used.
  FileIndex(std::vector<std::string> URISchemes = {}, bool UseDex = true);

  /// Update preamble symbols of file \p Path with all declarations in \p AST
  /// and macros in \p PP.
  void updatePreamble(PathRef Path, ASTContext &AST,
                      std::shared_ptr<Preprocessor> PP);

  /// Update symbols and references from main file \p Path with
  /// `indexMainDecls`.
  void updateMain(PathRef Path, ParsedAST &AST);

private:
  bool UseDex; // FIXME: this should be always on.
  std::vector<std::string> URISchemes;

  // Contains information from each file's preamble only.
  // These are large, but update fairly infrequently (preambles are stable).
  // Missing information:
  //  - symbol refs (these are always "from the main file")
  //  - definition locations in the main file
  //
  // FIXME: Because the preambles for different TUs have large overlap and
  // FileIndex doesn't deduplicate, this uses lots of extra RAM.
  // The biggest obstacle in fixing this: the obvious approach of partitioning
  // by declaring file (rather than main file) fails if headers provide
  // different symbols based on preprocessor state.
  FileSymbols PreambleSymbols;
  SwapIndex PreambleIndex;

  // Contains information from each file's main AST.
  // These are updated frequently (on file change), but are relatively small.
  // Mostly contains:
  //  - refs to symbols declared in the preamble and referenced from main
  //  - symbols declared both in the main file and the preamble
  // (Note that symbols *only* in the main file are not indexed).
  FileSymbols MainFileSymbols;
  SwapIndex MainFileIndex;
};

/// Retrieves symbols and refs of local top level decls in \p AST (i.e.
/// `AST.getLocalTopLevelDecls()`).
/// Exposed to assist in unit tests.
/// If URISchemes is empty, the default schemes in SymbolCollector will be used.
std::pair<SymbolSlab, RefSlab>
indexMainDecls(ParsedAST &AST, llvm::ArrayRef<std::string> URISchemes = {});

/// Idex declarations from \p AST and macros from \p PP that are declared in
/// included headers.
/// If URISchemes is empty, the default schemes in SymbolCollector will be used.
SymbolSlab indexHeaderSymbols(ASTContext &AST, std::shared_ptr<Preprocessor> PP,
                              llvm::ArrayRef<std::string> URISchemes = {});

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_FILEINDEX_H

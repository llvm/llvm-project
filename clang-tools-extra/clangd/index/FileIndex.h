//===--- FileIndex.h - Index for files. ---------------------------- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// FileIndex implements SymbolIndex for symbols from a set of files. Symbols are
// maintained at source-file granularity (e.g. with ASTs), and files can be
// updated dynamically.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_FILEINDEX_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_FILEINDEX_H

#include "Headers.h"
#include "Index.h"
#include "MemIndex.h"
#include "Merge.h"
#include "Path.h"
#include "index/CanonicalIncludes.h"
#include "index/Ref.h"
#include "index/Relation.h"
#include "index/Serialization.h"
#include "index/Symbol.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <vector>

namespace clang {
class ASTContext;
namespace clangd {
class ParsedAST;

/// Select between in-memory index implementations, which have tradeoffs.
enum class IndexType {
  // MemIndex is trivially cheap to build, but has poor query performance.
  Light,
  // Dex is relatively expensive to build and uses more memory, but is fast.
  Heavy,
};

/// How to handle duplicated symbols across multiple files.
enum class DuplicateHandling {
  // Pick a random symbol. Less accurate but faster.
  PickOne,
  // Merge symbols. More accurate but slower.
  Merge,
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
  /// If CountReferences is true, \p Refs will be used for counting References
  /// during merging.
  void update(PathRef Path, std::unique_ptr<SymbolSlab> Slab,
              std::unique_ptr<RefSlab> Refs,
              std::unique_ptr<RelationSlab> Relations, bool CountReferences);

  /// The index keeps the symbols alive.
  /// Will count Symbol::References based on number of references in the main
  /// files, while building the index with DuplicateHandling::Merge option.
  std::unique_ptr<SymbolIndex>
  buildIndex(IndexType,
             DuplicateHandling DuplicateHandle = DuplicateHandling::PickOne);

private:
  struct RefSlabAndCountReferences {
    std::shared_ptr<RefSlab> Slab;
    bool CountReferences = false;
  };
  mutable std::mutex Mutex;

  /// Stores the latest symbol snapshots for all active files.
  llvm::StringMap<std::shared_ptr<SymbolSlab>> FileToSymbols;
  /// Stores the latest ref snapshots for all active files.
  llvm::StringMap<RefSlabAndCountReferences> FileToRefs;
  /// Stores the latest relation snapshots for all active files.
  llvm::StringMap<std::shared_ptr<RelationSlab>> FileToRelations;
};

/// This manages symbols from files and an in-memory index on all symbols.
/// FIXME: Expose an interface to remove files that are closed.
class FileIndex : public MergedIndex {
public:
  FileIndex(bool UseDex = true);

  /// Update preamble symbols of file \p Path with all declarations in \p AST
  /// and macros in \p PP.
  void updatePreamble(PathRef Path, llvm::StringRef Version, ASTContext &AST,
                      std::shared_ptr<Preprocessor> PP,
                      const CanonicalIncludes &Includes);

  /// Update symbols and references from main file \p Path with
  /// `indexMainDecls`.
  void updateMain(PathRef Path, ParsedAST &AST);

private:
  bool UseDex; // FIXME: this should be always on.

  // Contains information from each file's preamble only. Symbols and relations
  // are sharded per declaration file to deduplicate multiple symbols and reduce
  // memory usage.
  // Missing information:
  //  - symbol refs (these are always "from the main file")
  //  - definition locations in the main file
  //
  // Note that we store only one version of a header, hence symbols appearing in
  // different PP states will be missing.
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

using SlabTuple = std::tuple<SymbolSlab, RefSlab, RelationSlab>;

/// Retrieves symbols and refs of local top level decls in \p AST (i.e.
/// `AST.getLocalTopLevelDecls()`).
/// Exposed to assist in unit tests.
SlabTuple indexMainDecls(ParsedAST &AST);

/// Index declarations from \p AST and macros from \p PP that are declared in
/// included headers.
SlabTuple indexHeaderSymbols(llvm::StringRef Version, ASTContext &AST,
                             std::shared_ptr<Preprocessor> PP,
                             const CanonicalIncludes &Includes);

/// Takes slabs coming from a TU (multiple files) and shards them per
/// declaration location.
struct FileShardedIndex {
  /// \p HintPath is used to convert file URIs stored in symbols into absolute
  /// paths.
  explicit FileShardedIndex(IndexFileIn Input, PathRef HintPath);

  /// Returns absolute paths for all files that has a shard.
  std::vector<PathRef> getAllFiles() const;

  /// Generates index shard for the \p File. Note that this function results in
  /// a copy of all the relevant data.
  /// Returned index will always have Symbol/Refs/Relation Slabs set, even if
  /// they are empty.
  IndexFileIn getShard(PathRef File) const;

private:
  // Contains all the information that belongs to a single file.
  struct FileShard {
    // Either declared or defined in the file.
    llvm::DenseSet<const Symbol *> Symbols;
    // Reference occurs in the file.
    llvm::DenseSet<const Ref *> Refs;
    // Subject is declared in the file.
    llvm::DenseSet<const Relation *> Relations;
    // Contains edges for only the direct includes.
    IncludeGraph IG;
  };

  // Keeps all the information alive.
  const IndexFileIn Index;
  // Mapping from absolute paths to slab information.
  llvm::StringMap<FileShard> Shards;
  // Used to build RefSlabs.
  llvm::DenseMap<const Ref *, SymbolID> RefToSymID;
};

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_FILEINDEX_H

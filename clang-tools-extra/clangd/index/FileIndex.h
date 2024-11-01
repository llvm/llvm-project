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
#include "clang-include-cleaner/Record.h"
#include "index/Index.h"
#include "index/Merge.h"
#include "index/Ref.h"
#include "index/Relation.h"
#include "index/Serialization.h"
#include "index/Symbol.h"
#include "support/MemoryTree.h"
#include "support/Path.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <optional>
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

/// A container of slabs associated with a key. It can be updated at key
/// granularity, replacing all slabs belonging to a key with a new set. Keys are
/// usually file paths/uris.
///
/// This implements snapshot semantics. Each update will create a new snapshot
/// for all slabs of the Key. Snapshots are managed with shared pointers that
/// are shared between this class and the users. For each key, this class only
/// stores a pointer pointing to the newest snapshot, and an outdated snapshot
/// is deleted by the last owner of the snapshot, either this class or the
/// symbol index.
///
/// The snapshot semantics keeps critical sections minimal since we only need
/// locking when we swap or obtain references to snapshots.
class FileSymbols {
public:
  FileSymbols(IndexContents IdxContents);
  /// Updates all slabs associated with the \p Key.
  /// If either is nullptr, corresponding data for \p Key will be removed.
  /// If CountReferences is true, \p Refs will be used for counting references
  /// during merging.
  void update(llvm::StringRef Key, std::unique_ptr<SymbolSlab> Symbols,
              std::unique_ptr<RefSlab> Refs,
              std::unique_ptr<RelationSlab> Relations, bool CountReferences);

  /// The index keeps the slabs alive.
  /// Will count Symbol::References based on number of references in the main
  /// files, while building the index with DuplicateHandling::Merge option.
  /// Version is populated with an increasing sequence counter.
  std::unique_ptr<SymbolIndex>
  buildIndex(IndexType,
             DuplicateHandling DuplicateHandle = DuplicateHandling::PickOne,
             size_t *Version = nullptr);

  void profile(MemoryTree &MT) const;

private:
  IndexContents IdxContents;

  struct RefSlabAndCountReferences {
    std::shared_ptr<RefSlab> Slab;
    bool CountReferences = false;
  };
  mutable std::mutex Mutex;

  size_t Version = 0;
  llvm::StringMap<std::shared_ptr<SymbolSlab>> SymbolsSnapshot;
  llvm::StringMap<RefSlabAndCountReferences> RefsSnapshot;
  llvm::StringMap<std::shared_ptr<RelationSlab>> RelationsSnapshot;
};

/// This manages symbols from files and an in-memory index on all symbols.
/// FIXME: Expose an interface to remove files that are closed.
class FileIndex : public MergedIndex {
public:
  FileIndex();

  /// Update preamble symbols of file \p Path with all declarations in \p AST
  /// and macros in \p PP.
  void updatePreamble(PathRef Path, llvm::StringRef Version, ASTContext &AST,
                      Preprocessor &PP,
                      const include_cleaner::PragmaIncludes &PI);
  void updatePreamble(IndexFileIn);

  /// Update symbols and references from main file \p Path with
  /// `indexMainDecls`.
  void updateMain(PathRef Path, ParsedAST &AST);

  void profile(MemoryTree &MT) const;

private:
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

  // While both the FileIndex and SwapIndex are threadsafe, we need to track
  // versions to ensure that we don't overwrite newer indexes with older ones.
  std::mutex UpdateIndexMu;
  unsigned MainIndexVersion = 0;
  unsigned PreambleIndexVersion = 0;
};

using SlabTuple = std::tuple<SymbolSlab, RefSlab, RelationSlab>;

/// Retrieves symbols and refs of local top level decls in \p AST (i.e.
/// `AST.getLocalTopLevelDecls()`).
/// Exposed to assist in unit tests.
SlabTuple indexMainDecls(ParsedAST &AST);

/// Index declarations from \p AST and macros from \p PP that are declared in
/// included headers.
SlabTuple indexHeaderSymbols(llvm::StringRef Version, ASTContext &AST,
                             Preprocessor &PP,
                             const include_cleaner::PragmaIncludes &PI);

/// Takes slabs coming from a TU (multiple files) and shards them per
/// declaration location.
struct FileShardedIndex {
  /// \p HintPath is used to convert file URIs stored in symbols into absolute
  /// paths.
  explicit FileShardedIndex(IndexFileIn Input);

  /// Returns uris for all files that has a shard.
  std::vector<llvm::StringRef> getAllSources() const;

  /// Generates index shard for the \p Uri. Note that this function results in
  /// a copy of all the relevant data.
  /// Returned index will always have Symbol/Refs/Relation Slabs set, even if
  /// they are empty.
  std::optional<IndexFileIn> getShard(llvm::StringRef Uri) const;

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
  // Mapping from URIs to slab information.
  llvm::StringMap<FileShard> Shards;
  // Used to build RefSlabs.
  llvm::DenseMap<const Ref *, SymbolID> RefToSymID;
};

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_FILEINDEX_H

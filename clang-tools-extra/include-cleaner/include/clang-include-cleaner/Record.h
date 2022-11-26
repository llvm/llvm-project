//===--- Record.h - Record compiler events ------------------------- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Where Analysis.h analyzes AST nodes and recorded preprocessor events, this
// file defines ways to capture AST and preprocessor information from a parse.
//
// These are the simplest way to connect include-cleaner logic to the parser,
// but other ways are possible (for example clangd records includes separately).
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_INCLUDE_CLEANER_RECORD_H
#define CLANG_INCLUDE_CLEANER_RECORD_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/FileSystem/UniqueID.h"
#include "clang-include-cleaner/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include <memory>
#include <vector>

namespace clang {
class ASTConsumer;
class ASTContext;
class CompilerInstance;
class Decl;
class FileEntry;
class Preprocessor;
class PPCallbacks;
class FileManager;

namespace include_cleaner {

/// Captures #include mapping information. It analyses IWYU Pragma comments and
/// other use-instead-like mechanisms (#pragma include_instead) on included
/// files.
///
/// This is a low-level piece being used in the "Location => Header" analysis
/// step to determine the final public header rather than the header directly
/// defines the symbol.
class PragmaIncludes {
public:
  /// Installs an analysing PPCallback and CommentHandler and populates results
  /// to the structure.
  void record(const CompilerInstance &CI);

  /// Returns true if the given #include of the main-file should never be
  /// removed.
  bool shouldKeep(unsigned HashLineNumber) const {
    return ShouldKeep.find(HashLineNumber) != ShouldKeep.end();
  }

  /// Returns the public mapping include for the given physical header file.
  /// Returns "" if there is none.
  llvm::StringRef getPublic(const FileEntry *File) const;

  /// Returns all direct exporter headers for the given header file.
  /// Returns empty if there is none.
  llvm::SmallVector<const FileEntry *> getExporters(const FileEntry *File,
                                                    FileManager &FM) const;

  /// Returns true if the given file is a self-contained file.
  bool isSelfContained(const FileEntry *File) const;

private:
  class RecordPragma;
  /// 1-based Line numbers for the #include directives of the main file that
  /// should always keep (e.g. has the `IWYU pragma: keep` or `IWYU pragma:
  /// export` right after).
  llvm::DenseSet</*LineNumber*/ unsigned> ShouldKeep;

  /// The public header mapping by the IWYU private pragma.
  //
  // !!NOTE: instead of using a FileEntry* to identify the physical file, we
  // deliberately use the UniqueID to ensure the result is stable across
  // FileManagers (for clangd's preamble and main-file builds).
  llvm::DenseMap<llvm::sys::fs::UniqueID, llvm::StringRef /*VerbatimSpelling*/>
      IWYUPublic;

  /// A reverse map from the underlying header to its exporter headers.
  //
  //  There's no way to get a FileEntry from a UniqueID, especially when it
  //  hasn't been opened before. So store the full path and convert it to a
  //  FileEntry by opening the file again through a FileManager.
  llvm::DenseMap<llvm::sys::fs::UniqueID,
                 llvm::SmallVector</*RealPathNames*/ llvm::StringRef>>
      IWYUExportBy;

  /// Contains all non self-contained files detected during the parsing.
  llvm::DenseSet<llvm::sys::fs::UniqueID> NonSelfContainedFiles;

  /// Owns the strings.
  llvm::BumpPtrAllocator Arena;

  // FIXME: add support for clang use_instead pragma
};

/// Recorded main-file parser events relevant to include-cleaner.
struct RecordedAST {
  /// The consumer (when installed into clang) tracks declarations in `*this`.
  std::unique_ptr<ASTConsumer> record();

  ASTContext *Ctx = nullptr;
  /// The set of declarations written at file scope inside the main file.
  ///
  /// These are the roots of the subtrees that should be traversed to find uses.
  /// (Traversing the TranslationUnitDecl would find uses inside headers!)
  std::vector<Decl *> Roots;
};

/// Recorded main-file preprocessor events relevant to include-cleaner.
///
/// This doesn't include facts that we record globally for the whole TU, even
/// when they occur in the main file (e.g. IWYU pragmas).
struct RecordedPP {
  /// The callback (when installed into clang) tracks macros/includes in this.
  std::unique_ptr<PPCallbacks> record(const Preprocessor &PP);

  /// Describes where macros were used in the main file.
  std::vector<SymbolReference> MacroReferences;

  /// A container for all includes present in the main file.
  /// Supports efficiently hit-testing Headers against Includes.
  /// FIXME: is there a more natural header for this class?
  class RecordedIncludes {
  public:
    void add(const Include &);

    /// All #includes seen, in the order they appear.
    llvm::ArrayRef<Include> all() const { return All; }

    /// Determine #includes that match a header (that provides a used symbol).
    ///
    /// Matching is based on the type of Header specified:
    ///  - for a physical file like /path/to/foo.h, we check Resolved
    ///  - for a logical file like <vector>, we check Spelled
    llvm::SmallVector<const Include *> match(Header H) const;

    /// Finds the include written on the specified line.
    const Include *atLine(unsigned OneBasedIndex) const;

  private:
    std::vector<Include> All;
    // Lookup structures for match(), values are index into All.
    llvm::StringMap<llvm::SmallVector<unsigned>> BySpelling;
    llvm::DenseMap<const FileEntry *, llvm::SmallVector<unsigned>> ByFile;
    llvm::DenseMap<unsigned, unsigned> ByLine;
  } Includes;
};

} // namespace include_cleaner
} // namespace clang

#endif

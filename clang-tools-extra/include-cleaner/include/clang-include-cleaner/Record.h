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
#include "llvm/Support/FileSystem/UniqueID.h"
#include <memory>
#include <vector>

namespace clang {
class ASTConsumer;
class ASTContext;
class CompilerInstance;
class Decl;
class FileEntry;

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
  llvm::DenseMap<llvm::sys::fs::UniqueID, std::string /*VerbatimSpelling*/>
      IWYUPublic;

  // FIXME: add other IWYU supports (export etc)
  // FIXME: add support for clang use_instead pragma
  // FIXME: add selfcontained file.
};

// Contains recorded parser events relevant to include-cleaner.
struct RecordedAST {
  // The consumer (when installed into clang) tracks declarations in this.
  std::unique_ptr<ASTConsumer> record();

  ASTContext *Ctx = nullptr;
  // The set of declarations written at file scope inside the main file.
  //
  // These are the roots of the subtrees that should be traversed to find uses.
  // (Traversing the TranslationUnitDecl would find uses inside headers!)
  std::vector<Decl *> Roots;
};

} // namespace include_cleaner
} // namespace clang

#endif

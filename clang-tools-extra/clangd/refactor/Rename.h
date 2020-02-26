//===--- Rename.h - Symbol-rename refactorings -------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_REFACTOR_RENAME_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_REFACTOR_RENAME_H

#include "Path.h"
#include "Protocol.h"
#include "SourceCode.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/Support/Error.h"

namespace clang {
namespace clangd {
class ParsedAST;
class SymbolIndex;

/// Gets dirty buffer for a given file \p AbsPath.
/// Returns None if there is no dirty buffer for the given file.
using DirtyBufferGetter =
    llvm::function_ref<llvm::Optional<std::string>(PathRef AbsPath)>;

struct RenameOptions {
  /// If true, enable cross-file rename; otherwise, only allows to rename a
  /// symbol that's only used in the current file.
  bool AllowCrossFile = false;
  /// The mamimum number of affected files (0 means no limit), only meaningful
  /// when AllowCrossFile = true.
  /// If the actual number exceeds the limit, rename is forbidden.
  size_t LimitFiles = 50;
  /// If true, format the rename edits, only meaningful in ClangdServer layer.
  bool WantFormat = false;
};

struct RenameInputs {
  Position Pos; // the position triggering the rename
  llvm::StringRef NewName;

  ParsedAST &AST;
  llvm::StringRef MainFilePath;

  const SymbolIndex *Index = nullptr;

  RenameOptions Opts = {};
  // When set, used by the rename to get file content for all rename-related
  // files.
  // If there is no corresponding dirty buffer, we will use the file content
  // from disk.
  DirtyBufferGetter GetDirtyBuffer = nullptr;
};

/// Renames all occurrences of the symbol. The result edits are unformatted.
/// If AllowCrossFile is false, returns an error if rename a symbol that's used
/// in another file (per the index).
llvm::Expected<FileEdits> rename(const RenameInputs &RInputs);

/// Generates rename edits that replaces all given occurrences with the
/// NewName.
/// Exposed for testing only.
/// REQUIRED: Occurrences is sorted and doesn't have duplicated ranges.
llvm::Expected<Edit> buildRenameEdit(llvm::StringRef AbsFilePath,
                                     llvm::StringRef InitialCode,
                                     std::vector<Range> Occurrences,
                                     llvm::StringRef NewName);

/// Adjusts indexed occurrences to match the current state of the file.
///
/// The Index is not always up to date. Blindly editing at the locations
/// reported by the index may mangle the code in such cases.
/// This function determines whether the indexed occurrences can be applied to
/// this file, and heuristically repairs the occurrences if necessary.
///
/// The API assumes that Indexed contains only named occurrences (each
/// occurrence has the same length).
/// REQUIRED: Indexed is sorted.
llvm::Optional<std::vector<Range>>
adjustRenameRanges(llvm::StringRef DraftCode, llvm::StringRef Identifier,
                   std::vector<Range> Indexed, const LangOptions &LangOpts);

/// Calculates the lexed occurrences that the given indexed occurrences map to.
/// Returns None if we don't find a mapping.
///
/// Exposed for testing only.
///
/// REQUIRED: Indexed and Lexed are sorted.
llvm::Optional<std::vector<Range>> getMappedRanges(ArrayRef<Range> Indexed,
                                                   ArrayRef<Range> Lexed);
/// Evaluates how good the mapped result is. 0 indicates a perfect match.
///
/// Exposed for testing only.
///
/// REQUIRED: Indexed and Lexed are sorted, Indexed and MappedIndex have the
/// same size.
size_t renameRangeAdjustmentCost(ArrayRef<Range> Indexed, ArrayRef<Range> Lexed,
                                 ArrayRef<size_t> MappedIndex);

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_REFACTOR_RENAME_H

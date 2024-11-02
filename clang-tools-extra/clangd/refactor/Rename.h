//===--- Rename.h - Symbol-rename refactorings -------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_REFACTOR_RENAME_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_REFACTOR_RENAME_H

#include "Protocol.h"
#include "SourceCode.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/LangOptions.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include <optional>

namespace clang {
namespace clangd {
class ParsedAST;
class SymbolIndex;

struct RenameOptions {
  /// The maximum number of affected files (0 means no limit), only meaningful
  /// when AllowCrossFile = true.
  /// If the actual number exceeds the limit, rename is forbidden.
  size_t LimitFiles = 50;
  /// If true, format the rename edits, only meaningful in ClangdServer layer.
  bool WantFormat = false;
  /// Allow rename of virtual method hierarchies.
  /// Disable to support broken index implementations with missing relations.
  /// FIXME: fix those implementations and remove this option.
  bool RenameVirtual = true;
};

struct RenameInputs {
  Position Pos; // the position triggering the rename
  llvm::StringRef NewName;

  ParsedAST &AST;
  llvm::StringRef MainFilePath;

  // The filesystem to query when performing cross file renames.
  // If this is set, Index must also be set, likewise if this is nullptr, Index
  // must also be nullptr.
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS = nullptr;

  const SymbolIndex *Index = nullptr;

  RenameOptions Opts = {};
};

struct RenameResult {
  // The range of the symbol that the user can attempt to rename.
  Range Target;
  // Placeholder text for the rename operation if non-empty.
  std::string Placeholder;
  // Rename occurrences for the current main file.
  std::vector<Range> LocalChanges;
  // Complete edits for the rename, including LocalChanges.
  // If the full set of changes is unknown, this field is empty.
  FileEdits GlobalChanges;
};

/// Represents a symbol range where the symbol can potentially have multiple
/// tokens.
struct SymbolRange {
  /// Ranges for the tokens that make up the symbol's name.
  /// Usually a single range, but there can be multiple ranges if the tokens for
  /// the symbol are split, e.g. ObjC selectors.
  std::vector<Range> Ranges;

  SymbolRange(Range R);
  SymbolRange(std::vector<Range> Ranges);

  /// Returns the first range.
  Range range() const;

  friend bool operator==(const SymbolRange &LHS, const SymbolRange &RHS);
  friend bool operator!=(const SymbolRange &LHS, const SymbolRange &RHS);
  friend bool operator<(const SymbolRange &LHS, const SymbolRange &RHS);
};

/// Renames all occurrences of the symbol. The result edits are unformatted.
/// If AllowCrossFile is false, returns an error if rename a symbol that's used
/// in another file (per the index).
llvm::Expected<RenameResult> rename(const RenameInputs &RInputs);

/// Generates rename edits that replaces all given occurrences with the
/// NewName.
/// Exposed for testing only.
/// REQUIRED: Occurrences is sorted and doesn't have duplicated ranges.
llvm::Expected<Edit> buildRenameEdit(llvm::StringRef AbsFilePath,
                                     llvm::StringRef InitialCode,
                                     std::vector<SymbolRange> Occurrences,
                                     llvm::ArrayRef<llvm::StringRef> NewNames);

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
std::optional<std::vector<SymbolRange>>
adjustRenameRanges(llvm::StringRef DraftCode, llvm::StringRef Identifier,
                   std::vector<Range> Indexed, const LangOptions &LangOpts,
                   std::optional<Selector> Selector);

/// Calculates the lexed occurrences that the given indexed occurrences map to.
/// Returns std::nullopt if we don't find a mapping.
///
/// Exposed for testing only.
///
/// REQUIRED: Indexed and Lexed are sorted.
std::optional<std::vector<SymbolRange>>
getMappedRanges(ArrayRef<Range> Indexed, ArrayRef<SymbolRange> Lexed);
/// Evaluates how good the mapped result is. 0 indicates a perfect match.
///
/// Exposed for testing only.
///
/// REQUIRED: Indexed and Lexed are sorted, Indexed and MappedIndex have the
/// same size.
size_t renameRangeAdjustmentCost(ArrayRef<Range> Indexed,
                                 ArrayRef<SymbolRange> Lexed,
                                 ArrayRef<size_t> MappedIndex);

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_REFACTOR_RENAME_H

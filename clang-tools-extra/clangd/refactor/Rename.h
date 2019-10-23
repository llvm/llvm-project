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

struct RenameInputs {
  Position Pos; // the position triggering the rename
  llvm::StringRef NewName;

  ParsedAST &AST;
  llvm::StringRef MainFilePath;

  const SymbolIndex *Index = nullptr;

  bool AllowCrossFile = false;
  // When set, used by the rename to get file content for all rename-related
  // files.
  // If there is no corresponding dirty buffer, we will use the file content
  // from disk.
  DirtyBufferGetter GetDirtyBuffer = nullptr;
};

/// Renames all occurrences of the symbol.
/// If AllowCrossFile is false, returns an error if rename a symbol that's used
/// in another file (per the index).
llvm::Expected<FileEdits> rename(const RenameInputs &RInputs);

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_REFACTOR_RENAME_H

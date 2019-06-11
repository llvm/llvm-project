//===--- Rename.h - Symbol-rename refactorings -------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_REFACTOR_RENAME_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_REFACTOR_RENAME_H

#include "ClangdUnit.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/Support/Error.h"

namespace clang {
namespace clangd {

/// Renames all occurrences of the symbol at \p Pos to \p NewName.
/// Occurrences outside the current file are not modified.
llvm::Expected<tooling::Replacements> renameWithinFile(ParsedAST &AST,
                                                       llvm::StringRef File,
                                                       Position Pos,
                                                       llvm::StringRef NewName);

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_REFACTOR_RENAME_H

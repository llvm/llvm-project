//===---------- ASTUtils.h - clang-tidy -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ASTUTILS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ASTUTILS_H

#include "clang/AST/AST.h"

namespace clang::tidy::utils {
// Returns the (closest) Function declaration surrounding |Statement| or NULL.
const FunctionDecl *getSurroundingFunction(ASTContext &Context,
                                           const Stmt &Statement);
// Determine whether Expr is a Binary or Ternary expression.
bool isBinaryOrTernary(const Expr *E);

/// Checks whether a macro flag is present in the given argument. Only considers
/// cases of single match or match in a binary OR expression. For example,
/// <needed-flag> or <flag> | <needed-flag> | ...
bool exprHasBitFlagWithSpelling(const Expr *Flags, const SourceManager &SM,
                                const LangOptions &LangOpts,
                                StringRef FlagName);

// Check if the range is entirely contained within a macro argument.
bool rangeIsEntirelyWithinMacroArgument(SourceRange Range,
                                        const SourceManager *SM);

// Check if the range contains any locations from a macro expansion.
bool rangeContainsMacroExpansion(SourceRange Range, const SourceManager *SM);

// Can a fix-it be issued for this whole Range?
// FIXME: false-negative if the entire range is fully expanded from a macro.
bool rangeCanBeFixed(SourceRange Range, const SourceManager *SM);

// Check if statements are same
bool areStatementsIdentical(const Stmt *FirstStmt, const Stmt *SecondStmt,
                            const ASTContext &Context, bool Canonical = false);

} // namespace clang::tidy::utils

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ASTUTILS_H

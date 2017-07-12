//===---------- ASTUtils.h - clang-tidy -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ASTUTILS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ASTUTILS_H

#include "clang/AST/AST.h"

namespace clang {
namespace tidy {
namespace utils {
// Returns the (closest) Function declaration surrounding |Statement| or NULL.
const FunctionDecl *getSurroundingFunction(ASTContext &Context,
                                           const Stmt &Statement);
// Determine whether Expr is a Binary or Ternary expression.
bool IsBinaryOrTernary(const Expr *E);

/// Checks whether a macro flag is present in the given argument. Only considers
/// cases of single match or match in a binary OR expression. For example,
/// <needed-flag> or <flag> | <needed-flag> | ...
bool exprHasBitFlagWithSpelling(const Expr *Flags, const SourceManager &SM,
                                const LangOptions &LangOpts,
                                StringRef FlagName);
} // namespace utils
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ASTUTILS_H

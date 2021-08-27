//===--- StmtUtils.h - Statement helper functions -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_TOOLING_REFACTOR_STMT_UTILS_H
#define LLVM_CLANG_LIB_TOOLING_REFACTOR_STMT_UTILS_H

#include "clang/Basic/SourceLocation.h"

namespace clang {

class Decl;
class LangOptions;
class Stmt;

namespace tooling {

SourceLocation getLexicalEndLocForDecl(const Decl *D, const SourceManager &SM,
                                       const LangOptions &LangOpts);

/// \brief Returns true if there should be a semicolon after the given
/// statement.
bool isSemicolonRequiredAfter(const Stmt *S);

/// Returns true if the given statement \p S is an actual expression in the
/// source. Assignment expressions are considered to be statements unless they
/// are a part of an expression.
bool isLexicalExpression(const Stmt *S, const Stmt *Parent);

} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_LIB_TOOLING_REFACTOR_STMT_UTILS_H

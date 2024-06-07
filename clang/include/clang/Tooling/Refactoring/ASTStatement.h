//===--- ASTStatement.h - Clang refactoring library -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_REFACTORING_ASTSTATEMENT_H
#define LLVM_CLANG_TOOLING_REFACTORING_ASTSTATEMENT_H

#include "clang/AST/Stmt.h"
#include "clang/Basic/SourceLocation.h"

namespace clang {

class ASTContext;

namespace tooling {

/// Traverses the given ASTContext and finds closest outer statement.
///
/// \returns nullptr if location is not surrounded by any statement, or a AST
/// statement otherwise.
Stmt *findOuterStmt(const ASTContext &Context,
                    SourceLocation Location);
} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_REFACTORING_ASTSTATEMENT_H

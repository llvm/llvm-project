//===--- PrimitiveVarDecl.h - Clang refactoring library ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_REFACTORING_VARINITS_PRIMITIVEVARDECL_H
#define LLVM_CLANG_TOOLING_REFACTORING_VARINITS_PRIMITIVEVARDECL_H


namespace clang {
namespace tooling {
/// \returns a ptr to concrete DeclRefExpr (it is basically a pointer to
/// VarDecl) if given SourceLocation is in between
/// a DeclRefExpr start location and end location
/// and nullptr otherwise
DeclRefExpr *getDeclRefExprFromSourceLocation(ASTContext &AST,
                                              SourceLocation Location);

} // namespace tooling
} // namespace clang


#endif // LLVM_CLANG_TOOLING_REFACTORING_VARINITS_PRIMITIVEVARDECL_H

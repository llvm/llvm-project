//===--- DeclRefExprUtils.h - clang-tidy-------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_DECLREFEXPRUTILS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_DECLREFEXPRUTILS_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/Type.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace clang::tidy::utils::decl_ref_expr {

/// Returns set of all ``DeclRefExprs`` to ``VarDecl`` within ``Stmt``.
llvm::SmallPtrSet<const DeclRefExpr *, 16>
allDeclRefExprs(const VarDecl &VarDecl, const Stmt &Stmt, ASTContext &Context);

/// Returns set of all ``DeclRefExprs`` to ``VarDecl`` within ``Decl``.
llvm::SmallPtrSet<const DeclRefExpr *, 16>
allDeclRefExprs(const VarDecl &VarDecl, const Decl &Decl, ASTContext &Context);

/// Returns set of all ``DeclRefExprs`` to ``VarDecl`` within ``Stmt`` where
/// ``VarDecl`` is guaranteed to be accessed in a const fashion.
///
/// If ``VarDecl`` is of pointer type, ``Indirections`` specifies the level
/// of indirection of the object whose mutations we are tracking.
///
/// For example, given:
///   ```
///   int i;
///   int* p;
///   p = &i;  // (A)
///   *p = 3;  // (B)
///   ```
///
///   - `constReferenceDeclRefExprs(P, Stmt, Context, 0)` returns the reference
//      to `p` in (B): the pointee is modified, but the pointer is not;
///   - `constReferenceDeclRefExprs(P, Stmt, Context, 1)` returns the reference
//      to `p` in (A): the pointee is modified, but the pointer is not;
llvm::SmallPtrSet<const DeclRefExpr *, 16>
constReferenceDeclRefExprs(const VarDecl &VarDecl, const Stmt &Stmt,
                           ASTContext &Context, int Indirections);

/// Returns true if all ``DeclRefExpr`` to the variable within ``Stmt``
/// do not modify it.
/// See `constReferenceDeclRefExprs` for the meaning of ``Indirections``.
bool isOnlyUsedAsConst(const VarDecl &Var, const Stmt &Stmt,
                       ASTContext &Context, int Indirections);

/// Returns ``true`` if ``DeclRefExpr`` is the argument of a copy-constructor
/// call expression within ``Decl``.
bool isCopyConstructorArgument(const DeclRefExpr &DeclRef, const Decl &Decl,
                               ASTContext &Context);

/// Returns ``true`` if ``DeclRefExpr`` is the argument of a copy-assignment
/// operator CallExpr within ``Decl``.
bool isCopyAssignmentArgument(const DeclRefExpr &DeclRef, const Decl &Decl,
                              ASTContext &Context);

} // namespace clang::tidy::utils::decl_ref_expr

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_DECLREFEXPRUTILS_H

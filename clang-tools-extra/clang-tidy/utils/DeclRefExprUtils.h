//===--- DeclRefExprUtils.h - clang-tidy-------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_DECLREFEXPRUTILS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_DECLREFEXPRUTILS_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/Type.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace clang {
namespace tidy {
namespace utils {
namespace decl_ref_expr {

/// \brief Returns true if all ``DeclRefExpr`` to the variable within ``Stmt``
/// do not modify it.
///
/// Returns ``true`` if only const methods or operators are called on the
/// variable or the variable is a const reference or value argument to a
/// ``callExpr()``.
bool isOnlyUsedAsConst(const VarDecl &Var, const Stmt &Stmt,
                       ASTContext &Context);

/// Returns set of all ``DeclRefExprs`` to ``VarDecl`` within ``Stmt``.
llvm::SmallPtrSet<const DeclRefExpr *, 16>
allDeclRefExprs(const VarDecl &VarDecl, const Stmt &Stmt, ASTContext &Context);

/// Returns set of all ``DeclRefExprs`` to ``VarDecl`` within ``Decl``.
llvm::SmallPtrSet<const DeclRefExpr *, 16>
allDeclRefExprs(const VarDecl &VarDecl, const Decl &Decl, ASTContext &Context);

/// Returns set of all ``DeclRefExprs`` to ``VarDecl`` within ``Stmt`` where
/// ``VarDecl`` is guaranteed to be accessed in a const fashion.
llvm::SmallPtrSet<const DeclRefExpr *, 16>
constReferenceDeclRefExprs(const VarDecl &VarDecl, const Stmt &Stmt,
                           ASTContext &Context);

/// Returns set of all ``DeclRefExprs`` to ``VarDecl`` within ``Decl`` where
/// ``VarDecl`` is guaranteed to be accessed in a const fashion.
llvm::SmallPtrSet<const DeclRefExpr *, 16>
constReferenceDeclRefExprs(const VarDecl &VarDecl, const Decl &Decl,
                           ASTContext &Context);

/// Returns ``true`` if ``DeclRefExpr`` is the argument of a copy-constructor
/// call expression within ``Decl``.
bool isCopyConstructorArgument(const DeclRefExpr &DeclRef, const Decl &Decl,
                               ASTContext &Context);

/// Returns ``true`` if ``DeclRefExpr`` is the argument of a copy-assignment
/// operator CallExpr within ``Decl``.
bool isCopyAssignmentArgument(const DeclRefExpr &DeclRef, const Decl &Decl,
                              ASTContext &Context);

} // namespace decl_ref_expr
} // namespace utils
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_DECLREFEXPRUTILS_H

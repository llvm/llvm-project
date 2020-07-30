//===------------- Aliasing.cpp - clang-tidy ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Aliasing.h"

#include "clang/AST/Expr.h"

namespace clang {
namespace tidy {
namespace utils {

/// Return whether \p S is a reference to the declaration of \p Var.
static bool isAccessForVar(const Stmt *S, const VarDecl *Var) {
  if (const auto *DRE = dyn_cast<DeclRefExpr>(S))
    return DRE->getDecl() == Var;

  return false;
}

/// Return whether \p Var has a pointer or reference in \p S.
static bool isPtrOrReferenceForVar(const Stmt *S, const VarDecl *Var) {
  if (const auto *DS = dyn_cast<DeclStmt>(S)) {
    for (const Decl *D : DS->getDeclGroup()) {
      if (const auto *LeftVar = dyn_cast<VarDecl>(D)) {
        if (LeftVar->hasInit() && LeftVar->getType()->isReferenceType()) {
          return isAccessForVar(LeftVar->getInit(), Var);
        }
      }
    }
  } else if (const auto *UnOp = dyn_cast<UnaryOperator>(S)) {
    if (UnOp->getOpcode() == UO_AddrOf)
      return isAccessForVar(UnOp->getSubExpr(), Var);
  }

  return false;
}

/// Return whether \p Var has a pointer or reference in \p S.
static bool hasPtrOrReferenceInStmt(const Stmt *S, const VarDecl *Var) {
  if (isPtrOrReferenceForVar(S, Var))
    return true;

  for (const Stmt *Child : S->children()) {
    if (!Child)
      continue;

    if (hasPtrOrReferenceInStmt(Child, Var))
      return true;
  }

  return false;
}

bool hasPtrOrReferenceInFunc(const FunctionDecl *Func, const VarDecl *Var) {
  return hasPtrOrReferenceInStmt(Func->getBody(), Var);
}

} // namespace utils
} // namespace tidy
} // namespace clang

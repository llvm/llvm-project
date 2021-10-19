//===------------- Aliasing.cpp - clang-tidy ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Aliasing.h"

#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"

namespace clang {
namespace tidy {
namespace utils {

/// Return whether \p S is a reference to the declaration of \p Var.
static bool isAccessForVar(const Stmt *S, const VarDecl *Var) {
  if (const auto *DRE = dyn_cast<DeclRefExpr>(S))
    return DRE->getDecl() == Var;

  return false;
}

static bool capturesByRef(const CXXRecordDecl *RD, const VarDecl *Var) {
  return llvm::any_of(RD->captures(), [Var](const LambdaCapture &C) {
    return C.capturesVariable() && C.getCaptureKind() == LCK_ByRef &&
           C.getCapturedVar() == Var;
  });
}

/// Return whether \p Var has a pointer or reference in \p S.
static bool isPtrOrReferenceForVar(const Stmt *S, const VarDecl *Var) {
  // Treat block capture by reference as a form of taking a reference.
  if (Var->isEscapingByref())
    return true;

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
  } else if (const auto *LE = dyn_cast<LambdaExpr>(S)) {
    // Treat lambda capture by reference as a form of taking a reference.
    return capturesByRef(LE->getLambdaClass(), Var);
  } else if (const auto *ILE = dyn_cast<InitListExpr>(S)) {
    return llvm::any_of(ILE->inits(), [Var](const Expr *ChildE) {
      // If the child expression is a reference to Var, this means that it's
      // used as an initializer of a reference-typed field. Otherwise
      // it would have been surrounded with an implicit lvalue-to-rvalue cast.
      return isAccessForVar(ChildE, Var);
    });
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

static bool refersToEnclosingLambdaCaptureByRef(const Decl *Func,
                                                const VarDecl *Var) {
  const auto *MD = dyn_cast<CXXMethodDecl>(Func);
  if (!MD)
    return false;

  const CXXRecordDecl *RD = MD->getParent();
  if (!RD->isLambda())
    return false;

  return capturesByRef(RD, Var);
}

bool hasPtrOrReferenceInFunc(const Decl *Func, const VarDecl *Var) {
  return hasPtrOrReferenceInStmt(Func->getBody(), Var) ||
         refersToEnclosingLambdaCaptureByRef(Func, Var);
}

} // namespace utils
} // namespace tidy
} // namespace clang

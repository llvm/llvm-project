//=======- ASTUtils.cpp ------------------------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ASTUtils.h"
#include "PtrTypesSemantics.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"

using llvm::Optional;
namespace clang {

std::pair<const Expr *, bool>
tryToFindPtrOrigin(const Expr *E, bool StopAtFirstRefCountedObj) {
  while (E) {
    if (auto *cast = dyn_cast<CastExpr>(E)) {
      if (StopAtFirstRefCountedObj) {
        if (auto *ConversionFunc =
                dyn_cast_or_null<FunctionDecl>(cast->getConversionFunction())) {
          if (isCtorOfRefCounted(ConversionFunc))
            return {E, true};
        }
      }
      // FIXME: This can give false "origin" that would lead to false negatives
      // in checkers. See https://reviews.llvm.org/D37023 for reference.
      E = cast->getSubExpr();
      continue;
    }
    if (auto *call = dyn_cast<CallExpr>(E)) {
      if (auto *memberCall = dyn_cast<CXXMemberCallExpr>(call)) {
        Optional<bool> IsGetterOfRefCt =
            isGetterOfRefCounted(memberCall->getMethodDecl());
        if (IsGetterOfRefCt && *IsGetterOfRefCt) {
          E = memberCall->getImplicitObjectArgument();
          if (StopAtFirstRefCountedObj) {
            return {E, true};
          }
          continue;
        }
      }

      if (auto *operatorCall = dyn_cast<CXXOperatorCallExpr>(E)) {
        if (operatorCall->getNumArgs() == 1) {
          E = operatorCall->getArg(0);
          continue;
        }
      }

      if (auto *callee = call->getDirectCallee()) {
        if (isCtorOfRefCounted(callee)) {
          if (StopAtFirstRefCountedObj)
            return {E, true};

          E = call->getArg(0);
          continue;
        }

        if (isPtrConversion(callee)) {
          E = call->getArg(0);
          continue;
        }
      }
    }
    if (auto *unaryOp = dyn_cast<UnaryOperator>(E)) {
      // FIXME: Currently accepts ANY unary operator. Is it OK?
      E = unaryOp->getSubExpr();
      continue;
    }

    break;
  }
  // Some other expression.
  return {E, false};
}

bool isASafeCallArg(const Expr *E) {
  assert(E);
  if (auto *Ref = dyn_cast<DeclRefExpr>(E)) {
    if (auto *D = dyn_cast_or_null<VarDecl>(Ref->getFoundDecl())) {
      if (isa<ParmVarDecl>(D) || D->isLocalVarDecl())
        return true;
    }
  }

  // TODO: checker for method calls on non-refcounted objects
  return isa<CXXThisExpr>(E);
}

} // namespace clang

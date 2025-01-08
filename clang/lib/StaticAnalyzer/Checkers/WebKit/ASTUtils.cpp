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
#include "clang/AST/ExprObjC.h"
#include "clang/AST/StmtVisitor.h"
#include <optional>

namespace clang {

bool isSafePtr(clang::CXXRecordDecl *Decl) {
  return isRefCounted(Decl) || isCheckedPtr(Decl);
}

bool tryToFindPtrOrigin(
    const Expr *E, bool StopAtFirstRefCountedObj,
    std::function<bool(const clang::Expr *, bool)> callback) {
  while (E) {
    if (auto *tempExpr = dyn_cast<MaterializeTemporaryExpr>(E)) {
      E = tempExpr->getSubExpr();
      continue;
    }
    if (auto *tempExpr = dyn_cast<CXXBindTemporaryExpr>(E)) {
      E = tempExpr->getSubExpr();
      continue;
    }
    if (auto *tempExpr = dyn_cast<CXXConstructExpr>(E)) {
      if (auto *C = tempExpr->getConstructor()) {
        if (auto *Class = C->getParent(); Class && isSafePtr(Class))
          return callback(E, true);
        break;
      }
    }
    if (auto *POE = dyn_cast<PseudoObjectExpr>(E)) {
      if (auto *RF = POE->getResultExpr()) {
        E = RF;
        continue;
      }
    }
    if (auto *tempExpr = dyn_cast<ParenExpr>(E)) {
      E = tempExpr->getSubExpr();
      continue;
    }
    if (auto *Expr = dyn_cast<ConditionalOperator>(E)) {
      return tryToFindPtrOrigin(Expr->getTrueExpr(), StopAtFirstRefCountedObj,
                                callback) &&
             tryToFindPtrOrigin(Expr->getFalseExpr(), StopAtFirstRefCountedObj,
                                callback);
    }
    if (auto *cast = dyn_cast<CastExpr>(E)) {
      if (StopAtFirstRefCountedObj) {
        if (auto *ConversionFunc =
                dyn_cast_or_null<FunctionDecl>(cast->getConversionFunction())) {
          if (isCtorOfSafePtr(ConversionFunc))
            return callback(E, true);
        }
      }
      // FIXME: This can give false "origin" that would lead to false negatives
      // in checkers. See https://reviews.llvm.org/D37023 for reference.
      E = cast->getSubExpr();
      continue;
    }
    if (auto *call = dyn_cast<CallExpr>(E)) {
      if (auto *memberCall = dyn_cast<CXXMemberCallExpr>(call)) {
        if (auto *decl = memberCall->getMethodDecl()) {
          std::optional<bool> IsGetterOfRefCt = isGetterOfSafePtr(decl);
          if (IsGetterOfRefCt && *IsGetterOfRefCt) {
            E = memberCall->getImplicitObjectArgument();
            if (StopAtFirstRefCountedObj) {
              return callback(E, true);
            }
            continue;
          }
        }
      }

      if (auto *operatorCall = dyn_cast<CXXOperatorCallExpr>(E)) {
        if (operatorCall->getNumArgs() == 1) {
          E = operatorCall->getArg(0);
          continue;
        }
      }

      if (auto *callee = call->getDirectCallee()) {
        if (isCtorOfRefCounted(callee) || isCtorOfCheckedPtr(callee)) {
          if (StopAtFirstRefCountedObj)
            return callback(E, true);

          E = call->getArg(0);
          continue;
        }

        if (isSafePtrType(callee->getReturnType()))
          return callback(E, true);

        if (isSingleton(callee))
          return callback(E, true);

        if (callee->isInStdNamespace() && safeGetName(callee) == "forward") {
          E = call->getArg(0);
          continue;
        }

        if (isPtrConversion(callee)) {
          E = call->getArg(0);
          continue;
        }
      }
    }
    if (auto *ObjCMsgExpr = dyn_cast<ObjCMessageExpr>(E)) {
      if (auto *Method = ObjCMsgExpr->getMethodDecl()) {
        if (isSafePtrType(Method->getReturnType()))
          return callback(E, true);
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
  return callback(E, false);
}

bool isASafeCallArg(const Expr *E) {
  assert(E);
  if (auto *Ref = dyn_cast<DeclRefExpr>(E)) {
    if (auto *D = dyn_cast_or_null<VarDecl>(Ref->getFoundDecl())) {
      if (isa<ParmVarDecl>(D) || D->isLocalVarDecl())
        return true;
    }
  }
  if (isConstOwnerPtrMemberExpr(E))
    return true;

  // TODO: checker for method calls on non-refcounted objects
  return isa<CXXThisExpr>(E);
}

bool isConstOwnerPtrMemberExpr(const clang::Expr *E) {
  if (auto *MCE = dyn_cast<CXXMemberCallExpr>(E)) {
    if (auto *Callee = MCE->getDirectCallee()) {
      auto Name = safeGetName(Callee);
      if (Name == "get" || Name == "ptr") {
        auto *ThisArg = MCE->getImplicitObjectArgument();
        E = ThisArg;
      }
    }
  } else if (auto *OCE = dyn_cast<CXXOperatorCallExpr>(E)) {
    if (OCE->getOperator() == OO_Star && OCE->getNumArgs() == 1)
      E = OCE->getArg(0);
  }
  auto *ME = dyn_cast<MemberExpr>(E);
  if (!ME)
    return false;
  auto *D = ME->getMemberDecl();
  if (!D)
    return false;
  auto T = D->getType();
  return isOwnerPtrType(T) && T.isConstQualified();
}

class EnsureFunctionVisitor
    : public ConstStmtVisitor<EnsureFunctionVisitor, bool> {
public:
  bool VisitStmt(const Stmt *S) {
    for (const Stmt *Child : S->children()) {
      if (Child && !Visit(Child))
        return false;
    }
    return true;
  }

  bool VisitReturnStmt(const ReturnStmt *RS) {
    if (auto *RV = RS->getRetValue()) {
      RV = RV->IgnoreParenCasts();
      if (isa<CXXNullPtrLiteralExpr>(RV))
        return true;
      return isConstOwnerPtrMemberExpr(RV);
    }
    return false;
  }
};

bool EnsureFunctionAnalysis::isACallToEnsureFn(const clang::Expr *E) const {
  auto *MCE = dyn_cast<CXXMemberCallExpr>(E);
  if (!MCE)
    return false;
  auto *Callee = MCE->getDirectCallee();
  if (!Callee)
    return false;
  auto *Body = Callee->getBody();
  if (!Body)
    return false;
  auto [CacheIt, IsNew] = Cache.insert(std::make_pair(Callee, false));
  if (IsNew)
    CacheIt->second = EnsureFunctionVisitor().Visit(Body);
  return CacheIt->second;
}

} // namespace clang

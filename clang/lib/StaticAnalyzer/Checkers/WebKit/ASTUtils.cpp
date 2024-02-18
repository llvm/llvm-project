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
#include "clang/AST/ParentMapContext.h"
#include <optional>

namespace clang {

std::pair<const Expr *, bool>
tryToFindPtrOrigin(const Expr *E, bool StopAtFirstRefCountedObj) {
  while (E) {
    if (auto *tempExpr = dyn_cast<MaterializeTemporaryExpr>(E)) {
      E = tempExpr->getSubExpr();
      continue;
    }
    if (auto *tempExpr = dyn_cast<CXXBindTemporaryExpr>(E)) {
      E = tempExpr->getSubExpr();
      continue;
    }
    if (auto *DRE = dyn_cast<DeclRefExpr>(E)) {
      auto *decl = DRE->getFoundDecl();
      if (auto *VD = dyn_cast<VarDecl>(decl)) {
        if (isTypeRefCounted(VD->getType()))
          return {E, true};
      }
    }
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
        if (auto *decl = memberCall->getMethodDecl()) {
          std::optional<bool> IsGetterOfRefCt =
              isGetterOfRefCounted(memberCall->getMethodDecl());
          if (IsGetterOfRefCt && *IsGetterOfRefCt) {
            E = memberCall->getImplicitObjectArgument();
            if (StopAtFirstRefCountedObj) {
              return {E, true};
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
        if (isCtorOfRefCounted(callee)) {
          if (StopAtFirstRefCountedObj)
            return {E, true};

          E = call->getArg(0);
          continue;
        }

        if (isReturnValueRefCounted(callee))
          return {E, true};

        if (isSingleton(callee))
          return {E, true};

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

bool isGuardedScopeEmbeddedInGuardianScope(const VarDecl *Guarded,
                                           const VarDecl *MaybeGuardian) {
  assert(Guarded);
  assert(MaybeGuardian);

  if (!MaybeGuardian->isLocalVarDecl())
    return false;

  const CompoundStmt *guardiansClosestCompStmtAncestor = nullptr;

  ASTContext &ctx = MaybeGuardian->getASTContext();

  for (DynTypedNodeList guardianAncestors = ctx.getParents(*MaybeGuardian);
       !guardianAncestors.empty();
       guardianAncestors = ctx.getParents(
           *guardianAncestors
                .begin()) // FIXME - should we handle all of the parents?
  ) {
    for (auto &guardianAncestor : guardianAncestors) {
      if (auto *CStmtParentAncestor = guardianAncestor.get<CompoundStmt>()) {
        guardiansClosestCompStmtAncestor = CStmtParentAncestor;
        break;
      }
    }
    if (guardiansClosestCompStmtAncestor)
      break;
  }

  if (!guardiansClosestCompStmtAncestor)
    return false;

  // We need to skip the first CompoundStmt to avoid situation when guardian is
  // defined in the same scope as guarded variable.
  bool HaveSkippedFirstCompoundStmt = false;
  for (DynTypedNodeList guardedVarAncestors = ctx.getParents(*Guarded);
       !guardedVarAncestors.empty();
       guardedVarAncestors = ctx.getParents(
           *guardedVarAncestors
                .begin()) // FIXME - should we handle all of the parents?
  ) {
    for (auto &guardedVarAncestor : guardedVarAncestors) {
      if (guardedVarAncestor.get<ForStmt>()) {
        if (!HaveSkippedFirstCompoundStmt)
          HaveSkippedFirstCompoundStmt = true;
        continue;
      }
      if (guardedVarAncestor.get<IfStmt>()) {
        if (!HaveSkippedFirstCompoundStmt)
          HaveSkippedFirstCompoundStmt = true;
        continue;
      }
      if (guardedVarAncestor.get<WhileStmt>()) {
        if (!HaveSkippedFirstCompoundStmt)
          HaveSkippedFirstCompoundStmt = true;
        continue;
      }
      if (auto *CStmtAncestor = guardedVarAncestor.get<CompoundStmt>()) {
        if (!HaveSkippedFirstCompoundStmt) {
          HaveSkippedFirstCompoundStmt = true;
          continue;
        }
        if (CStmtAncestor == guardiansClosestCompStmtAncestor)
          return true;
      }
    }
  }

  return false;
}

// FIXME: should be defined by annotations in the future
bool isRefcountedStringsHack(const VarDecl *V) {
  assert(V);
  auto safeClass = [](const std::string &className) {
    return className == "String" || className == "AtomString" ||
           className == "UniquedString" || className == "Identifier";
  };
  QualType QT = V->getType();
  auto *T = QT.getTypePtr();
  if (auto *CXXRD = T->getAsCXXRecordDecl()) {
    if (safeClass(safeGetName(CXXRD)))
      return true;
  }
  if (T->isPointerType() || T->isReferenceType()) {
    if (auto *CXXRD = T->getPointeeCXXRecordDecl()) {
      if (safeClass(safeGetName(CXXRD)))
        return true;
    }
  }
  return false;
}

bool isVarDeclGuardedInit(const VarDecl *V, const Expr *InitE) {
  // "this" parameter like any other argument is considered safe.
  if (isa<CXXThisExpr>(InitE))
    return true;

  if (auto *Ref = llvm::dyn_cast<DeclRefExpr>(InitE)) {
    if (auto *MaybeGuardian = dyn_cast_or_null<VarDecl>(Ref->getFoundDecl())) {
      // Parameters are guaranteed to be safe for the duration of the call.
      if (isa<ParmVarDecl>(MaybeGuardian))
        return true;

      const auto *MaybeGuardianArgType = MaybeGuardian->getType().getTypePtr();
      if (!MaybeGuardianArgType)
        return false;

      const CXXRecordDecl *const MaybeGuardianArgCXXRecord =
          MaybeGuardianArgType->getAsCXXRecordDecl();

      if (!MaybeGuardianArgCXXRecord)
        return false;

      if (MaybeGuardian->isLocalVarDecl() &&
          (isRefCounted(MaybeGuardianArgCXXRecord) ||
           isRefcountedStringsHack(MaybeGuardian)) &&
          isGuardedScopeEmbeddedInGuardianScope(V, MaybeGuardian)) {
        return true;
      }
    }
  }

  return false;
}

bool isASafeCallArg(const Expr *E) {
  assert(E);
  if (auto *Ref = dyn_cast<DeclRefExpr>(E)) {
    if (auto *D = dyn_cast_or_null<VarDecl>(Ref->getFoundDecl())) {
      if (isa<ParmVarDecl>(D))
        return true;
    }
  }

  // TODO: checker for method calls on non-refcounted objects
  return isa<CXXThisExpr>(E);
}

} // namespace clang

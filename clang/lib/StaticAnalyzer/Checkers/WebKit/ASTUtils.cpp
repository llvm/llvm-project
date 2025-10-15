//=======- ASTUtils.cpp ------------------------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ASTUtils.h"
#include "PtrTypesSemantics.h"
#include "clang/AST/Attr.h"
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
    std::function<bool(const clang::CXXRecordDecl *)> isSafePtr,
    std::function<bool(const clang::QualType)> isSafePtrType,
    std::function<bool(const clang::Decl *)> isSafeGlobalDecl,
    std::function<bool(const clang::Expr *, bool)> callback) {
  while (E) {
    if (auto *DRE = dyn_cast<DeclRefExpr>(E)) {
      if (auto *VD = dyn_cast_or_null<VarDecl>(DRE->getDecl())) {
        auto QT = VD->getType();
        auto IsImmortal = safeGetName(VD) == "NSApp";
        if (VD->hasGlobalStorage() && (IsImmortal || QT.isConstQualified()))
          return callback(E, true);
        if (VD->hasGlobalStorage() && isSafeGlobalDecl(VD))
          return callback(E, true);
      }
    }
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
    if (auto *TempExpr = dyn_cast<CXXUnresolvedConstructExpr>(E)) {
      if (isSafePtrType(TempExpr->getTypeAsWritten()))
        return callback(TempExpr, true);
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
    if (auto *OpaqueValue = dyn_cast<OpaqueValueExpr>(E)) {
      E = OpaqueValue->getSourceExpr();
      continue;
    }
    if (auto *Expr = dyn_cast<ConditionalOperator>(E)) {
      return tryToFindPtrOrigin(Expr->getTrueExpr(), StopAtFirstRefCountedObj,
                                isSafePtr, isSafePtrType, isSafeGlobalDecl,
                                callback) &&
             tryToFindPtrOrigin(Expr->getFalseExpr(), StopAtFirstRefCountedObj,
                                isSafePtr, isSafePtrType, isSafeGlobalDecl,
                                callback);
    }
    if (auto *cast = dyn_cast<CastExpr>(E)) {
      if (StopAtFirstRefCountedObj) {
        if (auto *ConversionFunc =
                dyn_cast_or_null<FunctionDecl>(cast->getConversionFunction())) {
          if (isCtorOfSafePtr(ConversionFunc))
            return callback(E, true);
        }
        if (isa<CXXFunctionalCastExpr>(E) && isSafePtrType(cast->getType()))
          return callback(E, true);
      }
      // FIXME: This can give false "origin" that would lead to false negatives
      // in checkers. See https://reviews.llvm.org/D37023 for reference.
      E = cast->getSubExpr();
      continue;
    }
    if (auto *call = dyn_cast<CallExpr>(E)) {
      if (auto *Callee = call->getCalleeDecl()) {
        if (Callee->hasAttr<CFReturnsRetainedAttr>() ||
            Callee->hasAttr<NSReturnsRetainedAttr>() ||
            Callee->hasAttr<NSReturnsAutoreleasedAttr>()) {
          return callback(E, true);
        }
      }

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
        if (auto *Callee = operatorCall->getDirectCallee()) {
          auto ClsName = safeGetName(Callee->getParent());
          if (isRefType(ClsName) || isCheckedPtr(ClsName) ||
              isRetainPtrOrOSPtr(ClsName) || ClsName == "unique_ptr" ||
              ClsName == "UniqueRef" || ClsName == "WeakPtr" ||
              ClsName == "WeakRef") {
            if (operatorCall->getNumArgs() == 1) {
              E = operatorCall->getArg(0);
              continue;
            }
          }
        }
      }

      if (call->isCallToStdMove() && call->getNumArgs() == 1) {
        E = call->getArg(0)->IgnoreParenCasts();
        continue;
      }

      if (auto *callee = call->getDirectCallee()) {
        if (isCtorOfSafePtr(callee)) {
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

        auto Name = safeGetName(callee);
        if (Name == "__builtin___CFStringMakeConstantString" ||
            Name == "NSClassFromString")
          return callback(E, true);
      } else if (auto *CalleeE = call->getCallee()) {
        if (auto *E = dyn_cast<DeclRefExpr>(CalleeE->IgnoreParenCasts())) {
          if (isSingleton(E->getFoundDecl()))
            return callback(E, true);
        }
      }

      // Sometimes, canonical type erroneously turns Ref<T> into T.
      // Workaround this problem by checking again if the original type was
      // a SubstTemplateTypeParmType of a safe smart pointer type (e.g. Ref).
      if (auto *CalleeDecl = call->getCalleeDecl()) {
        if (auto *FD = dyn_cast<FunctionDecl>(CalleeDecl)) {
          auto RetType = FD->getReturnType();
          if (auto *Subst = dyn_cast<SubstTemplateTypeParmType>(RetType)) {
            if (auto *SubstType = Subst->desugar().getTypePtr()) {
              if (auto *RD = dyn_cast<RecordType>(SubstType)) {
                if (auto *CXX = dyn_cast<CXXRecordDecl>(RD->getDecl()))
                  if (isSafePtr(CXX))
                    return callback(E, true);
              }
            }
          }
        }
      }
    }
    if (auto *ObjCMsgExpr = dyn_cast<ObjCMessageExpr>(E)) {
      if (auto *Method = ObjCMsgExpr->getMethodDecl()) {
        if (isSafePtrType(Method->getReturnType()))
          return callback(E, true);
      }
      auto Selector = ObjCMsgExpr->getSelector();
      auto NameForFirstSlot = Selector.getNameForSlot(0);
      if ((NameForFirstSlot == "class" || NameForFirstSlot == "superclass") &&
          !Selector.getNumArgs())
        return callback(E, true);
    }
    if (auto *ObjCDict = dyn_cast<ObjCDictionaryLiteral>(E))
      return callback(ObjCDict, true);
    if (auto *ObjCArray = dyn_cast<ObjCArrayLiteral>(E))
      return callback(ObjCArray, true);
    if (auto *ObjCStr = dyn_cast<ObjCStringLiteral>(E))
      return callback(ObjCStr, true);
    if (auto *unaryOp = dyn_cast<UnaryOperator>(E)) {
      // FIXME: Currently accepts ANY unary operator. Is it OK?
      E = unaryOp->getSubExpr();
      continue;
    }
    if (auto *BoxedExpr = dyn_cast<ObjCBoxedExpr>(E)) {
      if (StopAtFirstRefCountedObj)
        return callback(BoxedExpr, true);
      E = BoxedExpr->getSubExpr();
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
    auto *FoundDecl = Ref->getFoundDecl();
    if (auto *D = dyn_cast_or_null<VarDecl>(FoundDecl)) {
      if (isa<ParmVarDecl>(D) || D->isLocalVarDecl())
        return true;
      if (auto *ImplicitP = dyn_cast<ImplicitParamDecl>(D)) {
        auto Kind = ImplicitP->getParameterKind();
        if (Kind == ImplicitParamKind::ObjCSelf ||
            Kind == ImplicitParamKind::ObjCCmd ||
            Kind == ImplicitParamKind::CXXThis ||
            Kind == ImplicitParamKind::CXXVTT)
          return true;
      }
    } else if (auto *BD = dyn_cast_or_null<BindingDecl>(FoundDecl)) {
      VarDecl *VD = BD->getHoldingVar();
      if (VD && (isa<ParmVarDecl>(VD) || VD->isLocalVarDecl()))
        return true;
    }
  }
  if (isa<CXXTemporaryObjectExpr>(E))
    return true; // A temporary lives until the end of this statement.
  if (isConstOwnerPtrMemberExpr(E))
    return true;

  // TODO: checker for method calls on non-refcounted objects
  return isa<CXXThisExpr>(E);
}

bool isNullPtr(const clang::Expr *E) {
  if (isa<CXXNullPtrLiteralExpr>(E) || isa<GNUNullExpr>(E))
    return true;
  if (auto *Int = dyn_cast_or_null<IntegerLiteral>(E)) {
    if (Int->getValue().isZero())
      return true;
  }
  return false;
}

bool isConstOwnerPtrMemberExpr(const clang::Expr *E) {
  if (auto *MCE = dyn_cast<CXXMemberCallExpr>(E)) {
    if (auto *Callee = MCE->getDirectCallee()) {
      auto Name = safeGetName(Callee);
      if (Name == "get" || Name == "ptr")
        E = MCE->getImplicitObjectArgument();
      if (isa<CXXConversionDecl>(Callee))
        E = MCE->getImplicitObjectArgument();
    }
  } else if (auto *OCE = dyn_cast<CXXOperatorCallExpr>(E)) {
    if (OCE->getOperator() == OO_Star && OCE->getNumArgs() == 1)
      E = OCE->getArg(0);
  }
  const ValueDecl *D = nullptr;
  if (auto *ME = dyn_cast<MemberExpr>(E))
    D = ME->getMemberDecl();
  else if (auto *IVR = dyn_cast<ObjCIvarRefExpr>(E))
    D = IVR->getDecl();
  if (!D)
    return false;
  auto T = D->getType();
  return isOwnerPtrType(T) && T.isConstQualified();
}

bool isExprToGetCheckedPtrCapableMember(const clang::Expr *E) {
  auto *ME = dyn_cast<MemberExpr>(E);
  if (!ME)
    return false;
  auto *Base = ME->getBase();
  if (!Base)
    return false;
  if (!isa<CXXThisExpr>(Base->IgnoreParenCasts()))
    return false;
  auto *D = ME->getMemberDecl();
  if (!D)
    return false;
  auto T = D->getType();
  auto *CXXRD = T->getAsCXXRecordDecl();
  if (!CXXRD)
    return false;
  auto result = isCheckedPtrCapable(CXXRD);
  return result && *result;
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
      if (isNullPtr(RV))
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
  if (!Body || Callee->isVirtualAsWritten())
    return false;
  auto [CacheIt, IsNew] = Cache.insert(std::make_pair(Callee, false));
  if (IsNew)
    CacheIt->second = EnsureFunctionVisitor().Visit(Body);
  return CacheIt->second;
}

} // namespace clang

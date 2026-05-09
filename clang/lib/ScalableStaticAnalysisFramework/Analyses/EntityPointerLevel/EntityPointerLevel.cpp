//===- EntityPointerLevel.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysisFramework/Analyses/EntityPointerLevel/EntityPointerLevel.h"
#include "SSAFAnalysesCommon.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/ScalableStaticAnalysisFramework/Core/TUSummary/TUSummaryExtractor.h"
#include <optional>

using namespace clang;
using namespace ssaf;

namespace clang::ssaf {
// Translate a pointer type expression 'E' to a (set of) EntityPointerLevel(s)
// associated with the declared type of the base address of `E`. If the base
// address of `E` is not associated with an entity, the translation result is an
// empty set.
//
// The translation is a process of traversing into the pointer 'E' until its
// base address can be represented by an entity, with the number of dereferences
// tracked by incrementing the pointer level.  Naturally, taking address of, as
// the inverse operation of dereference, is tracked by decrementing the pointer
// level.
//
// For example, suppose there are pointers and arrays declared as
//   int *ptr, **p1, **p2;
//   int arr[10][10];
// , the translation of expressions involving these base addresses will be:
//   Translate(ptr + 5)            -> {(ptr, 1)}
//   Translate(arr[5])             -> {(arr, 2)}
//   Translate(cond ? p1[5] : p2)  -> {(p1, 2), (p2, 1)}
//   Translate(&arr[5])            -> {(arr, 1)}
class EntityPointerLevelTranslator
    : ConstStmtVisitor<EntityPointerLevelTranslator,
                       Expected<EntityPointerLevelSet>> {
  friend class StmtVisitorBase;

  // Fallback method for all unsupported expression kind:
  llvm::Error fallback(const Stmt *E) {
    return makeErrAtNode(Ctx, E,
                         "attempt to translate %s to EntityPointerLevels",
                         E->getStmtClassName());
  }

  Expected<EntityPointerLevel>
  createEntityPointerLevelFor(const NamedDecl *ND) {
    std::optional<EntityId> Id = Extractor.addEntity(ND);
    if (!Id)
      return makeErrAtNode(Ctx, ND, "failed to create EntityId for %s",
                           ND->getDeclKindName());
    return EntityPointerLevel{buildEntityPointerLevel(*Id, 1)};
  }

  Expected<EntityPointerLevel>
  createEntityPointerLevelForReturn(const FunctionDecl *FD) {
    std::optional<EntityId> Id = Extractor.addEntityForReturn(FD);
    if (!Id) {
      return makeErrAtNode(Ctx, FD, "failed to create EntityId for function %s",
                           cast<NamedDecl>(FD)->getNameAsString().c_str());
    }
    return EntityPointerLevel{buildEntityPointerLevel(*Id, 1)};
  }

  // The common helper function for Translate(*base):
  // Translate(*base) -> Translate(base) with .pointerLevel + 1
  Expected<EntityPointerLevelSet> translateDereferencePointer(const Expr *Ptr) {
    assert(hasPtrOrArrType(Ptr));

    Expected<EntityPointerLevelSet> SubResult = Visit(Ptr);
    if (!SubResult)
      return SubResult.takeError();

    auto Incremented = llvm::map_range(*SubResult, incrementPointerLevel);
    return EntityPointerLevelSet{Incremented.begin(), Incremented.end()};
  }

  TUSummaryExtractor &Extractor;
  ASTContext &Ctx;

public:
  EntityPointerLevelTranslator(TUSummaryExtractor &Extractor, ASTContext &Ctx)
      : Extractor(Extractor), Ctx(Ctx) {}

  Expected<EntityPointerLevelSet> translate(const Expr *E) { return Visit(E); }
  Expected<EntityPointerLevel> translate(const NamedDecl *D, bool IsRet) {
    if (!IsRet)
      return createEntityPointerLevelFor(D);

    if (const auto *FD = dyn_cast<FunctionDecl>(D))
      return createEntityPointerLevelForReturn(FD);

    return makeErrAtNode(Ctx, D, "attempt to get entity for return of %s",
                         D->getDeclKindName());
  }

  static EntityPointerLevel incrementPointerLevel(const EntityPointerLevel &E) {
    return EntityPointerLevel({E.getEntity(), E.getPointerLevel() + 1});
  }

  static EntityPointerLevel decrementPointerLevel(const EntityPointerLevel &E) {
    assert(E.getPointerLevel() > 0);
    return EntityPointerLevel({E.getEntity(), E.getPointerLevel() - 1});
  }

private:
  Expected<EntityPointerLevelSet> VisitStmt(const Stmt *E) {
    return fallback(E);
  }

  // Translate(base + x)           -> Translate(base)
  // Translate(x + base)           -> Translate(base)
  // Translate(base - x)           -> Translate(base)
  // Translate(base {+=, -=, =} x) -> Translate(base)
  // Translate(x, base)            -> Translate(base)
  Expected<EntityPointerLevelSet> VisitBinaryOperator(const BinaryOperator *E) {
    switch (E->getOpcode()) {
    case clang::BO_Add:
      if (hasPtrOrArrType(E->getLHS()))
        return Visit(E->getLHS());
      return Visit(E->getRHS());
    case clang::BO_Sub:
    case clang::BO_AddAssign:
    case clang::BO_SubAssign:
    case clang::BO_Assign:
      return Visit(E->getLHS());
    case clang::BO_Comma:
      return Visit(E->getRHS());
    default:
      return fallback(E);
    }
  }

  // Translate({++, --}base)   -> Translate(base)
  // Translate(base{++, --})   -> Translate(base)
  // Translate(*base)          -> Translate(base) with .pointerLevel += 1
  // Translate(&base)          -> {}, if Translate(base) is {}
  //                           -> Translate(base) with .pointerLevel -= 1
  Expected<EntityPointerLevelSet> VisitUnaryOperator(const UnaryOperator *E) {
    switch (E->getOpcode()) {
    case clang::UO_PostInc:
    case clang::UO_PostDec:
    case clang::UO_PreInc:
    case clang::UO_PreDec:
      return Visit(E->getSubExpr());
    case clang::UO_AddrOf: {
      Expected<EntityPointerLevelSet> SubResult = Visit(E->getSubExpr());
      if (!SubResult)
        return SubResult.takeError();

      auto Decremented = llvm::map_range(*SubResult, decrementPointerLevel);
      return EntityPointerLevelSet{Decremented.begin(), Decremented.end()};
    }
    case clang::UO_Deref:
      return translateDereferencePointer(E->getSubExpr());
    default:
      return fallback(E);
    }
  }

  // Translate((T*)base) -> Translate(base) if base has pointer type
  //                     -> {} otherwise
  Expected<EntityPointerLevelSet> VisitCastExpr(const CastExpr *E) {
    if (hasPtrOrArrType(E->getSubExpr()))
      return Visit(E->getSubExpr());
    return EntityPointerLevelSet{};
  }

  // Translate(f(...)) -> {} if it is an indirect call
  //                   -> {(f_return, 1)}, otherwise
  Expected<EntityPointerLevelSet> VisitCallExpr(const CallExpr *E) {
    if (auto *FD = E->getDirectCallee()) {
      if (auto ReturnId = Extractor.addEntityForReturn(FD))
        return EntityPointerLevelSet{buildEntityPointerLevel(*ReturnId, 1)};
    }
    return EntityPointerLevelSet{};
  }

  // Translate(base[x]) -> Translate(*base)
  Expected<EntityPointerLevelSet>
  VisitArraySubscriptExpr(const ArraySubscriptExpr *E) {
    return translateDereferencePointer(E->getBase());
  }

  // Translate(cond ? base1 : base2) := Translate(base1) U Translate(base2)
  Expected<EntityPointerLevelSet>
  VisitAbstractConditionalOperator(const AbstractConditionalOperator *E) {
    Expected<EntityPointerLevelSet> ReT = Visit(E->getTrueExpr());
    Expected<EntityPointerLevelSet> ReF = Visit(E->getFalseExpr());

    if (ReT && ReF) {
      ReT->insert(ReF->begin(), ReF->end());
      return ReT;
    }
    if (!ReF && !ReT)
      return llvm::joinErrors(ReT.takeError(), ReF.takeError());
    if (!ReF)
      return ReF.takeError();
    return ReT.takeError();
  }

  Expected<EntityPointerLevelSet> VisitParenExpr(const ParenExpr *E) {
    return Visit(E->getSubExpr());
  }

  // Translate("string-literal") -> {}
  // Buffer accesses on string literals are unsafe, but string literals are not
  // entities so there is no EntityPointerLevel associated with it.
  Expected<EntityPointerLevelSet> VisitStringLiteral(const StringLiteral *E) {
    return EntityPointerLevelSet{};
  }

  // Translate(DRE) -> {(Decl, 1)}
  Expected<EntityPointerLevelSet> VisitDeclRefExpr(const DeclRefExpr *E) {
    auto Res = createEntityPointerLevelFor(E->getDecl());
    if (!Res)
      return Res.takeError();
    return EntityPointerLevelSet{*Res};
  }

  // Translate({., ->}f) -> {(MemberDecl, 1)}
  Expected<EntityPointerLevelSet> VisitMemberExpr(const MemberExpr *E) {
    auto Res = createEntityPointerLevelFor(E->getMemberDecl());
    if (!Res)
      return Res.takeError();
    return EntityPointerLevelSet{*Res};
  }

  // Translate(`DefaultArg`) -> Translate(`DefaultArg->getExpr()`)
  Expected<EntityPointerLevelSet>
  VisitCXXDefaultArgExpr(const CXXDefaultArgExpr *E) {
    return Visit(E->getExpr());
  }

  Expected<EntityPointerLevelSet>
  VisitOpaqueValueExpr(const OpaqueValueExpr *S) {
    return Visit(S->getSourceExpr());
  }
};
} // namespace clang::ssaf

Expected<EntityPointerLevelSet>
clang::ssaf::translateEntityPointerLevel(const Expr *E, ASTContext &Ctx,
                                         TUSummaryExtractor &Extractor) {
  EntityPointerLevelTranslator Translator(Extractor, Ctx);

  return Translator.translate(E);
}

/// Create an EntityPointerLevel from a ValueDecl of a pointer type.
Expected<EntityPointerLevel> clang::ssaf::createEntityPointerLevel(
    const NamedDecl *ND, TUSummaryExtractor &Extractor, bool IsFunRet) {
  EntityPointerLevelTranslator Translator(Extractor, ND->getASTContext());

  return Translator.translate(ND, IsFunRet);
}

EntityPointerLevel
clang::ssaf::incrementPointerLevel(const EntityPointerLevel &E) {
  return EntityPointerLevelTranslator::incrementPointerLevel(E);
}

EntityPointerLevel clang::ssaf::buildEntityPointerLevel(EntityId Id,
                                                        unsigned PtrLv) {
  return EntityPointerLevel({Id, PtrLv});
}

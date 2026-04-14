//===- EntityPointerLevel.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysisFramework/Analyses/EntityPointerLevel/EntityPointerLevel.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/ScalableStaticAnalysisFramework/Core/ASTEntityMapping.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/EntityName.h"

using namespace clang;
using namespace ssaf;

static bool hasPointerType(const Expr *E) {
  auto Ty = E->getType();
  return !Ty.isNull() && !Ty->isFunctionPointerType() &&
         (Ty->isPointerType() || Ty->isArrayType());
}

static llvm::Error makeUnsupportedStmtKindError(const Stmt *Unsupported) {
  return llvm::createStringError(
      "unsupported expression kind for translation to "
      "EntityPointerLevel: %s",
      Unsupported->getStmtClassName());
}

static llvm::Error makeCreateEntityNameError(const NamedDecl *FailedDecl,
                                             ASTContext &Ctx) {
  std::string LocStr = FailedDecl->getSourceRange().getBegin().printToString(
      Ctx.getSourceManager());
  return llvm::createStringError(
      "failed to create entity name for %s declared at %s",
      FailedDecl->getNameAsString().c_str(), LocStr.c_str());
}

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
    return makeUnsupportedStmtKindError(E);
  }

  static EntityPointerLevel incrementPointerLevel(const EntityPointerLevel &E) {
    return EntityPointerLevel(E.getEntity(), E.getPointerLevel() + 1);
  }

  static EntityPointerLevel decrementPointerLevel(const EntityPointerLevel &E) {
    assert(E.getPointerLevel() > 0);
    return EntityPointerLevel(E.getEntity(), E.getPointerLevel() - 1);
  }

  EntityPointerLevel createEntityPointerLevelFor(const EntityName &Name) {
    return EntityPointerLevel(AddEntity(Name), 1);
  }

  // The common helper function for Translate(*base):
  // Translate(*base) -> Translate(base) with .pointerLevel + 1
  Expected<EntityPointerLevelSet> translateDereferencePointer(const Expr *Ptr) {
    assert(hasPointerType(Ptr));

    Expected<EntityPointerLevelSet> SubResult = Visit(Ptr);
    if (!SubResult)
      return SubResult.takeError();

    auto Incremented = llvm::map_range(*SubResult, incrementPointerLevel);
    return EntityPointerLevelSet{Incremented.begin(), Incremented.end()};
  }

  std::function<EntityId(EntityName EN)> AddEntity;
  ASTContext &Ctx;

public:
  EntityPointerLevelTranslator(std::function<EntityId(EntityName EN)> AddEntity,
                               ASTContext &Ctx)
      : AddEntity(AddEntity), Ctx(Ctx) {}

  Expected<EntityPointerLevelSet> translate(const Expr *E) { return Visit(E); }

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
      if (hasPointerType(E->getLHS()))
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

  // Translate((T*)base) -> Translate(p) if p has pointer type
  //                     -> {} otherwise
  Expected<EntityPointerLevelSet> VisitCastExpr(const CastExpr *E) {
    if (hasPointerType(E->getSubExpr()))
      return Visit(E->getSubExpr());
    return EntityPointerLevelSet{};
  }

  // Translate(f(...)) -> {} if it is an indirect call
  //                   -> {(f_return, 1)}, otherwise
  Expected<EntityPointerLevelSet> VisitCallExpr(const CallExpr *E) {
    if (auto *FD = E->getDirectCallee())
      if (auto FDEntityName = getEntityNameForReturn(FD))
        return EntityPointerLevelSet{
            createEntityPointerLevelFor(*FDEntityName)};
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
    if (auto EntityName = getEntityName(E->getDecl()))
      return EntityPointerLevelSet{createEntityPointerLevelFor(*EntityName)};
    return makeCreateEntityNameError(E->getDecl(), Ctx);
  }

  // Translate({., ->}f) -> {(MemberDecl, 1)}
  Expected<EntityPointerLevelSet> VisitMemberExpr(const MemberExpr *E) {
    if (auto EntityName = getEntityName(E->getMemberDecl()))
      return EntityPointerLevelSet{createEntityPointerLevelFor(*EntityName)};
    return makeCreateEntityNameError(E->getMemberDecl(), Ctx);
  }

  Expected<EntityPointerLevelSet>
  VisitOpaqueValueExpr(const OpaqueValueExpr *S) {
    return Visit(S->getSourceExpr());
  }
};
} // namespace clang::ssaf

Expected<EntityPointerLevelSet> clang::ssaf::translateEntityPointerLevel(
    const Expr *E, ASTContext &Ctx,
    std::function<EntityId(EntityName EN)> AddEntity) {
  EntityPointerLevelTranslator Translator(AddEntity, Ctx);

  return Translator.translate(E);
}

EntityPointerLevel clang::ssaf::buildEntityPointerLevel(EntityId Id,
                                                        unsigned PtrLv) {
  return EntityPointerLevel({Id, PtrLv});
}

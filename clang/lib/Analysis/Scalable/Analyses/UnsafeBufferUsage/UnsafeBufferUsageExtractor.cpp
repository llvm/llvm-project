//===- UnsafeBufferUsage.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Scalable/Analyses/UnsafeBufferUsage/UnsafeBufferUsageExtractor.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DynamicRecursiveASTVisitor.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Analysis/Analyses/UnsafeBufferUsage.h"
#include "clang/Analysis/Scalable/ASTEntityMapping.h"
#include "clang/Analysis/Scalable/Analyses/UnsafeBufferUsage/UnsafeBufferUsage.h"
#include "clang/Analysis/Scalable/Analyses/UnsafeBufferUsage/UnsafeBufferUsageBuilder.h"
#include "clang/Analysis/Scalable/Model/EntityId.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Error.h"
#include <memory>

namespace {
using namespace clang;
using namespace ssaf;

template <typename ExprOrDecl> static bool hasPointerType(const ExprOrDecl *E) {
  auto Ty = E->getType();
  return !Ty.isNull() && !Ty->isFunctionPointerType() &&
         (Ty->isPointerType() || Ty->isArrayType());
}

constexpr inline auto buildEntityPointerLevel =
    UnsafeBufferUsageTUSummaryBuilder::buildEntityPointerLevel;

// Translate a pointer type expression 'E' to a (set of) EntityPointerLevel(s)
// associated with the declared type of the base address of `E`.
//
// The translation is a process of stripping off the pointer 'E' until its base
// address can be represented by an entity, with the number of dereferences
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
    : public ConstStmtVisitor<EntityPointerLevelTranslator,
                              Expected<EntityPointerLevelSet>> {
  // Fallback method for all unsupported expression kind:
  llvm::Error fallback(const Stmt *E) {
    return llvm::createStringError(
        "unsupported expression kind for translation to "
        "EntityPointerLevel: %s",
        E->getStmtClassName());
  }

  UnsafeBufferUsageTUSummaryBuilder &Builder;

  EntityPointerLevel incrementPointerLevel(const EntityPointerLevel &E) {
    return buildEntityPointerLevel(E.getEntity(), E.getPointerLevel() + 1);
  }

  EntityPointerLevel decrementPointerLevel(const EntityPointerLevel &E) {
    assert(E.getPointerLevel() > 0);
    return buildEntityPointerLevel(E.getEntity(), E.getPointerLevel() - 1);
  }

  EntityPointerLevel createEntityPointerLevelFor(const EntityName &Name) {
    return buildEntityPointerLevel(Builder.addEntity(Name), 1);
  }

  // The common helper function for Translate(*base):
  // Translate(*base) -> Translate(base) with .pointerLevel + 1
  Expected<EntityPointerLevelSet> translateDereferencePointer(const Expr *Ptr) {
    assert(hasPointerType(Ptr));

    Expected<EntityPointerLevelSet> SubResult = Visit(Ptr);

    if (!SubResult)
      return SubResult.takeError();
    if (SubResult->empty())
      return SubResult;

    EntityPointerLevelSet Result{};

    for (EntityPointerLevel DereffedPtr : *SubResult)
      Result.insert(incrementPointerLevel(DereffedPtr));
    return Result;
  }

public:
  EntityPointerLevelTranslator(UnsafeBufferUsageTUSummaryBuilder &Builder)
      : Builder(Builder) {}

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

      EntityPointerLevelSet Result;

      for (auto Elt : *SubResult)
        Result.insert(decrementPointerLevel(Elt));
      return Result;
    }
    case clang::UO_Deref:
      return translateDereferencePointer(E->getSubExpr());
    default:
      return fallback(E);
    }
  }

  // Translate((T*)base) -> Translate(p) if p has pointer type
  //                  -> {} otherwise
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
    // Translate(ptr[x]) := Translate(*ptr)
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
    return llvm::createStringError(
        "failed to create entity name from the Decl of " +
        E->getNameInfo().getAsString());
  }

  // Translate({., ->}f) -> {(MemberDecl, 1)}
  Expected<EntityPointerLevelSet> VisitMemberExpr(const MemberExpr *E) {
    if (auto EntityName = getEntityName(E->getMemberDecl()))
      return EntityPointerLevelSet{createEntityPointerLevelFor(*EntityName)};
    return llvm::createStringError(
        "failed to create entity name from the MemberDecl of " +
        E->getMemberDecl()->getNameAsString());
  }

  Expected<EntityPointerLevelSet>
  VisitOpaqueValueExpr(const OpaqueValueExpr *S) {
    return Visit(S->getSourceExpr());
  }
};

EntityPointerLevelSet
buildEntityPointerLevels(std::set<const Expr *> &&UnsafePointers,
                         UnsafeBufferUsageTUSummaryBuilder &Builder) {
  EntityPointerLevelSet Result{};
  EntityPointerLevelTranslator Translator{Builder};

  for (const Expr *Ptr : UnsafePointers) {
    Expected<EntityPointerLevelSet> Translation = Translator.Visit(Ptr);

    if (Translation) {
      // Filter out those invalid EntityPointerLevels associated with `&E`
      // pointers:
      auto FilteredTranslation = llvm::make_filter_range(
          *Translation, [](const EntityPointerLevel &V) -> bool {
            return V.getPointerLevel() > 0;
          });
      Result.insert(FilteredTranslation.begin(), FilteredTranslation.end());
      continue;
    }
#ifndef NDEBUG
    // FIXME: Log error message
#endif
    llvm::consumeError(Translation.takeError());
  }
  return Result;
}
} // namespace

std::unique_ptr<UnsafeBufferUsageEntitySummary>
UnsafeBufferUsageTUSummaryExtractor::extractEntitySummary(
    EntityId Contributor, const Decl *ContributorDefn, ASTContext &Ctx) {
  return getBuilder().buildUnsafeBufferUsageEntitySummary(
      buildEntityPointerLevels(findUnsafePointers(ContributorDefn),
                               getBuilder()));
}

//===- EntityPointerLevel.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysis/Analyses/EntityPointerLevel/EntityPointerLevel.h"
#include "SSAFAnalysesCommon.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/TUSummaryExtractor.h"
#include "llvm/ADT/STLExtras.h"
#include <optional>
#include <vector>

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
                       Expected<DeclPointerLevels>> {
  friend class StmtVisitorBase;

  // Fallback method for all unsupported expression kind:
  Expected<DeclPointerLevels> fallback(const Stmt *S) {
    // Report an error/warning (at least in debug mode) for any unsupported kind
    // of pointer/array typed expression, because we want to understand every
    // pointer/array expression. But for non-pointer/array typed expressions, we
    // could silently ignore unsupported kinds. This translator visits
    // non-pointer/array typed expressions because of address-of expressions.
    if (const Expr *E = dyn_cast<Expr>(S); E && hasPtrOrArrType(E))
      return makeErrAtNode(Ctx, E,
                           "attempt to translate %s to EntityPointerLevels",
                           E->getStmtClassName());
    return DeclPointerLevels{};
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
  Expected<DeclPointerLevels> translateDereferencePointer(const Expr *Ptr) {
    assert(hasPtrOrArrType(Ptr));

    Expected<DeclPointerLevels> SubResult = Visit(Ptr);
    if (!SubResult)
      return SubResult.takeError();

    llvm::for_each(*SubResult, [](DeclPointerLevel &D) { ++D.PointerLevel; });
    return SubResult;
  }

  TUSummaryExtractor &Extractor;
  ASTContext &Ctx;

public:
  EntityPointerLevelTranslator(TUSummaryExtractor &Extractor, ASTContext &Ctx)
      : Extractor(Extractor), Ctx(Ctx) {}

  Expected<DeclPointerLevels> translate(const Expr *E) { return Visit(E); }
  Expected<EntityPointerLevel> translate(const NamedDecl *D, bool IsRet) {
    if (!IsRet)
      return createEntityPointerLevelFor(D);

    if (const auto *FD = dyn_cast<FunctionDecl>(D))
      return createEntityPointerLevelForReturn(FD);

    return makeErrAtNode(Ctx, D, "attempt to get entity for return of %s",
                         D->getDeclKindName());
  }

  // Converts a `DeclPointerLevel` to an `EntityPointerLevel`
  Expected<EntityPointerLevel> toEntityPointerLevel(const DeclPointerLevel &D) {
    Expected<EntityPointerLevel> Base = translate(D.Decl, D.IsReturn);
    if (!Base)
      return Base.takeError();
    return buildEntityPointerLevel(Base->getEntity(), D.PointerLevel);
  }

private:
  Expected<DeclPointerLevels> VisitStmt(const Stmt *E) { return fallback(E); }

  // Translate(base + x)           -> Translate(base)
  // Translate(x + base)           -> Translate(base)
  // Translate(base - x)           -> Translate(base)
  // Translate(base {+=, -=, =} x) -> Translate(base)
  // Translate(x, base)            -> Translate(base)
  Expected<DeclPointerLevels> VisitBinaryOperator(const BinaryOperator *E) {
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
  // Translate(+base)          -> Translate(base)
  Expected<DeclPointerLevels> VisitUnaryOperator(const UnaryOperator *E) {
    switch (E->getOpcode()) {
    case clang::UO_PostInc:
    case clang::UO_PostDec:
    case clang::UO_PreInc:
    case clang::UO_PreDec:
      return Visit(E->getSubExpr());
    case clang::UO_AddrOf: {
      Expected<DeclPointerLevels> SubResult = Visit(E->getSubExpr());
      if (!SubResult)
        return SubResult.takeError();

      llvm::for_each(*SubResult, [](DeclPointerLevel &D) {
        assert(D.PointerLevel > 0);
        --D.PointerLevel;
      });
      return SubResult;
    }
    case clang::UO_Deref:
      return translateDereferencePointer(E->getSubExpr());
    case clang::UO_Plus:
      return Visit(E->getSubExpr());
    default:
      return fallback(E);
    }
  }

  // Translate((T*)base) -> Translate(base) if base has pointer type
  //                     -> {} otherwise
  Expected<DeclPointerLevels> VisitCastExpr(const CastExpr *E) {
    if (hasPtrOrArrType(E->getSubExpr()))
      return Visit(E->getSubExpr());
    return DeclPointerLevels{};
  }

  // Translate(f(...)) -> {} if it is an indirect call
  //                   -> {(f_return, 1)}, otherwise
  Expected<DeclPointerLevels> VisitCallExpr(const CallExpr *E) {
    if (auto *FD = E->getDirectCallee())
      if (Extractor.addEntityForReturn(FD))
        return DeclPointerLevels{{FD, /*PointerLevel=*/1, /*IsReturn=*/true}};
    return DeclPointerLevels{};
  }

  // Translate(base[x]) -> Translate(*base)
  Expected<DeclPointerLevels>
  VisitArraySubscriptExpr(const ArraySubscriptExpr *E) {
    return translateDereferencePointer(E->getBase());
  }

  // Translate(cond ? base1 : base2) := Translate(base1) U Translate(base2)
  Expected<DeclPointerLevels>
  VisitAbstractConditionalOperator(const AbstractConditionalOperator *E) {
    Expected<DeclPointerLevels> ReT = Visit(E->getTrueExpr());
    Expected<DeclPointerLevels> ReF = Visit(E->getFalseExpr());

    if (ReT && ReF) {
      ReT->insert(ReT->end(), ReF->begin(), ReF->end());
      return ReT;
    }
    if (!ReF && !ReT)
      return llvm::joinErrors(ReT.takeError(), ReF.takeError());
    if (!ReF)
      return ReF.takeError();
    return ReT.takeError();
  }

  Expected<DeclPointerLevels> VisitParenExpr(const ParenExpr *E) {
    return Visit(E->getSubExpr());
  }

  // Translate("string-literal") -> {} // no entity involved
  Expected<DeclPointerLevels> VisitStringLiteral(const StringLiteral *E) {
    return DeclPointerLevels{};
  }

  // Translate(predefined-expr) -> {} // treated the same as string literals
  Expected<DeclPointerLevels> VisitPredefinedExpr(const PredefinedExpr *E) {
    return DeclPointerLevels{};
  }

  // Translate(integer-literal) -> {} // no entity involved
  Expected<DeclPointerLevels> VisitIntegerLiteral(const IntegerLiteral *E) {
    return DeclPointerLevels{};
  }

  // Translate(DRE) -> {(Decl, 1)}
  Expected<DeclPointerLevels> VisitDeclRefExpr(const DeclRefExpr *E) {
    return DeclPointerLevels{
        {E->getDecl(), /*PointerLevel=*/1, /*IsReturn=*/false}};
  }

  // Translate({., ->}f) -> {(MemberDecl, 1)}
  Expected<DeclPointerLevels> VisitMemberExpr(const MemberExpr *E) {
    return DeclPointerLevels{
        {E->getMemberDecl(), /*PointerLevel=*/1, /*IsReturn=*/false}};
  }

  // Unwrap CXXDefaultArgExpr
  Expected<DeclPointerLevels>
  VisitCXXDefaultArgExpr(const CXXDefaultArgExpr *E) {
    return Visit(E->getExpr());
  }

  // Unwrap OpaqueValueExpr
  Expected<DeclPointerLevels> VisitOpaqueValueExpr(const OpaqueValueExpr *S) {
    return Visit(S->getSourceExpr());
  }

  // Unwrap ExprWithCleanups
  Expected<DeclPointerLevels> VisitExprWithCleanups(const ExprWithCleanups *S) {
    return Visit(S->getSubExpr());
  }

  // Unwrap MaterializeTemporaryExpr
  Expected<DeclPointerLevels>
  VisitMaterializeTemporaryExpr(const MaterializeTemporaryExpr *S) {
    return Visit(S->getSubExpr());
  }

  // Unwrap CXXDefaultInitExpr
  Expected<DeclPointerLevels>
  VisitCXXDefaultInitExpr(const CXXDefaultInitExpr *E) {
    return Visit(E->getExpr());
  }

  // Translate(`nullptr`) -> {}
  Expected<DeclPointerLevels>
  VisitCXXNullPtrLiteralExpr(const CXXNullPtrLiteralExpr *S) {
    return DeclPointerLevels{};
  }

  // Translate(`this`) -> {}
  Expected<DeclPointerLevels> VisitCXXThisExpr(const CXXThisExpr *S) {
    return DeclPointerLevels{};
  }

  // Translate(`new`/`new [*]`) -> {}
  Expected<DeclPointerLevels> VisitCXXNewExpr(const CXXNewExpr *S) {
    return DeclPointerLevels{};
  }

  // ImplicitValueInitExpr, for raw pointer type,
  // evaluates to a compile-time constant zero (or null). So no EPL in the
  // result.
  Expected<DeclPointerLevels>
  VisitImplicitValueInitExpr(const ImplicitValueInitExpr *S) {
    return DeclPointerLevels{};
  }

  // The InitListExpr must be an empty or singleton list that
  // initializes a pointer scalar.  Other cases are unexpected thus an error.
  Expected<DeclPointerLevels> VisitInitListExpr(const InitListExpr *E) {
    if (E->getNumInits() < 1)
      return DeclPointerLevels{};
    if (E->getType()->isPointerType())
      return Visit(E->getInit(0));
    return llvm::createStringError(
        "Cannot translate an InitListExpr to EntityPointerLevels if it is not "
        "an empty or singleton list that initializes a pointer scalar");
  }

  // Clang may default initializes an array with a CXXConstructExpr. Fallback on
  // other cases, if they exist.
  // When a CXXConstructExpr has an array type, clang is initializing an array
  // of class-type objects with default values.  In this case, no entity is
  // associated with the initializer.
  Expected<DeclPointerLevels> VisitCXXConstructExpr(const CXXConstructExpr *E) {
    if (E->getType()->isArrayType()) {
      return DeclPointerLevels{};
    }
    return fallback(E);
  }

  // No entity is associated with a CXXScalarValueInitExpr:
  Expected<DeclPointerLevels>
  VisitCXXScalarValueInitExpr(const CXXScalarValueInitExpr *E) {
    return DeclPointerLevels{};
  }
};
} // namespace clang::ssaf

Expected<DeclPointerLevels>
clang::ssaf::translateDeclPointerLevel(const Expr *E, ASTContext &Ctx,
                                       TUSummaryExtractor &Extractor) {
  EntityPointerLevelTranslator Translator(Extractor, Ctx);

  return Translator.translate(E);
}

Expected<EntityPointerLevelSet>
clang::ssaf::translateEntityPointerLevel(const Expr *E, ASTContext &Ctx,
                                         TUSummaryExtractor &Extractor) {
  EntityPointerLevelTranslator Translator(Extractor, Ctx);
  auto DPLs = Translator.translate(E);
  if (!DPLs)
    return DPLs.takeError();
  return toEntityPointerLevels(*DPLs, Ctx, Extractor);
}

DeclPointerLevel clang::ssaf::createDeclPointerLevel(const NamedDecl *ND,
                                                     bool IsFunRet) {
  return {ND, 1, IsFunRet};
}

/// Create an EntityPointerLevel from a ValueDecl of a pointer type.
Expected<EntityPointerLevel> clang::ssaf::createEntityPointerLevel(
    const NamedDecl *ND, TUSummaryExtractor &Extractor, bool IsFunRet) {
  EntityPointerLevelTranslator Translator(Extractor, ND->getASTContext());

  return Translator.translate(ND, IsFunRet);
}

DeclPointerLevels
clang::ssaf::elaborateHigherDeclPointerLevels(const DeclPointerLevel &DPL) {
  DeclPointerLevels Result{DPL};
  QualType T;

  if (DPL.IsReturn) {
    if (const auto *FD = dyn_cast<FunctionDecl>(DPL.Decl))
      T = FD->getReturnType().getNonReferenceType();
  } else if (const auto *VD = dyn_cast<ValueDecl>(DPL.Decl)) {
    T = VD->getType().getNonReferenceType();
  }
  if (T.isNull())
    return Result;

  // Count the max pointer/array levels of `T`:
  unsigned MaxLevel = 0;
  for (T = T.getCanonicalType();; ++MaxLevel) {
    if (const auto *PT = dyn_cast<PointerType>(T))
      T = PT->getPointeeType().getCanonicalType();
    else if (const auto *AT = dyn_cast<ArrayType>(T))
      T = AT->getElementType().getCanonicalType();
    else
      break;
  }

  for (unsigned Level = DPL.PointerLevel + 1; Level <= MaxLevel; ++Level)
    Result.push_back({DPL.Decl, Level, DPL.IsReturn});
  return Result;
}

Expected<EntityPointerLevelSet>
clang::ssaf::toEntityPointerLevels(const DeclPointerLevels &DPLs,
                                   ASTContext &Ctx,
                                   TUSummaryExtractor &Extractor) {
  EntityPointerLevelTranslator Translator(Extractor, Ctx);
  EntityPointerLevelSet Result;

  for (const auto &DPL : DPLs) {
    Expected<EntityPointerLevel> EPL = Translator.toEntityPointerLevel(DPL);
    if (!EPL)
      return EPL.takeError();
    Result.insert(*EPL);
  }
  return Result;
}

Expected<EntityPointerLevel>
clang::ssaf::toEntityPointerLevel(const DeclPointerLevel &DPL, ASTContext &Ctx,
                                  TUSummaryExtractor &Extractor) {
  EntityPointerLevelTranslator Translator(Extractor, Ctx);
  return Translator.toEntityPointerLevel(DPL);
}

EntityPointerLevel clang::ssaf::buildEntityPointerLevel(EntityId Id,
                                                        unsigned PtrLv) {
  return EntityPointerLevel({Id, PtrLv});
}

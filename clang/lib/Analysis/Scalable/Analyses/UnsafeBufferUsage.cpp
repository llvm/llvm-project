//===- UnsafeBufferUsage.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Scalable/Analyses/UnsafeBufferUsage.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DynamicRecursiveASTVisitor.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Analysis/Analyses/UnsafeBufferUsage.h"
#include "clang/Analysis/Scalable/ASTEntityMapping.h"
#include "clang/Analysis/Scalable/Model/EntityId.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include <memory>

namespace {
using namespace clang;
using namespace llvm;
using namespace clang::ssaf;
template <typename ExprOrDecl> static bool hasPointerType(const ExprOrDecl *E) {
  auto Ty = E->getType();
  return !Ty.isNull() && !Ty->isFunctionPointerType() &&
         (Ty->isPointerType() || Ty->isArrayType());
}

// Translate a pointer type expression `E` to a (set of) PointerKindVariable(s)
// associated with the declared type of the base address.
// For example,
// Translate(ptr + 5)            -> {(ptr, 1)}
// Translate(ptr[5])             -> {(ptr, 2)} (ptr[5] has pointer type)
// Translate(cond ? p1[5] : p2)  -> {(p1, 2), (p2, 1)}
class PointerKindVariableTranslator
    : public ConstStmtVisitor<PointerKindVariableTranslator,
                              Expected<PointerKindVariableSet>> {
  UnsafeBufferUsageTUSummaryBuilder &Builder;

  // Fallback method for all unsupported expression kind:
  llvm::Error fallback(const Stmt *E) {
    return llvm::createStringError(
        "unsupported expression kind for translation to "
        "PointerKindVariable: %s",
        E->getStmtClassName());
  }

  // The common helper function for Translate(*p/p[x]).
  //
  //  Translate(*ptr)   -> Translate(ptr) with .pointerLevel += 1
  //  Translate(ptr[x]) -> Translate(*ptr)
  //
  //  For example, for `int **P`, if `*P` is an unsafe pointer, we are
  //  interested in the PointerVariableKind '(P, 2)', which is associated
  //  with the second `*` from the right in `int **P`.
  Expected<PointerKindVariableSet>
  translateDereferencePointer(const Expr *Ptr) {
    assert(hasPointerType(Ptr));

    Expected<PointerKindVariableSet> SubResult = Visit(Ptr);

    if (!SubResult)
      return SubResult.takeError();
    if (SubResult->empty())
      return SubResult;

    PointerKindVariableSet Result{};

    for (PointerKindVariable DereffedPtr : *SubResult)
      Result.insert(Builder.buildPointerKindVariable(
          DereffedPtr.getEntity(), DereffedPtr.getPointerLevel() + 1));
    return Result;
  }

public:
  PointerKindVariableTranslator(UnsafeBufferUsageTUSummaryBuilder &Builder)
      : Builder(Builder) {}

  Expected<PointerKindVariableSet> VisitStmt(const Stmt *E) {
    return fallback(E);
  }

  // Translate(ptr + x/ptr - x)           -> Translate(ptr)
  // Translate(ptr += x/ptr -= x/ptr = x) -> Translate(ptr)
  // Translate(x, ptr)                    -> Translate(ptr)
  Expected<PointerKindVariableSet>
  VisitBinaryOperator(const BinaryOperator *E) {
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

  // Translate(++p/p++/--p/p--)         -> Translate(p)
  // Translate(*p)                      -> Translate(p) with .pointerLevel += 1
  // Translate(&x)                      -> {}, if Translate(x) is {}
  //                                    -> Translate(x) with .pointerLevel -= 1
  Expected<PointerKindVariableSet> VisitUnaryOperator(const UnaryOperator *E) {
    switch (E->getOpcode()) {
    case clang::UO_PostInc:
    case clang::UO_PostDec:
    case clang::UO_PreInc:
    case clang::UO_PreDec:
      return Visit(E->getSubExpr());
    case clang::UO_AddrOf: {
      Expected<PointerKindVariableSet> SubResult = Visit(E->getSubExpr());

      if (!SubResult)
        return SubResult.takeError();

      PointerKindVariableSet Result;

      for (auto PKV : *SubResult) {
        assert(PKV.getPointerLevel() > 0);
        Result.insert(Builder.buildPointerKindVariable(
            PKV.getEntity(), PKV.getPointerLevel() - 1));
      }
      return Result;
    }
    case clang::UO_Deref:
      return translateDereferencePointer(E->getSubExpr());
    default:
      return fallback(E);
    }
  }

  // Translate((T*)p) -> Translate(p) if p has pointer type
  //                  -> {} otherwise
  Expected<PointerKindVariableSet> VisitCastExpr(const CastExpr *E) {
    if (hasPointerType(E->getSubExpr()))
      return Visit(E->getSubExpr());
    return PointerKindVariableSet{};
  }

  // Translate(f(...)) -> {} if it is an indirect call
  //                   -> PointerKindVariable(f_return, 1), otherwise
  Expected<PointerKindVariableSet> VisitCallExpr(const CallExpr *E) {
    if (auto *FD = E->getDirectCallee())
      if (auto FDEntityName = getEntityNameForReturn(FD)) {
        EntityId Entity = Builder.addEntity(*FDEntityName);

        return PointerKindVariableSet{
            Builder.buildPointerKindVariable(Entity, 1)};
      }
    return PointerKindVariableSet{};
  }

  // Translate(p[x]) -> Translate(*p)
  Expected<PointerKindVariableSet>
  VisitArraySubscriptExpr(const ArraySubscriptExpr *E) {
    // Translate(ptr[x]) := Translate(*ptr)
    return translateDereferencePointer(E->getLHS());
  }

  // Translate(cond ? p1 : p2) := Translate(p1) U Translate(p2)
  Expected<PointerKindVariableSet>
  VisitAbstractConditionalOperator(const AbstractConditionalOperator *E) {
    Expected<PointerKindVariableSet> ReT = Visit(E->getTrueExpr());
    Expected<PointerKindVariableSet> ReF = Visit(E->getFalseExpr());

    if (ReT && ReF) {
      ReT->insert(ReF->begin(), ReF->end());
      return ReT;
    }
    if (!ReF)
      return ReF.takeError();
    return ReT.takeError();
  }

  Expected<PointerKindVariableSet> VisitParenExpr(const ParenExpr *E) {
    return Visit(E->getSubExpr());
  }

  // Translate("string-literal") -> {}
  // Buffer accesses on string literals are unsafe, but string literals are not
  // entities so there is no PointerKindVariable associated with it.
  Expected<PointerKindVariableSet>
  VisitStringLiteral(const clang::StringLiteral *E) {
    return PointerKindVariableSet{};
  }

  // Translate(DRE) -> {(Decl, 1)}
  Expected<PointerKindVariableSet> VisitDeclRefExpr(const DeclRefExpr *E) {
    if (auto EntityName = getEntityName(E->getDecl()))
      return PointerKindVariableSet{
          Builder.buildPointerKindVariable(Builder.addEntity(*EntityName), 1)};
    return llvm::createStringError(
        "failed to create entity name from the Decl of " +
        E->getNameInfo().getAsString());
  }

  // Translate(.f/->f) -> {(MemberDecl, 1)}
  Expected<PointerKindVariableSet> VisitMemberExpr(const MemberExpr *E) {
    if (auto EntityName = getEntityName(E->getMemberDecl()))
      return PointerKindVariableSet{
          Builder.buildPointerKindVariable(Builder.addEntity(*EntityName), 1)};
    return llvm::createStringError(
        "failed to create entity name from the MemberDecl of " +
        E->getMemberDecl()->getNameAsString());
  }

  Expected<PointerKindVariableSet>
  VisitOpaqueValueExpr(const OpaqueValueExpr *S) {
    return Visit(S->getSourceExpr());
  }
};

PointerKindVariableSet
buildPointerKindVariables(std::set<const Expr *> UnsafePointers,
                          UnsafeBufferUsageTUSummaryBuilder &Builder) {
  PointerKindVariableSet Result{};
  PointerKindVariableTranslator Translator{Builder};

  for (const Expr *Ptr : UnsafePointers) {
    Expected<PointerKindVariableSet> Translation = Translator.Visit(Ptr);

    if (Translation) {
      // Filter out those invalid PointerKindVariables associated with `&E`
      // pointers:
      auto FilteredTranslation = llvm::make_filter_range(
          *Translation, [](const PointerKindVariable &V) -> bool {
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

namespace clang::ssaf {

std::unique_ptr<UnsafeBufferUsageEntitySummary>
UnsafeBufferUsageTUSummaryExtractor::extractEntitySummary(
    EntityId Contributor, const Decl *ContributorDefn, ASTContext &Ctx) {
  switch (ContributorDefn->getKind()) {
  case Decl::Kind::Function: {
    const auto *FD = cast<FunctionDecl>(ContributorDefn);

    assert(FD->hasBody());
    return builder().buildUnsafeBufferUsageEntitySummary(
        Contributor,
        buildPointerKindVariables(findUnsafePointers(FD), builder()));
  }
  // FIXME: Add more contributor entity kinds
  default:
#ifndef NDEBUG
    // FIXME: Log missing case
#endif
    return nullptr;
  }
}
} // namespace clang::ssaf
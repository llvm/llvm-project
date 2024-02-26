//=== PolymorphicPtrArithmetic.cpp ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This checker reports pointer arithmetic operations on arrays of
// polymorphic objects, where the array has the type of its base class.
// Corresponds to the CTR56-CPP. Do not use pointer arithmetic on
// polymorphic objects
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/BugReporter/CommonBugCategories.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/DynamicType.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramStateTrait.h"

using namespace clang;
using namespace ento;

namespace {
class PolymorphicPtrArithmeticChecker
    : public Checker<check::PreStmt<BinaryOperator>,
                     check::PreStmt<ArraySubscriptExpr>> {
  const BugType BT{this,
                   "Pointer arithmetic on polymorphic objects is undefined"};

protected:
  class PtrCastVisitor : public BugReporterVisitor {
  public:
    void Profile(llvm::FoldingSetNodeID &ID) const override {
      static int X = 0;
      ID.AddPointer(&X);
    }
    PathDiagnosticPieceRef VisitNode(const ExplodedNode *N,
                                     BugReporterContext &BRC,
                                     PathSensitiveBugReport &BR) override;
  };

  void checkTypedExpr(const Expr *E, CheckerContext &C) const;

public:
  void checkPreStmt(const BinaryOperator *B, CheckerContext &C) const;
  void checkPreStmt(const ArraySubscriptExpr *B, CheckerContext &C) const;
};
} // namespace

void PolymorphicPtrArithmeticChecker::checkPreStmt(const BinaryOperator *B,
                                                   CheckerContext &C) const {
  if (!B->isAdditiveOp())
    return;

  bool IsLHSPtr = B->getLHS()->getType()->isAnyPointerType();
  bool IsRHSPtr = B->getRHS()->getType()->isAnyPointerType();
  if (!IsLHSPtr && !IsRHSPtr)
    return;

  const Expr *E = IsLHSPtr ? B->getLHS() : B->getRHS();

  checkTypedExpr(E, C);
}

void PolymorphicPtrArithmeticChecker::checkPreStmt(const ArraySubscriptExpr *B,
                                                   CheckerContext &C) const {
  bool IsLHSPtr = B->getLHS()->getType()->isAnyPointerType();
  bool IsRHSPtr = B->getRHS()->getType()->isAnyPointerType();
  if (!IsLHSPtr && !IsRHSPtr)
    return;

  const Expr *E = IsLHSPtr ? B->getLHS() : B->getRHS();

  checkTypedExpr(E, C);
}

void PolymorphicPtrArithmeticChecker::checkTypedExpr(const Expr *E,
                                                     CheckerContext &C) const {
  const MemRegion *MR = C.getSVal(E).getAsRegion();
  if (!MR)
    return;

  const auto *BaseClassRegion = MR->getAs<TypedValueRegion>();
  const auto *DerivedClassRegion = MR->getBaseRegion()->getAs<SymbolicRegion>();
  if (!BaseClassRegion || !DerivedClassRegion)
    return;

  const auto *BaseClass = BaseClassRegion->getValueType()->getAsCXXRecordDecl();
  const auto *DerivedClass =
      DerivedClassRegion->getSymbol()->getType()->getPointeeCXXRecordDecl();
  if (!BaseClass || !DerivedClass)
    return;

  if (!BaseClass->hasDefinition() || !DerivedClass->hasDefinition())
    return;

  if (!DerivedClass->isDerivedFrom(BaseClass))
    return;

  ExplodedNode *N = C.generateNonFatalErrorNode();
  if (!N)
    return;

  SmallString<256> Buf;
  llvm::raw_svector_ostream OS(Buf);

  QualType SourceType = BaseClassRegion->getValueType();
  QualType TargetType =
      DerivedClassRegion->getSymbol()->getType()->getPointeeType();

  OS << "Doing pointer arithmetic with '" << TargetType.getAsString()
     << "' objects as their base class '"
     << SourceType.getAsString(C.getASTContext().getPrintingPolicy())
     << "' is undefined";

  auto R = std::make_unique<PathSensitiveBugReport>(BT, OS.str(), N);

  // Mark region of problematic base class for later use in the BugVisitor.
  R->markInteresting(BaseClassRegion);
  R->addVisitor<PtrCastVisitor>();
  C.emitReport(std::move(R));
}

PathDiagnosticPieceRef
PolymorphicPtrArithmeticChecker::PtrCastVisitor::VisitNode(
    const ExplodedNode *N, BugReporterContext &BRC,
    PathSensitiveBugReport &BR) {
  const Stmt *S = N->getStmtForDiagnostics();
  if (!S)
    return nullptr;

  const auto *CastE = dyn_cast<CastExpr>(S);
  if (!CastE)
    return nullptr;

  // FIXME: This way of getting base types does not support reference types.
  QualType SourceType = CastE->getSubExpr()->getType()->getPointeeType();
  QualType TargetType = CastE->getType()->getPointeeType();

  if (SourceType.isNull() || TargetType.isNull() || SourceType == TargetType)
    return nullptr;

  // Region associated with the current cast expression.
  const MemRegion *M = N->getSVal(CastE).getAsRegion();
  if (!M)
    return nullptr;

  // Check if target region was marked as problematic previously.
  if (!BR.isInteresting(M))
    return nullptr;

  SmallString<256> Buf;
  llvm::raw_svector_ostream OS(Buf);

  OS << "Casting from '" << SourceType.getAsString() << "' to '"
     << TargetType.getAsString() << "' here";

  PathDiagnosticLocation Pos(S, BRC.getSourceManager(),
                             N->getLocationContext());
  return std::make_shared<PathDiagnosticEventPiece>(Pos, OS.str(),
                                                    /*addPosRange=*/true);
}

void ento::registerPolymorphicPtrArithmeticChecker(CheckerManager &mgr) {
  mgr.registerChecker<PolymorphicPtrArithmeticChecker>();
}

bool ento::shouldRegisterPolymorphicPtrArithmeticChecker(
    const CheckerManager &mgr) {
  return true;
}

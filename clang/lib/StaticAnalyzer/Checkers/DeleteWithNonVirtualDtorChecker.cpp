//===-- DeleteWithNonVirtualDtorChecker.cpp -----------------------*- C++ -*--//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines a checker for the OOP52-CPP CERT rule: Do not delete a polymorphic
// object without a virtual destructor.
//
// Diagnostic flags -Wnon-virtual-dtor and -Wdelete-non-virtual-dtor report if
// an object with a virtual function but a non-virtual destructor exists or is
// deleted, respectively.
//
// This check exceeds them by comparing the dynamic and static types of the
// object at the point of destruction and only warns if it happens through a
// pointer to a base type without a virtual destructor. The check places a note
// at the last point where the conversion from derived to base happened.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/DynamicType.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramStateTrait.h"

using namespace clang;
using namespace ento;

namespace {
class DeleteWithNonVirtualDtorChecker
    : public Checker<check::PreStmt<CXXDeleteExpr>> {
  mutable std::unique_ptr<BugType> BT;

  class DeleteBugVisitor : public BugReporterVisitor {
  public:
    DeleteBugVisitor() : Satisfied(false) {}
    void Profile(llvm::FoldingSetNodeID &ID) const override {
      static int X = 0;
      ID.AddPointer(&X);
    }
    PathDiagnosticPieceRef VisitNode(const ExplodedNode *N,
                                     BugReporterContext &BRC,
                                     PathSensitiveBugReport &BR) override;

  private:
    bool Satisfied;
  };

public:
  void checkPreStmt(const CXXDeleteExpr *DE, CheckerContext &C) const;
};
} // end anonymous namespace

void DeleteWithNonVirtualDtorChecker::checkPreStmt(const CXXDeleteExpr *DE,
                                                   CheckerContext &C) const {
  const Expr *DeletedObj = DE->getArgument();
  const MemRegion *MR = C.getSVal(DeletedObj).getAsRegion();
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

  if (BaseClass->getDestructor()->isVirtual())
    return;

  if (!DerivedClass->isDerivedFrom(BaseClass))
    return;

  if (!BT)
    BT.reset(new BugType(this,
                         "Destruction of a polymorphic object with no "
                         "virtual destructor",
                         "Logic error"));

  ExplodedNode *N = C.generateNonFatalErrorNode();
  if (!N)
    return;
  auto R = std::make_unique<PathSensitiveBugReport>(*BT, BT->getDescription(), N);

  // Mark region of problematic base class for later use in the BugVisitor.
  R->markInteresting(BaseClassRegion);
  R->addVisitor(std::make_unique<DeleteBugVisitor>());
  C.emitReport(std::move(R));
}

PathDiagnosticPieceRef
DeleteWithNonVirtualDtorChecker::DeleteBugVisitor::VisitNode(
    const ExplodedNode *N, BugReporterContext &BRC,
    PathSensitiveBugReport &BR) {
  // Stop traversal after the first conversion was found on a path.
  if (Satisfied)
    return nullptr;

  const Stmt *S = N->getStmtForDiagnostics();
  if (!S)
    return nullptr;

  const auto *CastE = dyn_cast<CastExpr>(S);
  if (!CastE)
    return nullptr;

  // Only interested in DerivedToBase implicit casts.
  // Explicit casts can have different CastKinds.
  if (const auto *ImplCastE = dyn_cast<ImplicitCastExpr>(CastE)) {
    if (ImplCastE->getCastKind() != CK_DerivedToBase)
      return nullptr;
  }

  // Region associated with the current cast expression.
  const MemRegion *M = N->getSVal(CastE).getAsRegion();
  if (!M)
    return nullptr;

  // Check if target region was marked as problematic previously.
  if (!BR.isInteresting(M))
    return nullptr;

  // Stop traversal on this path.
  Satisfied = true;

  SmallString<256> Buf;
  llvm::raw_svector_ostream OS(Buf);
  OS << "Conversion from derived to base happened here";
  PathDiagnosticLocation Pos(S, BRC.getSourceManager(),
                             N->getLocationContext());
  return std::make_shared<PathDiagnosticEventPiece>(Pos, OS.str(), true);
}

void ento::registerDeleteWithNonVirtualDtorChecker(CheckerManager &mgr) {
  mgr.registerChecker<DeleteWithNonVirtualDtorChecker>();
}

bool ento::shouldRegisterDeleteWithNonVirtualDtorChecker(
                                                    const CheckerManager &mgr) {
  return true;
}

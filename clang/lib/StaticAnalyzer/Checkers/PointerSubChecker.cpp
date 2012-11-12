//=== PointerSubChecker.cpp - Pointer subtraction checker ------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This files defines PointerSubChecker, a builtin checker that checks for
// pointer subtractions on two pointers pointing to different memory chunks. 
// This check corresponds to CWE-469.
//
//===----------------------------------------------------------------------===//

#include "ClangSACheckers.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"

using namespace clang;
using namespace ento;

namespace {
class PointerSubChecker 
  : public Checker< check::PreStmt<BinaryOperator> > {
  mutable OwningPtr<BuiltinBug> BT;

public:
  void checkPreStmt(const BinaryOperator *B, CheckerContext &C) const;
};
}

void PointerSubChecker::checkPreStmt(const BinaryOperator *B,
                                     CheckerContext &C) const {
  // When doing pointer subtraction, if the two pointers do not point to the
  // same memory chunk, emit a warning.
  if (B->getOpcode() != BO_Sub)
    return;

  ProgramStateRef state = C.getState();
  const LocationContext *LCtx = C.getLocationContext();
  SVal LV = state->getSVal(B->getLHS(), LCtx);
  SVal RV = state->getSVal(B->getRHS(), LCtx);

  const MemRegion *LR = LV.getAsRegion();
  const MemRegion *RR = RV.getAsRegion();

  if (!(LR && RR))
    return;

  const MemRegion *BaseLR = LR->getBaseRegion();
  const MemRegion *BaseRR = RR->getBaseRegion();

  if (BaseLR == BaseRR)
    return;

  // Allow arithmetic on different symbolic regions.
  if (isa<SymbolicRegion>(BaseLR) || isa<SymbolicRegion>(BaseRR))
    return;

  if (ExplodedNode *N = C.addTransition()) {
    if (!BT)
      BT.reset(new BuiltinBug("Pointer subtraction", 
                          "Subtraction of two pointers that do not point to "
                          "the same memory chunk may cause incorrect result."));
    BugReport *R = new BugReport(*BT, BT->getDescription(), N);
    R->addRange(B->getSourceRange());
    C.emitReport(R);
  }
}

void ento::registerPointerSubChecker(CheckerManager &mgr) {
  mgr.registerChecker<PointerSubChecker>();
}

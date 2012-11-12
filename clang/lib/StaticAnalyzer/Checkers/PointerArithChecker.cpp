//=== PointerArithChecker.cpp - Pointer arithmetic checker -----*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This files defines PointerArithChecker, a builtin checker that checks for
// pointer arithmetic on locations other than array elements.
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
class PointerArithChecker 
  : public Checker< check::PreStmt<BinaryOperator> > {
  mutable OwningPtr<BuiltinBug> BT;

public:
  void checkPreStmt(const BinaryOperator *B, CheckerContext &C) const;
};
}

void PointerArithChecker::checkPreStmt(const BinaryOperator *B,
                                       CheckerContext &C) const {
  if (B->getOpcode() != BO_Sub && B->getOpcode() != BO_Add)
    return;

  ProgramStateRef state = C.getState();
  const LocationContext *LCtx = C.getLocationContext();
  SVal LV = state->getSVal(B->getLHS(), LCtx);
  SVal RV = state->getSVal(B->getRHS(), LCtx);

  const MemRegion *LR = LV.getAsRegion();

  if (!LR || !RV.isConstant())
    return;

  // If pointer arithmetic is done on variables of non-array type, this often
  // means behavior rely on memory organization, which is dangerous.
  if (isa<VarRegion>(LR) || isa<CodeTextRegion>(LR) || 
      isa<CompoundLiteralRegion>(LR)) {

    if (ExplodedNode *N = C.addTransition()) {
      if (!BT)
        BT.reset(new BuiltinBug("Dangerous pointer arithmetic",
                            "Pointer arithmetic done on non-array variables "
                            "means reliance on memory layout, which is "
                            "dangerous."));
      BugReport *R = new BugReport(*BT, BT->getDescription(), N);
      R->addRange(B->getSourceRange());
      C.emitReport(R);
    }
  }
}

void ento::registerPointerArithChecker(CheckerManager &mgr) {
  mgr.registerChecker<PointerArithChecker>();
}

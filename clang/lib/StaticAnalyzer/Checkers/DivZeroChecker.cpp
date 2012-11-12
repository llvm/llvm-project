//== DivZeroChecker.cpp - Division by zero checker --------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines DivZeroChecker, a builtin check in ExprEngine that performs
// checks for division by zeros.
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
class DivZeroChecker : public Checker< check::PreStmt<BinaryOperator> > {
  mutable OwningPtr<BuiltinBug> BT;
  void reportBug(const char *Msg,
                 ProgramStateRef StateZero,
                 CheckerContext &C) const ;
public:
  void checkPreStmt(const BinaryOperator *B, CheckerContext &C) const;
};  
} // end anonymous namespace

void DivZeroChecker::reportBug(const char *Msg,
                               ProgramStateRef StateZero,
                               CheckerContext &C) const {
  if (ExplodedNode *N = C.generateSink(StateZero)) {
    if (!BT)
      BT.reset(new BuiltinBug("Division by zero"));

    BugReport *R = new BugReport(*BT, Msg, N);
    bugreporter::trackNullOrUndefValue(N, bugreporter::GetDenomExpr(N), *R);
    C.emitReport(R);
  }
}

void DivZeroChecker::checkPreStmt(const BinaryOperator *B,
                                  CheckerContext &C) const {
  BinaryOperator::Opcode Op = B->getOpcode();
  if (Op != BO_Div &&
      Op != BO_Rem &&
      Op != BO_DivAssign &&
      Op != BO_RemAssign)
    return;

  if (!B->getRHS()->getType()->isScalarType())
    return;

  SVal Denom = C.getState()->getSVal(B->getRHS(), C.getLocationContext());
  const DefinedSVal *DV = dyn_cast<DefinedSVal>(&Denom);

  // Divide-by-undefined handled in the generic checking for uses of
  // undefined values.
  if (!DV)
    return;

  // Check for divide by zero.
  ConstraintManager &CM = C.getConstraintManager();
  ProgramStateRef stateNotZero, stateZero;
  llvm::tie(stateNotZero, stateZero) = CM.assumeDual(C.getState(), *DV);

  if (!stateNotZero) {
    assert(stateZero);
    reportBug("Division by zero", stateZero, C);
    return;
  }

  bool TaintedD = C.getState()->isTainted(*DV);
  if ((stateNotZero && stateZero && TaintedD)) {
    reportBug("Division by a tainted value, possibly zero", stateZero, C);
    return;
  }

  // If we get here, then the denom should not be zero. We abandon the implicit
  // zero denom case for now.
  C.addTransition(stateNotZero);
}

void ento::registerDivZeroChecker(CheckerManager &mgr) {
  mgr.registerChecker<DivZeroChecker>();
}

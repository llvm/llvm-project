//===--- UndefinedArraySubscriptChecker.h ----------------------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines UndefinedArraySubscriptChecker, a builtin check in ExprEngine
// that performs checks for undefined array subscripts.
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
class UndefinedArraySubscriptChecker
  : public Checker< check::PreStmt<ArraySubscriptExpr> > {
  mutable OwningPtr<BugType> BT;

public:
  void checkPreStmt(const ArraySubscriptExpr *A, CheckerContext &C) const;
};
} // end anonymous namespace

void 
UndefinedArraySubscriptChecker::checkPreStmt(const ArraySubscriptExpr *A,
                                             CheckerContext &C) const {
  if (C.getState()->getSVal(A->getIdx(), C.getLocationContext()).isUndef()) {
    if (ExplodedNode *N = C.generateSink()) {
      if (!BT)
        BT.reset(new BuiltinBug("Array subscript is undefined"));

      // Generate a report for this bug.
      BugReport *R = new BugReport(*BT, BT->getName(), N);
      R->addRange(A->getIdx()->getSourceRange());
      bugreporter::trackNullOrUndefValue(N, A->getIdx(), *R);
      C.emitReport(R);
    }
  }
}

void ento::registerUndefinedArraySubscriptChecker(CheckerManager &mgr) {
  mgr.registerChecker<UndefinedArraySubscriptChecker>();
}

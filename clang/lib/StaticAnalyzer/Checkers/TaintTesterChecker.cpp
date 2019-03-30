//== TaintTesterChecker.cpp ----------------------------------- -*- C++ -*--=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This checker can be used for testing how taint data is propagated.
//
//===----------------------------------------------------------------------===//

#include "Taint.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"

using namespace clang;
using namespace ento;
using namespace taint;

namespace {
class TaintTesterChecker : public Checker< check::PostStmt<Expr> > {

  mutable std::unique_ptr<BugType> BT;
  void initBugType() const;

  /// Given a pointer argument, get the symbol of the value it contains
  /// (points to).
  SymbolRef getPointedToSymbol(CheckerContext &C,
                               const Expr* Arg,
                               bool IssueWarning = true) const;

public:
  void checkPostStmt(const Expr *E, CheckerContext &C) const;
};
}

inline void TaintTesterChecker::initBugType() const {
  if (!BT)
    BT.reset(new BugType(this, "Tainted data", "General"));
}

void TaintTesterChecker::checkPostStmt(const Expr *E,
                                       CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  if (!State)
    return;

  if (isTainted(State, E, C.getLocationContext())) {
    if (ExplodedNode *N = C.generateNonFatalErrorNode()) {
      initBugType();
      auto report = llvm::make_unique<BugReport>(*BT, "tainted",N);
      report->addRange(E->getSourceRange());
      C.emitReport(std::move(report));
    }
  }
}

void ento::registerTaintTesterChecker(CheckerManager &mgr) {
  mgr.registerChecker<TaintTesterChecker>();
}

bool ento::shouldRegisterTaintTesterChecker(const LangOptions &LO) {
  return true;
}

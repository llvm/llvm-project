//== PutenvWithAutoChecker.cpp --------------------------------- -*- C++ -*--=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines PutenvWithAutoChecker which finds calls of ``putenv``
// function with automatic variable as the argument.
// https://wiki.sei.cmu.edu/confluence/x/6NYxBQ
//
//===----------------------------------------------------------------------===//

#include "../AllocationState.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallDescription.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/MemRegion.h"

using namespace clang;
using namespace ento;

namespace {
class PutenvWithAutoChecker : public Checker<check::PostCall> {
private:
  BugType BT{this, "'putenv' function should not be called with auto variables",
             categories::SecurityError};
  const CallDescription Putenv{{"putenv"}, 1};

public:
  void checkPostCall(const CallEvent &Call, CheckerContext &C) const;
};
} // namespace

void PutenvWithAutoChecker::checkPostCall(const CallEvent &Call,
                                          CheckerContext &C) const {
  if (!Putenv.matches(Call))
    return;

  SVal ArgV = Call.getArgSVal(0);
  const Expr *ArgExpr = Call.getArgExpr(0);
  const MemSpaceRegion *MSR = ArgV.getAsRegion()->getMemorySpace();

  if (!isa<StackSpaceRegion>(MSR))
    return;

  StringRef ErrorMsg = "The 'putenv' function should not be called with "
                       "arguments that have automatic storage";
  ExplodedNode *N = C.generateErrorNode();
  auto Report = std::make_unique<PathSensitiveBugReport>(BT, ErrorMsg, N);

  // Track the argument.
  bugreporter::trackExpressionValue(Report->getErrorNode(), ArgExpr, *Report);

  C.emitReport(std::move(Report));
}

void ento::registerPutenvWithAuto(CheckerManager &Mgr) {
  Mgr.registerChecker<PutenvWithAutoChecker>();
}

bool ento::shouldRegisterPutenvWithAuto(const CheckerManager &) { return true; }

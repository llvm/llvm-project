//==- ExprInspectionChecker.cpp - Used for regression tests ------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ClangSACheckers.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "llvm/ADT/StringSwitch.h"

using namespace clang;
using namespace ento;

namespace {
class ExprInspectionChecker : public Checker< eval::Call > {
  mutable std::unique_ptr<BugType> BT;

  void analyzerEval(const CallExpr *CE, CheckerContext &C) const;
  void analyzerCheckInlined(const CallExpr *CE, CheckerContext &C) const;
  void analyzerWarnIfReached(const CallExpr *CE, CheckerContext &C) const;
  void analyzerCrash(const CallExpr *CE, CheckerContext &C) const;

  typedef void (ExprInspectionChecker::*FnCheck)(const CallExpr *,
                                                 CheckerContext &C) const;

public:
  bool evalCall(const CallExpr *CE, CheckerContext &C) const;
};
}

bool ExprInspectionChecker::evalCall(const CallExpr *CE,
                                     CheckerContext &C) const {
  // These checks should have no effect on the surrounding environment
  // (globals should not be invalidated, etc), hence the use of evalCall.
  FnCheck Handler = llvm::StringSwitch<FnCheck>(C.getCalleeName(CE))
    .Case("clang_analyzer_eval", &ExprInspectionChecker::analyzerEval)
    .Case("clang_analyzer_checkInlined",
          &ExprInspectionChecker::analyzerCheckInlined)
    .Case("clang_analyzer_crash", &ExprInspectionChecker::analyzerCrash)
    .Case("clang_analyzer_warnIfReached", &ExprInspectionChecker::analyzerWarnIfReached)
    .Default(nullptr);

  if (!Handler)
    return false;

  (this->*Handler)(CE, C);
  return true;
}

static const char *getArgumentValueString(const CallExpr *CE,
                                          CheckerContext &C) {
  if (CE->getNumArgs() == 0)
    return "Missing assertion argument";

  ExplodedNode *N = C.getPredecessor();
  const LocationContext *LC = N->getLocationContext();
  ProgramStateRef State = N->getState();

  const Expr *Assertion = CE->getArg(0);
  SVal AssertionVal = State->getSVal(Assertion, LC);

  if (AssertionVal.isUndef())
    return "UNDEFINED";

  ProgramStateRef StTrue, StFalse;
  std::tie(StTrue, StFalse) =
    State->assume(AssertionVal.castAs<DefinedOrUnknownSVal>());

  if (StTrue) {
    if (StFalse)
      return "UNKNOWN";
    else
      return "TRUE";
  } else {
    if (StFalse)
      return "FALSE";
    else
      llvm_unreachable("Invalid constraint; neither true or false.");
  }
}

void ExprInspectionChecker::analyzerEval(const CallExpr *CE,
                                         CheckerContext &C) const {
  const LocationContext *LC = C.getPredecessor()->getLocationContext();

  // A specific instantiation of an inlined function may have more constrained
  // values than can generally be assumed. Skip the check.
  if (LC->getCurrentStackFrame()->getParent() != nullptr)
    return;

  if (!BT)
    BT.reset(new BugType(this, "Checking analyzer assumptions", "debug"));

  ExplodedNode *N = C.generateNonFatalErrorNode();
  if (!N)
    return;
  C.emitReport(
      llvm::make_unique<BugReport>(*BT, getArgumentValueString(CE, C), N));
}

void ExprInspectionChecker::analyzerWarnIfReached(const CallExpr *CE,
                                                  CheckerContext &C) const {

  if (!BT)
    BT.reset(new BugType(this, "Checking analyzer assumptions", "debug"));

  ExplodedNode *N = C.generateNonFatalErrorNode();
  if (!N)
    return;
  C.emitReport(llvm::make_unique<BugReport>(*BT, "REACHABLE", N));
}

void ExprInspectionChecker::analyzerCheckInlined(const CallExpr *CE,
                                                 CheckerContext &C) const {
  const LocationContext *LC = C.getPredecessor()->getLocationContext();

  // An inlined function could conceivably also be analyzed as a top-level
  // function. We ignore this case and only emit a message (TRUE or FALSE)
  // when we are analyzing it as an inlined function. This means that
  // clang_analyzer_checkInlined(true) should always print TRUE, but
  // clang_analyzer_checkInlined(false) should never actually print anything.
  if (LC->getCurrentStackFrame()->getParent() == nullptr)
    return;

  if (!BT)
    BT.reset(new BugType(this, "Checking analyzer assumptions", "debug"));

  ExplodedNode *N = C.generateNonFatalErrorNode();
  if (!N)
    return;
  C.emitReport(
      llvm::make_unique<BugReport>(*BT, getArgumentValueString(CE, C), N));
}

void ExprInspectionChecker::analyzerCrash(const CallExpr *CE,
                                          CheckerContext &C) const {
  LLVM_BUILTIN_TRAP;
}

void ento::registerExprInspectionChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<ExprInspectionChecker>();
}


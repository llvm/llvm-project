//=== ErrnoTesterChecker.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This defines ErrnoTesterChecker, which is used to test functionality of the
// errno_check API.
//
//===----------------------------------------------------------------------===//

#include "ErrnoModeling.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallDescription.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"

using namespace clang;
using namespace ento;

namespace {

class ErrnoTesterChecker : public Checker<eval::Call> {
public:
  bool evalCall(const CallEvent &Call, CheckerContext &C) const;

private:
  static void evalSetErrno(CheckerContext &C, const CallEvent &Call);
  static void evalGetErrno(CheckerContext &C, const CallEvent &Call);
  static void evalSetErrnoIfError(CheckerContext &C, const CallEvent &Call);
  static void evalSetErrnoIfErrorRange(CheckerContext &C,
                                       const CallEvent &Call);

  using EvalFn = std::function<void(CheckerContext &, const CallEvent &)>;
  const CallDescriptionMap<EvalFn> TestCalls{
      {{"ErrnoTesterChecker_setErrno", 1}, &ErrnoTesterChecker::evalSetErrno},
      {{"ErrnoTesterChecker_getErrno", 0}, &ErrnoTesterChecker::evalGetErrno},
      {{"ErrnoTesterChecker_setErrnoIfError", 0},
       &ErrnoTesterChecker::evalSetErrnoIfError},
      {{"ErrnoTesterChecker_setErrnoIfErrorRange", 0},
       &ErrnoTesterChecker::evalSetErrnoIfErrorRange}};
};

} // namespace

void ErrnoTesterChecker::evalSetErrno(CheckerContext &C,
                                      const CallEvent &Call) {
  C.addTransition(errno_modeling::setErrnoValue(
      C.getState(), C.getLocationContext(), Call.getArgSVal(0)));
}

void ErrnoTesterChecker::evalGetErrno(CheckerContext &C,
                                      const CallEvent &Call) {
  ProgramStateRef State = C.getState();

  Optional<SVal> ErrnoVal = errno_modeling::getErrnoValue(State);
  assert(ErrnoVal && "Errno value should be available.");
  State =
      State->BindExpr(Call.getOriginExpr(), C.getLocationContext(), *ErrnoVal);

  C.addTransition(State);
}

void ErrnoTesterChecker::evalSetErrnoIfError(CheckerContext &C,
                                             const CallEvent &Call) {
  ProgramStateRef State = C.getState();
  SValBuilder &SVB = C.getSValBuilder();

  ProgramStateRef StateSuccess = State->BindExpr(
      Call.getOriginExpr(), C.getLocationContext(), SVB.makeIntVal(0, true));

  ProgramStateRef StateFailure = State->BindExpr(
      Call.getOriginExpr(), C.getLocationContext(), SVB.makeIntVal(1, true));
  StateFailure = errno_modeling::setErrnoValue(StateFailure, C, 11);

  C.addTransition(StateSuccess);
  C.addTransition(StateFailure);
}

void ErrnoTesterChecker::evalSetErrnoIfErrorRange(CheckerContext &C,
                                                  const CallEvent &Call) {
  ProgramStateRef State = C.getState();
  SValBuilder &SVB = C.getSValBuilder();

  ProgramStateRef StateSuccess = State->BindExpr(
      Call.getOriginExpr(), C.getLocationContext(), SVB.makeIntVal(0, true));

  ProgramStateRef StateFailure = State->BindExpr(
      Call.getOriginExpr(), C.getLocationContext(), SVB.makeIntVal(1, true));
  DefinedOrUnknownSVal ErrnoVal = SVB.conjureSymbolVal(
      nullptr, Call.getOriginExpr(), C.getLocationContext(), C.blockCount());
  StateFailure = StateFailure->assume(ErrnoVal, true);
  assert(StateFailure && "Failed to assume on an initial value.");
  StateFailure = errno_modeling::setErrnoValue(
      StateFailure, C.getLocationContext(), ErrnoVal);

  C.addTransition(StateSuccess);
  C.addTransition(StateFailure);
}

bool ErrnoTesterChecker::evalCall(const CallEvent &Call,
                                  CheckerContext &C) const {
  const EvalFn *Fn = TestCalls.lookup(Call);
  if (Fn) {
    (*Fn)(C, Call);
    return C.isDifferent();
  }
  return false;
}

void ento::registerErrnoTesterChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<ErrnoTesterChecker>();
}

bool ento::shouldRegisterErrnoTesterChecker(const CheckerManager &Mgr) {
  return true;
}

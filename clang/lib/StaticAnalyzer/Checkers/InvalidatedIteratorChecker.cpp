//===-- InvalidatedIteratorChecker.cpp ----------------------------*- C++ -*--//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines a checker for access of invalidated iterators.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"


#include "Iterator.h"

using namespace clang;
using namespace ento;
using namespace iterator;

namespace {

class InvalidatedIteratorChecker
  : public Checker<check::PreCall> {

  std::unique_ptr<BugType> InvalidatedBugType;

  void verifyAccess(CheckerContext &C, const SVal &Val) const;
  void reportBug(const StringRef &Message, const SVal &Val,
                 CheckerContext &C, ExplodedNode *ErrNode) const;
public:
  InvalidatedIteratorChecker();

  void checkPreCall(const CallEvent &Call, CheckerContext &C) const;

};

} //namespace

InvalidatedIteratorChecker::InvalidatedIteratorChecker() {
  InvalidatedBugType.reset(
      new BugType(this, "Iterator invalidated", "Misuse of STL APIs"));
}

void InvalidatedIteratorChecker::checkPreCall(const CallEvent &Call,
                                              CheckerContext &C) const {
  // Check for access of invalidated position
  const auto *Func = dyn_cast_or_null<FunctionDecl>(Call.getDecl());
  if (!Func)
    return;

  if (Func->isOverloadedOperator() &&
      isAccessOperator(Func->getOverloadedOperator())) {
    // Check for any kind of access of invalidated iterator positions
    if (const auto *InstCall = dyn_cast<CXXInstanceCall>(&Call)) {
      verifyAccess(C, InstCall->getCXXThisVal());
    } else {
      verifyAccess(C, Call.getArgSVal(0));
    }
  }
}

void InvalidatedIteratorChecker::verifyAccess(CheckerContext &C, const SVal &Val) const {
  auto State = C.getState();
  const auto *Pos = getIteratorPosition(State, Val);
  if (Pos && !Pos->isValid()) {
    auto *N = C.generateErrorNode(State);
    if (!N) {
      return;
    }
    reportBug("Invalidated iterator accessed.", Val, C, N);
  }
}

void InvalidatedIteratorChecker::reportBug(const StringRef &Message,
                                           const SVal &Val, CheckerContext &C,
                                           ExplodedNode *ErrNode) const {
  auto R = std::make_unique<PathSensitiveBugReport>(*InvalidatedBugType,
                                                    Message, ErrNode);
  R->markInteresting(Val);
  C.emitReport(std::move(R));
}

void ento::registerInvalidatedIteratorChecker(CheckerManager &mgr) {
  mgr.registerChecker<InvalidatedIteratorChecker>();
}

bool ento::shouldRegisterInvalidatedIteratorChecker(const LangOptions &LO) {
  return true;
}

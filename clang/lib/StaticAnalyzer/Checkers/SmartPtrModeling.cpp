// SmartPtrModeling.cpp - Model behavior of C++ smart pointers - C++ ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a checker that models various aspects of
// C++ smart pointer behavior.
//
//===----------------------------------------------------------------------===//

#include "Move.h"

#include "clang/AST/ExprCXX.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"

using namespace clang;
using namespace ento;

namespace {
class SmartPtrModeling : public Checker<eval::Call> {
  bool isNullAfterMoveMethod(const CallEvent &Call) const;

public:
  bool evalCall(const CallEvent &Call, CheckerContext &C) const;
};
} // end of anonymous namespace

bool SmartPtrModeling::isNullAfterMoveMethod(const CallEvent &Call) const {
  // TODO: Update CallDescription to support anonymous calls?
  // TODO: Handle other methods, such as .get() or .release().
  // But once we do, we'd need a visitor to explain null dereferences
  // that are found via such modeling.
  const auto *CD = dyn_cast_or_null<CXXConversionDecl>(Call.getDecl());
  return CD && CD->getConversionType()->isBooleanType();
}

bool SmartPtrModeling::evalCall(const CallEvent &Call,
                                CheckerContext &C) const {
  if (!isNullAfterMoveMethod(Call))
    return false;

  ProgramStateRef State = C.getState();
  const MemRegion *ThisR =
      cast<CXXInstanceCall>(&Call)->getCXXThisVal().getAsRegion();

  if (!move::isMovedFrom(State, ThisR)) {
    // TODO: Model this case as well. At least, avoid invalidation of globals.
    return false;
  }

  // TODO: Add a note to bug reports describing this decision.
  C.addTransition(
      State->BindExpr(Call.getOriginExpr(), C.getLocationContext(),
                      C.getSValBuilder().makeZeroVal(Call.getResultType())));
  return true;
}

void ento::registerSmartPtrModeling(CheckerManager &Mgr) {
  Mgr.registerChecker<SmartPtrModeling>();
}

bool ento::shouldRegisterSmartPtrModeling(const CheckerManager &mgr) {
  const LangOptions &LO = mgr.getLangOpts();
  return LO.CPlusPlus;
}

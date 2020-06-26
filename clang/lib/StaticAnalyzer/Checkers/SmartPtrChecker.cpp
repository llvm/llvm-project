// SmartPtrChecker.cpp - Check for smart pointer dereference - C++ --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a checker that check for null dereference of C++ smart
// pointer.
//
//===----------------------------------------------------------------------===//
#include "SmartPtr.h"

#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Type.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SVals.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SymExpr.h"

using namespace clang;
using namespace ento;

namespace {
class SmartPtrChecker : public Checker<check::PreCall> {
  BugType NullDereferenceBugType{this, "Null SmartPtr dereference",
                                 "C++ Smart Pointer"};

public:
  void checkPreCall(const CallEvent &Call, CheckerContext &C) const;

private:
  void reportBug(CheckerContext &C, const CallEvent &Call) const;
};
} // end of anonymous namespace

void SmartPtrChecker::checkPreCall(const CallEvent &Call,
                                   CheckerContext &C) const {
  if (!smartptr::isStdSmartPtrCall(Call))
    return;
  ProgramStateRef State = C.getState();
  const auto *OC = dyn_cast<CXXMemberOperatorCall>(&Call);
  if (!OC)
    return;
  const MemRegion *ThisRegion = OC->getCXXThisVal().getAsRegion();
  if (!ThisRegion)
    return;

  OverloadedOperatorKind OOK = OC->getOverloadedOperator();
  if (OOK == OO_Star || OOK == OO_Arrow) {
    if (smartptr::isNullSmartPtr(State, ThisRegion))
      reportBug(C, Call);
  }
}

void SmartPtrChecker::reportBug(CheckerContext &C,
                                const CallEvent &Call) const {
  ExplodedNode *ErrNode = C.generateErrorNode();
  if (!ErrNode)
    return;

  auto R = std::make_unique<PathSensitiveBugReport>(
      NullDereferenceBugType, "Dereference of null smart pointer", ErrNode);
  C.emitReport(std::move(R));
}

void ento::registerSmartPtrChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<SmartPtrChecker>();
}

bool ento::shouldRegisterSmartPtrChecker(const CheckerManager &mgr) {
  const LangOptions &LO = mgr.getLangOpts();
  return LO.CPlusPlus;
}

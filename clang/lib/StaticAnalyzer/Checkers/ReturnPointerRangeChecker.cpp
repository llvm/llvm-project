//== ReturnPointerRangeChecker.cpp ------------------------------*- C++ -*--==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines ReturnPointerRangeChecker, which is a path-sensitive check
// which looks for an out-of-bound pointer being returned to callers.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/DynamicExtent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h"

using namespace clang;
using namespace ento;

namespace {
class ReturnPointerRangeChecker :
    public Checker< check::PreStmt<ReturnStmt> > {
  mutable std::unique_ptr<BuiltinBug> BT;

public:
    void checkPreStmt(const ReturnStmt *RS, CheckerContext &C) const;
};
}

void ReturnPointerRangeChecker::checkPreStmt(const ReturnStmt *RS,
                                             CheckerContext &C) const {
  ProgramStateRef state = C.getState();

  const Expr *RetE = RS->getRetValue();
  if (!RetE)
    return;

  SVal V = C.getSVal(RetE);
  const MemRegion *R = V.getAsRegion();

  const ElementRegion *ER = dyn_cast_or_null<ElementRegion>(R);
  if (!ER)
    return;

  DefinedOrUnknownSVal Idx = ER->getIndex().castAs<DefinedOrUnknownSVal>();
  // Zero index is always in bound, this also passes ElementRegions created for
  // pointer casts.
  if (Idx.isZeroConstant())
    return;

  // FIXME: All of this out-of-bounds checking should eventually be refactored
  // into a common place.
  DefinedOrUnknownSVal ElementCount = getDynamicElementCount(
      state, ER->getSuperRegion(), C.getSValBuilder(), ER->getValueType());

  // We assume that the location after the last element in the array is used as
  // end() iterator. Reporting on these would return too many false positives.
  if (Idx == ElementCount)
    return;

  ProgramStateRef StInBound = state->assumeInBound(Idx, ElementCount, true);
  ProgramStateRef StOutBound = state->assumeInBound(Idx, ElementCount, false);
  if (StOutBound && !StInBound) {
    ExplodedNode *N = C.generateErrorNode(StOutBound);

    if (!N)
      return;

    // FIXME: This bug correspond to CWE-466.  Eventually we should have bug
    // types explicitly reference such exploit categories (when applicable).
    if (!BT)
      BT.reset(new BuiltinBug(
          this, "Buffer overflow",
          "Returned pointer value points outside the original object "
          "(potential buffer overflow)"));

    // FIXME: It would be nice to eventually make this diagnostic more clear,
    // e.g., by referencing the original declaration or by saying *why* this
    // reference is outside the range.

    // Generate a report for this bug.
    auto report =
        std::make_unique<PathSensitiveBugReport>(*BT, BT->getDescription(), N);

    report->addRange(RetE->getSourceRange());
    C.emitReport(std::move(report));
  }
}

void ento::registerReturnPointerRangeChecker(CheckerManager &mgr) {
  mgr.registerChecker<ReturnPointerRangeChecker>();
}

bool ento::shouldRegisterReturnPointerRangeChecker(const CheckerManager &mgr) {
  return true;
}

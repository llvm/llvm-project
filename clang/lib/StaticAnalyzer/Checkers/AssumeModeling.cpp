//=== AssumeModeling.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This checker evaluates the builting assume functions.
// This checker also sinks execution paths leaving [[assume]] attributes with
// false assumptions.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/AttrIterator.h"
#include "clang/Basic/Builtins.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "llvm/ADT/STLExtras.h"

using namespace clang;
using namespace ento;

namespace {
class AssumeModelingChecker
    : public Checker<eval::Call, check::PostStmt<AttributedStmt>> {
public:
  void checkPostStmt(const AttributedStmt *A, CheckerContext &C) const;
  bool evalCall(const CallEvent &Call, CheckerContext &C) const;
};
} // namespace

void AssumeModelingChecker::checkPostStmt(const AttributedStmt *A,
                                          CheckerContext &C) const {
  if (!hasSpecificAttr<CXXAssumeAttr>(A->getAttrs()))
    return;

  for (const auto *Attr : getSpecificAttrs<CXXAssumeAttr>(A->getAttrs())) {
    SVal AssumptionVal = C.getSVal(Attr->getAssumption());

    // The assumption is not evaluated at all if it had sideffects; skip them.
    if (AssumptionVal.isUnknown())
      continue;

    const auto *Assumption = AssumptionVal.getAsInteger();
    if (Assumption && Assumption->isZero()) {
      C.addSink();
    }
  }
}

bool AssumeModelingChecker::evalCall(const CallEvent &Call,
                                     CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  const auto *FD = dyn_cast_or_null<FunctionDecl>(Call.getDecl());
  if (!FD)
    return false;

  if (!llvm::is_contained({Builtin::BI__builtin_assume, Builtin::BI__assume},
                          FD->getBuiltinID())) {
    return false;
  }

  assert(Call.getNumArgs() > 0);
  SVal Arg = Call.getArgSVal(0);
  if (Arg.isUndef())
    return true; // Return true to model purity.

  State = State->assume(Arg.castAs<DefinedOrUnknownSVal>(), true);
  if (!State) {
    C.addSink();
    return true;
  }

  C.addTransition(State);
  return true;
}

void ento::registerAssumeModeling(CheckerManager &Mgr) {
  Mgr.registerChecker<AssumeModelingChecker>();
}

bool ento::shouldRegisterAssumeModeling(const CheckerManager &) { return true; }

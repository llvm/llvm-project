//== TrustNonnullChecker.cpp - Checker for trusting annotations -*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This checker adds an assumption that methods annotated with _Nonnull
// which come from system headers actually return a non-null pointer.
//
//===----------------------------------------------------------------------===//

#include "ClangSACheckers.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerHelpers.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"

using namespace clang;
using namespace ento;

namespace {

class TrustNonnullChecker : public Checker<check::PostCall> {
public:
  void checkPostCall(const CallEvent &Call, CheckerContext &C) const {
    // Only trust annotations for system headers for non-protocols.
    if (!Call.isInSystemHeader())
      return;

    QualType RetType = Call.getResultType();
    if (!RetType->isAnyPointerType())
      return;

    ProgramStateRef State = C.getState();
    if (getNullabilityAnnotation(RetType) == Nullability::Nonnull)
      if (auto L = Call.getReturnValue().getAs<Loc>())
        State = State->assume(*L, /*Assumption=*/true);

    C.addTransition(State);
  }
};

} // end empty namespace


void ento::registerTrustNonnullChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<TrustNonnullChecker>();
}

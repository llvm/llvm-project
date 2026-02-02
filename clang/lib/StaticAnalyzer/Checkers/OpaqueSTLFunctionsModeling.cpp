//===--- OpaqueSTLFunctionsModeling.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Forces conservative evaluation for STL internal implementation functions
// (prefixed with '__') known to cause false positives. This prevents inlining
// of complex STL internals and avoids wasting analysis time spent in
// `BugReporterVisitor`s.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/ProgramPoint.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallDescription.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"

using namespace clang;
using namespace ento;

namespace {
class OpaqueSTLFunctionsModeling : public Checker<eval::Call> {
public:
  bool evalCall(const CallEvent &Call, CheckerContext &C) const;

private:
  using CDM = CallDescription::Mode;
  const CallDescriptionSet ModeledFunctions{
      {CDM::SimpleFunc, {"std", "sort"}},
      {CDM::SimpleFunc, {"std", "stable_sort"}}};
};
} // anonymous namespace

bool OpaqueSTLFunctionsModeling::evalCall(const CallEvent &Call,
                                          CheckerContext &C) const {
  if (!ModeledFunctions.contains(Call))
    return false;

  ProgramStateRef State = C.getState();
  State = Call.invalidateRegions(C.blockCount(), State);
  static const SimpleProgramPointTag OpaqueCallTag{getDebugTag(),
                                                   "Forced Opaque Call"};
  C.addTransition(State, &OpaqueCallTag);
  return true;
}

void ento::registerOpaqueSTLFunctionsModeling(CheckerManager &Mgr) {
  Mgr.registerChecker<OpaqueSTLFunctionsModeling>();
}

bool ento::shouldRegisterOpaqueSTLFunctionsModeling(const CheckerManager &Mgr) {
  return Mgr.getLangOpts().CPlusPlus;
}

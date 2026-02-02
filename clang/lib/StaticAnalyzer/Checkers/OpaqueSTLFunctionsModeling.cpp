//===--- OpaqueSTLFunctionsModeling.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
      {CDM::SimpleFunc, {"std", "stable_sort"}},
      {CDM::SimpleFunc, {"std", "inplace_merge"}}};
};
} // anonymous namespace

bool OpaqueSTLFunctionsModeling::evalCall(const CallEvent &Call,
                                          CheckerContext &C) const {
  if (!ModeledFunctions.contains(Call))
    return false;

  ProgramStateRef InvalidatedRegionsState =
      Call.invalidateRegions(C.blockCount(), C.getState());
  C.addTransition(InvalidatedRegionsState);
  return true;
}

void ento::registerOpaqueSTLFunctionsModeling(CheckerManager &Mgr) {
  Mgr.registerChecker<OpaqueSTLFunctionsModeling>();
}

bool ento::shouldRegisterOpaqueSTLFunctionsModeling(const CheckerManager &Mgr) {
  return Mgr.getLangOpts().CPlusPlus;
}

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
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"

using namespace clang;
using namespace ento;

namespace {
class OpaqueSTLFunctionsModeling : public Checker<eval::Call> {
public:
  bool evalCall(const CallEvent &Call, CheckerContext &C) const;

private:
  bool shouldForceConservativeEval(const CallEvent &Call) const;
};
} // anonymous namespace

bool OpaqueSTLFunctionsModeling::evalCall(const CallEvent &Call,
                                          CheckerContext &C) const {
  if (!shouldForceConservativeEval(Call))
    return false;

  ProgramStateRef State = C.getState();
  State = Call.invalidateRegions(C.blockCount(), State);
  static const SimpleProgramPointTag OpaqueCallTag{getDebugTag(),
                                                   "Forced Opaque Call"};
  C.addTransition(State, &OpaqueCallTag);
  return true;
}

bool OpaqueSTLFunctionsModeling::shouldForceConservativeEval(
    const CallEvent &Call) const {
  const Decl *D = Call.getDecl();
  if (!D || !AnalysisDeclContext::isInStdNamespace(D))
    return false;

  // __uninitialized_construct_buf_dispatch::__ucr is used by stable_sort
  // and inplace_merge.
  if (const auto *MD = dyn_cast<CXXMethodDecl>(D)) {
    if (const IdentifierInfo *II = MD->getIdentifier()) {
      if (II->getName() == "__ucr") {
        const CXXRecordDecl *RD = MD->getParent();
        if (RD->getName().starts_with("__uninitialized_construct_buf_dispatch"))
          return true;
      }
    }
  }

  return false;
}

void ento::registerOpaqueSTLFunctionsModeling(CheckerManager &Mgr) {
  Mgr.registerChecker<OpaqueSTLFunctionsModeling>();
}

bool ento::shouldRegisterOpaqueSTLFunctionsModeling(const CheckerManager &Mgr) {
  return Mgr.getLangOpts().CPlusPlus;
}

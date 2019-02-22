//== MIGChecker.cpp - MIG calling convention checker ------------*- C++ -*--==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines MIGChecker, a Mach Interface Generator calling convention
// checker. Namely, in MIG callback implementation the following rules apply:
// - When a server routine returns KERN_SUCCESS, it must take ownership of
//   resources (and eventually release them).
// - Additionally, when returning KERN_SUCCESS, all out-parameters must be
//   initialized.
// - When it returns anything except KERN_SUCCESS it must not take ownership,
//   because the message and its descriptors will be destroyed by the server
//   function.
// For now we only check the last rule, as its violations lead to dangerous
// use-after-free exploits.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/AnyCall.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"

using namespace clang;
using namespace ento;

namespace {
class MIGChecker : public Checker<check::PostCall, check::PreStmt<ReturnStmt>> {
  BugType BT{this, "Use-after-free (MIG calling convention violation)",
             categories::MemoryError};

  CallDescription vm_deallocate { "vm_deallocate", 3 };

public:
  void checkPostCall(const CallEvent &Call, CheckerContext &C) const;
  void checkPreStmt(const ReturnStmt *RS, CheckerContext &C) const;
};
} // end anonymous namespace

REGISTER_TRAIT_WITH_PROGRAMSTATE(ReleasedParameter, bool)

static bool isCurrentArgSVal(SVal V, CheckerContext &C) {
  SymbolRef Sym = V.getAsSymbol();
  if (!Sym)
    return false;

  const auto *VR = dyn_cast_or_null<VarRegion>(Sym->getOriginRegion());
  return VR && VR->hasStackParametersStorage() &&
         VR->getStackFrame()->inTopFrame();
}

static bool isInMIGCall(CheckerContext &C) {
  const LocationContext *LC = C.getLocationContext();
  const StackFrameContext *SFC;
  // Find the top frame.
  while (LC) {
    SFC = LC->getStackFrame();
    LC = SFC->getParent();
  }

  const Decl *D = SFC->getDecl();

  if (Optional<AnyCall> AC = AnyCall::forDecl(D)) {
    // Even though there's a Sema warning when the return type of an annotated
    // function is not a kern_return_t, this warning isn't an error, so we need
    // an extra sanity check here.
    // FIXME: AnyCall doesn't support blocks yet, so they remain unchecked
    // for now.
    if (!AC->getReturnType(C.getASTContext())
             .getCanonicalType()->isSignedIntegerType())
      return false;
  }

  if (D->hasAttr<MIGServerRoutineAttr>())
    return true;

  return false;
}

void MIGChecker::checkPostCall(const CallEvent &Call, CheckerContext &C) const {
  if (!isInMIGCall(C))
    return;

  if (!Call.isGlobalCFunction())
    return;

  if (!Call.isCalled(vm_deallocate))
    return;

  // TODO: Unhardcode "1".
  SVal Arg = Call.getArgSVal(1);
  if (isCurrentArgSVal(Arg, C))
    C.addTransition(C.getState()->set<ReleasedParameter>(true));
}

void MIGChecker::checkPreStmt(const ReturnStmt *RS, CheckerContext &C) const {
  // It is very unlikely that a MIG callback will be called from anywhere
  // within the project under analysis and the caller isn't itself a routine
  // that follows the MIG calling convention. Therefore we're safe to believe
  // that it's always the top frame that is of interest. There's a slight chance
  // that the user would want to enforce the MIG calling convention upon
  // a random routine in the middle of nowhere, but given that the convention is
  // fairly weird and hard to follow in the first place, there's relatively
  // little motivation to spread it this way.
  if (!C.inTopFrame())
    return;

  if (!isInMIGCall(C))
    return;

  // We know that the function is non-void, but what if the return statement
  // is not there in the code? It's not a compile error, we should not crash.
  if (!RS)
    return;

  ProgramStateRef State = C.getState();
  if (!State->get<ReleasedParameter>())
    return;

  SVal V = C.getSVal(RS);
  if (!State->isNonNull(V).isConstrainedTrue())
    return;

  ExplodedNode *N = C.generateErrorNode();
  if (!N)
    return;

  auto R = llvm::make_unique<BugReport>(
      BT,
      "MIG callback fails with error after deallocating argument value. "
      "This is a use-after-free vulnerability because the caller will try to "
      "deallocate it again",
      N);

  C.emitReport(std::move(R));
}

void ento::registerMIGChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<MIGChecker>();
}

bool ento::shouldRegisterMIGChecker(const LangOptions &LO) {
  return true;
}

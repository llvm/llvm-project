//=== DanglingInternalBufferChecker.cpp ---------------------------*- C++ -*--//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a check that marks a raw pointer to a C++ standard library
// container's inner buffer released when the object is destroyed. This
// information can be used by MallocChecker to detect use-after-free problems.
//
//===----------------------------------------------------------------------===//

#include "ClangSACheckers.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/BugReporter/CommonBugCategories.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "AllocationState.h"

using namespace clang;
using namespace ento;

namespace {

class DanglingInternalBufferChecker : public Checker<check::DeadSymbols,
                                                     check::PostCall> {
  CallDescription CStrFn;

public:
  DanglingInternalBufferChecker() : CStrFn("c_str") {}

  /// Record the connection between the symbol returned by c_str() and the
  /// corresponding string object region in the ProgramState. Mark the symbol
  /// released if the string object is destroyed.
  void checkPostCall(const CallEvent &Call, CheckerContext &C) const;

  /// Clean up the ProgramState map.
  void checkDeadSymbols(SymbolReaper &SymReaper, CheckerContext &C) const;
};

} // end anonymous namespace

// FIXME: c_str() may be called on a string object many times, so it should
// have a list of symbols associated with it.
REGISTER_MAP_WITH_PROGRAMSTATE(RawPtrMap, const MemRegion *, SymbolRef)

void DanglingInternalBufferChecker::checkPostCall(const CallEvent &Call,
                                                  CheckerContext &C) const {
  const auto *ICall = dyn_cast<CXXInstanceCall>(&Call);
  if (!ICall)
    return;

  SVal Obj = ICall->getCXXThisVal();
  const auto *TypedR = dyn_cast_or_null<TypedValueRegion>(Obj.getAsRegion());
  if (!TypedR)
    return;

  auto *TypeDecl = TypedR->getValueType()->getAsCXXRecordDecl();
  if (TypeDecl->getName() != "basic_string")
    return;

  ProgramStateRef State = C.getState();

  if (Call.isCalled(CStrFn)) {
    SVal RawPtr = Call.getReturnValue();
    if (!RawPtr.isUnknown()) {
      State = State->set<RawPtrMap>(TypedR, RawPtr.getAsSymbol());
      C.addTransition(State);
    }
    return;
  }

  if (isa<CXXDestructorCall>(ICall)) {
    if (State->contains<RawPtrMap>(TypedR)) {
      const SymbolRef *StrBufferPtr = State->get<RawPtrMap>(TypedR);
      // FIXME: What if Origin is null?
      const Expr *Origin = Call.getOriginExpr();
      State = allocation_state::markReleased(State, *StrBufferPtr, Origin);
      State = State->remove<RawPtrMap>(TypedR);
      C.addTransition(State);
      return;
    }
  }
}

void DanglingInternalBufferChecker::checkDeadSymbols(SymbolReaper &SymReaper,
                                                     CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  RawPtrMapTy RPM = State->get<RawPtrMap>();
  for (const auto Entry : RPM) {
    if (!SymReaper.isLive(Entry.second))
      State = State->remove<RawPtrMap>(Entry.first);
    if (!SymReaper.isLiveRegion(Entry.first)) {
      // Due to incomplete destructor support, some dead regions might still
      // remain in the program state map. Clean them up.
      State = State->remove<RawPtrMap>(Entry.first);
    }
  }
  C.addTransition(State);
}

void ento::registerDanglingInternalBufferChecker(CheckerManager &Mgr) {
  registerNewDeleteChecker(Mgr);
  Mgr.registerChecker<DanglingInternalBufferChecker>();
}

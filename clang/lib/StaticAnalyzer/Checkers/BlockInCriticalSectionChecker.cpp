//===-- BlockInCriticalSectionChecker.cpp -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Defines a checker for blocks in critical sections. This checker should find
// the calls to blocking functions (for example: sleep, getc, fgets, read,
// recv etc.) inside a critical section. When sleep(x) is called while a mutex
// is held, other threades cannot lock the same mutex. This might take some
// time, leading to bad performance or even deadlock.
//
//===----------------------------------------------------------------------===//

#include "ClangSACheckers.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"

using namespace clang;
using namespace ento;

namespace {

class BlockInCriticalSectionChecker : public Checker<check::PostCall,
                                                     check::PreCall> {

  CallDescription LockFn, UnlockFn, SleepFn, GetcFn, FgetsFn, ReadFn, RecvFn;

  std::unique_ptr<BugType> BlockInCritSectionBugType;

  void reportBlockInCritSection(SymbolRef FileDescSym,
                                const CallEvent &call,
                                CheckerContext &C) const;

public:
  BlockInCriticalSectionChecker();

  void checkPreCall(const CallEvent &Call, CheckerContext &C) const;

  /// Process unlock.
  /// Process lock.
  /// Process blocking functions (sleep, getc, fgets, read, recv)
  void checkPostCall(const CallEvent &Call, CheckerContext &C) const;

};

} // end anonymous namespace

REGISTER_TRAIT_WITH_PROGRAMSTATE(MutexCounter, unsigned)

BlockInCriticalSectionChecker::BlockInCriticalSectionChecker()
    : LockFn("lock"), UnlockFn("unlock"), SleepFn("sleep"), GetcFn("getc"),
      FgetsFn("fgets"), ReadFn("read"), RecvFn("recv") {
  // Initialize the bug type.
  BlockInCritSectionBugType.reset(
      new BugType(this, "Call to blocking function in critical section",
                        "Blocking Error"));
}

void BlockInCriticalSectionChecker::checkPreCall(const CallEvent &Call,
                                                 CheckerContext &C) const {
}

void BlockInCriticalSectionChecker::checkPostCall(const CallEvent &Call,
                                                  CheckerContext &C) const {
  if (!Call.isCalled(LockFn)
      && !Call.isCalled(SleepFn)
      && !Call.isCalled(GetcFn)
      && !Call.isCalled(FgetsFn)
      && !Call.isCalled(ReadFn)
      && !Call.isCalled(RecvFn)
      && !Call.isCalled(UnlockFn))
    return;

  ProgramStateRef State = C.getState();
  unsigned mutexCount = State->get<MutexCounter>();
  if (Call.isCalled(UnlockFn) && mutexCount > 0) {
    State = State->set<MutexCounter>(--mutexCount);
    C.addTransition(State);
  } else if (Call.isCalled(LockFn)) {
    State = State->set<MutexCounter>(++mutexCount);
    C.addTransition(State);
  } else if (mutexCount > 0) {
    SymbolRef BlockDesc = Call.getReturnValue().getAsSymbol();
    reportBlockInCritSection(BlockDesc, Call, C);
  }
}

void BlockInCriticalSectionChecker::reportBlockInCritSection(
    SymbolRef BlockDescSym, const CallEvent &Call, CheckerContext &C) const {
  ExplodedNode *ErrNode = C.generateNonFatalErrorNode();
  if (!ErrNode)
    return;

  auto R = llvm::make_unique<BugReport>(*BlockInCritSectionBugType,
      "A blocking function %s is called inside a critical section.", ErrNode);
  R->addRange(Call.getSourceRange());
  R->markInteresting(BlockDescSym);
  C.emitReport(std::move(R));
}

void ento::registerBlockInCriticalSectionChecker(CheckerManager &mgr) {
  mgr.registerChecker<BlockInCriticalSectionChecker>();
}

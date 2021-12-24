//===-- BlockInCriticalSectionChecker.cpp -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallDescription.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"

using namespace clang;
using namespace ento;

namespace {

class BlockInCriticalSectionChecker : public Checker<check::PostCall> {

  mutable IdentifierInfo *IILockGuard, *IIUniqueLock;

  CallDescription LockFn, UnlockFn, SleepFn, GetcFn, FgetsFn, ReadFn, RecvFn,
                  PthreadLockFn, PthreadTryLockFn, PthreadUnlockFn,
                  MtxLock, MtxTimedLock, MtxTryLock, MtxUnlock;

  StringRef ClassLockGuard, ClassUniqueLock;

  mutable bool IdentifierInfoInitialized;

  std::unique_ptr<BugType> BlockInCritSectionBugType;

  void initIdentifierInfo(ASTContext &Ctx) const;

  void reportBlockInCritSection(SymbolRef FileDescSym,
                                const CallEvent &call,
                                CheckerContext &C) const;

public:
  BlockInCriticalSectionChecker();

  bool isBlockingFunction(const CallEvent &Call) const;
  bool isLockFunction(const CallEvent &Call) const;
  bool isUnlockFunction(const CallEvent &Call) const;

  /// Process unlock.
  /// Process lock.
  /// Process blocking functions (sleep, getc, fgets, read, recv)
  void checkPostCall(const CallEvent &Call, CheckerContext &C) const;
};

} // end anonymous namespace

REGISTER_TRAIT_WITH_PROGRAMSTATE(MutexCounter, unsigned)

BlockInCriticalSectionChecker::BlockInCriticalSectionChecker()
    : IILockGuard(nullptr), IIUniqueLock(nullptr),
      LockFn("lock"), UnlockFn("unlock"), SleepFn("sleep"), GetcFn("getc"),
      FgetsFn("fgets"), ReadFn("read"), RecvFn("recv"),
      PthreadLockFn("pthread_mutex_lock"),
      PthreadTryLockFn("pthread_mutex_trylock"),
      PthreadUnlockFn("pthread_mutex_unlock"),
      MtxLock("mtx_lock"),
      MtxTimedLock("mtx_timedlock"),
      MtxTryLock("mtx_trylock"),
      MtxUnlock("mtx_unlock"),
      ClassLockGuard("lock_guard"),
      ClassUniqueLock("unique_lock"),
      IdentifierInfoInitialized(false) {
  // Initialize the bug type.
  BlockInCritSectionBugType.reset(
      new BugType(this, "Call to blocking function in critical section",
                        "Blocking Error"));
}

void BlockInCriticalSectionChecker::initIdentifierInfo(ASTContext &Ctx) const {
  if (!IdentifierInfoInitialized) {
    /* In case of checking C code, or when the corresponding headers are not
     * included, we might end up query the identifier table every time when this
     * function is called instead of early returning it. To avoid this, a bool
     * variable (IdentifierInfoInitialized) is used and the function will be run
     * only once. */
    IILockGuard  = &Ctx.Idents.get(ClassLockGuard);
    IIUniqueLock = &Ctx.Idents.get(ClassUniqueLock);
    IdentifierInfoInitialized = true;
  }
}

bool BlockInCriticalSectionChecker::isBlockingFunction(const CallEvent &Call) const {
  return matchesAny(Call, SleepFn, GetcFn, FgetsFn, ReadFn, RecvFn);
}

bool BlockInCriticalSectionChecker::isLockFunction(const CallEvent &Call) const {
  if (const auto *Ctor = dyn_cast<CXXConstructorCall>(&Call)) {
    auto IdentifierInfo = Ctor->getDecl()->getParent()->getIdentifier();
    if (IdentifierInfo == IILockGuard || IdentifierInfo == IIUniqueLock)
      return true;
  }

  return matchesAny(Call, LockFn, PthreadLockFn, PthreadTryLockFn, MtxLock,
                    MtxTimedLock, MtxTryLock);
}

bool BlockInCriticalSectionChecker::isUnlockFunction(const CallEvent &Call) const {
  if (const auto *Dtor = dyn_cast<CXXDestructorCall>(&Call)) {
    const auto *DRecordDecl = cast<CXXRecordDecl>(Dtor->getDecl()->getParent());
    auto IdentifierInfo = DRecordDecl->getIdentifier();
    if (IdentifierInfo == IILockGuard || IdentifierInfo == IIUniqueLock)
      return true;
  }

  return matchesAny(Call, UnlockFn, PthreadUnlockFn, MtxUnlock);
}

void BlockInCriticalSectionChecker::checkPostCall(const CallEvent &Call,
                                                  CheckerContext &C) const {
  initIdentifierInfo(C.getASTContext());

  if (!isBlockingFunction(Call)
      && !isLockFunction(Call)
      && !isUnlockFunction(Call))
    return;

  ProgramStateRef State = C.getState();
  unsigned mutexCount = State->get<MutexCounter>();
  if (isUnlockFunction(Call) && mutexCount > 0) {
    State = State->set<MutexCounter>(--mutexCount);
    C.addTransition(State);
  } else if (isLockFunction(Call)) {
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

  std::string msg;
  llvm::raw_string_ostream os(msg);
  os << "Call to blocking function '" << Call.getCalleeIdentifier()->getName()
     << "' inside of critical section";
  auto R = std::make_unique<PathSensitiveBugReport>(*BlockInCritSectionBugType,
                                                    os.str(), ErrNode);
  R->addRange(Call.getSourceRange());
  R->markInteresting(BlockDescSym);
  C.emitReport(std::move(R));
}

void ento::registerBlockInCriticalSectionChecker(CheckerManager &mgr) {
  mgr.registerChecker<BlockInCriticalSectionChecker>();
}

bool ento::shouldRegisterBlockInCriticalSectionChecker(const CheckerManager &mgr) {
  return true;
}

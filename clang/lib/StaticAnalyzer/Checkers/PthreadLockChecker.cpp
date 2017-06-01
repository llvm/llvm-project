//===--- PthreadLockChecker.cpp - Check for locking problems ---*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines PthreadLockChecker, a simple lock -> unlock checker.
// Also handles XNU locks, which behave similarly enough to share code.
//
//===----------------------------------------------------------------------===//

#include "ClangSACheckers.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramStateTrait.h"

using namespace clang;
using namespace ento;

namespace {

struct LockState {
  enum Kind { Destroyed, Locked, Unlocked } K;

private:
  LockState(Kind K) : K(K) {}

public:
  static LockState getLocked() { return LockState(Locked); }
  static LockState getUnlocked() { return LockState(Unlocked); }
  static LockState getDestroyed() { return LockState(Destroyed); }

  bool operator==(const LockState &X) const {
    return K == X.K;
  }

  bool isLocked() const { return K == Locked; }
  bool isUnlocked() const { return K == Unlocked; }
  bool isDestroyed() const { return K == Destroyed; }

  void Profile(llvm::FoldingSetNodeID &ID) const {
    ID.AddInteger(K);
  }
};

class PthreadLockChecker : public Checker< check::PostStmt<CallExpr> > {
  mutable std::unique_ptr<BugType> BT_doublelock;
  mutable std::unique_ptr<BugType> BT_doubleunlock;
  mutable std::unique_ptr<BugType> BT_destroylock;
  mutable std::unique_ptr<BugType> BT_initlock;
  mutable std::unique_ptr<BugType> BT_lor;
  enum LockingSemantics {
    NotApplicable = 0,
    PthreadSemantics,
    XNUSemantics
  };
public:
  void checkPostStmt(const CallExpr *CE, CheckerContext &C) const;

  void AcquireLock(CheckerContext &C, const CallExpr *CE, SVal lock,
                   bool isTryLock, enum LockingSemantics semantics) const;

  void ReleaseLock(CheckerContext &C, const CallExpr *CE, SVal lock) const;
  void DestroyLock(CheckerContext &C, const CallExpr *CE, SVal Lock) const;
  void InitLock(CheckerContext &C, const CallExpr *CE, SVal Lock) const;
  void reportUseDestroyedBug(CheckerContext &C, const CallExpr *CE) const;
};
} // end anonymous namespace

// GDM Entry for tracking lock state.
REGISTER_LIST_WITH_PROGRAMSTATE(LockSet, const MemRegion *)

REGISTER_MAP_WITH_PROGRAMSTATE(LockMap, const MemRegion *, LockState)

void PthreadLockChecker::checkPostStmt(const CallExpr *CE,
                                       CheckerContext &C) const {
  ProgramStateRef state = C.getState();
  const LocationContext *LCtx = C.getLocationContext();
  StringRef FName = C.getCalleeName(CE);
  if (FName.empty())
    return;

  if (CE->getNumArgs() != 1 && CE->getNumArgs() != 2)
    return;

  if (FName == "pthread_mutex_lock" ||
      FName == "pthread_rwlock_rdlock" ||
      FName == "pthread_rwlock_wrlock")
    AcquireLock(C, CE, state->getSVal(CE->getArg(0), LCtx),
                false, PthreadSemantics);
  else if (FName == "lck_mtx_lock" ||
           FName == "lck_rw_lock_exclusive" ||
           FName == "lck_rw_lock_shared")
    AcquireLock(C, CE, state->getSVal(CE->getArg(0), LCtx),
                false, XNUSemantics);
  else if (FName == "pthread_mutex_trylock" ||
           FName == "pthread_rwlock_tryrdlock" ||
           FName == "pthread_rwlock_trywrlock")
    AcquireLock(C, CE, state->getSVal(CE->getArg(0), LCtx),
                true, PthreadSemantics);
  else if (FName == "lck_mtx_try_lock" ||
           FName == "lck_rw_try_lock_exclusive" ||
           FName == "lck_rw_try_lock_shared")
    AcquireLock(C, CE, state->getSVal(CE->getArg(0), LCtx),
                true, XNUSemantics);
  else if (FName == "pthread_mutex_unlock" ||
           FName == "pthread_rwlock_unlock" ||
           FName == "lck_mtx_unlock" ||
           FName == "lck_rw_done")
    ReleaseLock(C, CE, state->getSVal(CE->getArg(0), LCtx));
  else if (FName == "pthread_mutex_destroy" ||
           FName == "lck_mtx_destroy")
    DestroyLock(C, CE, state->getSVal(CE->getArg(0), LCtx));
  else if (FName == "pthread_mutex_init")
    InitLock(C, CE, state->getSVal(CE->getArg(0), LCtx));
}

void PthreadLockChecker::AcquireLock(CheckerContext &C, const CallExpr *CE,
                                     SVal lock, bool isTryLock,
                                     enum LockingSemantics semantics) const {

  const MemRegion *lockR = lock.getAsRegion();
  if (!lockR)
    return;

  ProgramStateRef state = C.getState();

  SVal X = state->getSVal(CE, C.getLocationContext());
  if (X.isUnknownOrUndef())
    return;

  DefinedSVal retVal = X.castAs<DefinedSVal>();

  if (const LockState *LState = state->get<LockMap>(lockR)) {
    if (LState->isLocked()) {
      if (!BT_doublelock)
        BT_doublelock.reset(new BugType(this, "Double locking",
                                        "Lock checker"));
      ExplodedNode *N = C.generateErrorNode();
      if (!N)
        return;
      auto report = llvm::make_unique<BugReport>(
          *BT_doublelock, "This lock has already been acquired", N);
      report->addRange(CE->getArg(0)->getSourceRange());
      C.emitReport(std::move(report));
      return;
    } else if (LState->isDestroyed()) {
      reportUseDestroyedBug(C, CE);
      return;
    }
  }

  ProgramStateRef lockSucc = state;
  if (isTryLock) {
    // Bifurcate the state, and allow a mode where the lock acquisition fails.
    ProgramStateRef lockFail;
    switch (semantics) {
    case PthreadSemantics:
      std::tie(lockFail, lockSucc) = state->assume(retVal);
      break;
    case XNUSemantics:
      std::tie(lockSucc, lockFail) = state->assume(retVal);
      break;
    default:
      llvm_unreachable("Unknown tryLock locking semantics");
    }
    assert(lockFail && lockSucc);
    C.addTransition(lockFail);

  } else if (semantics == PthreadSemantics) {
    // Assume that the return value was 0.
    lockSucc = state->assume(retVal, false);
    assert(lockSucc);

  } else {
    // XNU locking semantics return void on non-try locks
    assert((semantics == XNUSemantics) && "Unknown locking semantics");
    lockSucc = state;
  }

  // Record that the lock was acquired.
  lockSucc = lockSucc->add<LockSet>(lockR);
  lockSucc = lockSucc->set<LockMap>(lockR, LockState::getLocked());
  C.addTransition(lockSucc);
}

void PthreadLockChecker::ReleaseLock(CheckerContext &C, const CallExpr *CE,
                                     SVal lock) const {

  const MemRegion *lockR = lock.getAsRegion();
  if (!lockR)
    return;

  ProgramStateRef state = C.getState();

  if (const LockState *LState = state->get<LockMap>(lockR)) {
    if (LState->isUnlocked()) {
      if (!BT_doubleunlock)
        BT_doubleunlock.reset(new BugType(this, "Double unlocking",
                                          "Lock checker"));
      ExplodedNode *N = C.generateErrorNode();
      if (!N)
        return;
      auto Report = llvm::make_unique<BugReport>(
          *BT_doubleunlock, "This lock has already been unlocked", N);
      Report->addRange(CE->getArg(0)->getSourceRange());
      C.emitReport(std::move(Report));
      return;
    } else if (LState->isDestroyed()) {
      reportUseDestroyedBug(C, CE);
      return;
    }
  }

  LockSetTy LS = state->get<LockSet>();

  // FIXME: Better analysis requires IPA for wrappers.

  if (!LS.isEmpty()) {
    const MemRegion *firstLockR = LS.getHead();
    if (firstLockR != lockR) {
      if (!BT_lor)
        BT_lor.reset(new BugType(this, "Lock order reversal", "Lock checker"));
      ExplodedNode *N = C.generateErrorNode();
      if (!N)
        return;
      auto report = llvm::make_unique<BugReport>(
          *BT_lor, "This was not the most recently acquired lock. Possible "
                   "lock order reversal", N);
      report->addRange(CE->getArg(0)->getSourceRange());
      C.emitReport(std::move(report));
      return;
    }
    // Record that the lock was released.
    state = state->set<LockSet>(LS.getTail());
  }

  state = state->set<LockMap>(lockR, LockState::getUnlocked());
  C.addTransition(state);
}

void PthreadLockChecker::DestroyLock(CheckerContext &C, const CallExpr *CE,
                                     SVal Lock) const {

  const MemRegion *LockR = Lock.getAsRegion();
  if (!LockR)
    return;

  ProgramStateRef State = C.getState();

  const LockState *LState = State->get<LockMap>(LockR);
  if (!LState || LState->isUnlocked()) {
    State = State->set<LockMap>(LockR, LockState::getDestroyed());
    C.addTransition(State);
    return;
  }

  StringRef Message;

  if (LState->isLocked()) {
    Message = "This lock is still locked";
  } else {
    Message = "This lock has already been destroyed";
  }

  if (!BT_destroylock)
    BT_destroylock.reset(new BugType(this, "Destroy invalid lock",
                                     "Lock checker"));
  ExplodedNode *N = C.generateErrorNode();
  if (!N)
    return;
  auto Report = llvm::make_unique<BugReport>(*BT_destroylock, Message, N);
  Report->addRange(CE->getArg(0)->getSourceRange());
  C.emitReport(std::move(Report));
}

void PthreadLockChecker::InitLock(CheckerContext &C, const CallExpr *CE,
                                  SVal Lock) const {

  const MemRegion *LockR = Lock.getAsRegion();
  if (!LockR)
    return;

  ProgramStateRef State = C.getState();

  const struct LockState *LState = State->get<LockMap>(LockR);
  if (!LState || LState->isDestroyed()) {
    State = State->set<LockMap>(LockR, LockState::getUnlocked());
    C.addTransition(State);
    return;
  }

  StringRef Message;

  if (LState->isLocked()) {
    Message = "This lock is still being held";
  } else {
    Message = "This lock has already been initialized";
  }

  if (!BT_initlock)
    BT_initlock.reset(new BugType(this, "Init invalid lock",
                                  "Lock checker"));
  ExplodedNode *N = C.generateErrorNode();
  if (!N)
    return;
  auto Report = llvm::make_unique<BugReport>(*BT_initlock, Message, N);
  Report->addRange(CE->getArg(0)->getSourceRange());
  C.emitReport(std::move(Report));
}

void PthreadLockChecker::reportUseDestroyedBug(CheckerContext &C,
                                               const CallExpr *CE) const {
  if (!BT_destroylock)
    BT_destroylock.reset(new BugType(this, "Use destroyed lock",
                                     "Lock checker"));
  ExplodedNode *N = C.generateErrorNode();
  if (!N)
    return;
  auto Report = llvm::make_unique<BugReport>(
      *BT_destroylock, "This lock has already been destroyed", N);
  Report->addRange(CE->getArg(0)->getSourceRange());
  C.emitReport(std::move(Report));
}

void ento::registerPthreadLockChecker(CheckerManager &mgr) {
  mgr.registerChecker<PthreadLockChecker>();
}

//===--- PthreadLockChecker.cpp - Check for locking problems ---*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This defines PthreadLockChecker, a simple lock -> unlock checker.
// Also handles XNU locks, which behave similarly enough to share code.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"

using namespace clang;
using namespace ento;

namespace {

struct LockState {
  enum Kind {
    Destroyed,
    Locked,
    Unlocked,
    UntouchedAndPossiblyDestroyed,
    UnlockedAndPossiblyDestroyed
  } K;

private:
  LockState(Kind K) : K(K) {}

public:
  static LockState getLocked() { return LockState(Locked); }
  static LockState getUnlocked() { return LockState(Unlocked); }
  static LockState getDestroyed() { return LockState(Destroyed); }
  static LockState getUntouchedAndPossiblyDestroyed() {
    return LockState(UntouchedAndPossiblyDestroyed);
  }
  static LockState getUnlockedAndPossiblyDestroyed() {
    return LockState(UnlockedAndPossiblyDestroyed);
  }

  bool operator==(const LockState &X) const {
    return K == X.K;
  }

  bool isLocked() const { return K == Locked; }
  bool isUnlocked() const { return K == Unlocked; }
  bool isDestroyed() const { return K == Destroyed; }
  bool isUntouchedAndPossiblyDestroyed() const {
    return K == UntouchedAndPossiblyDestroyed;
  }
  bool isUnlockedAndPossiblyDestroyed() const {
    return K == UnlockedAndPossiblyDestroyed;
  }

  void Profile(llvm::FoldingSetNodeID &ID) const {
    ID.AddInteger(K);
  }
};

class PthreadLockChecker
    : public Checker<check::PostCall, check::DeadSymbols,
                     check::RegionChanges> {
  BugType BT_doublelock{this, "Double locking", "Lock checker"},
          BT_doubleunlock{this, "Double unlocking", "Lock checker"},
          BT_destroylock{this, "Use destroyed lock", "Lock checker"},
          BT_initlock{this, "Init invalid lock", "Lock checker"},
          BT_lor{this, "Lock order reversal", "Lock checker"};

  enum LockingSemantics {
    NotApplicable = 0,
    PthreadSemantics,
    XNUSemantics
  };

  typedef void (PthreadLockChecker::*FnCheck)(const CallEvent &Call,
                                              CheckerContext &C) const;
  CallDescriptionMap<FnCheck> Callbacks = {
    // Init.
    {{"pthread_mutex_init",        2}, &PthreadLockChecker::InitAnyLock},
    // TODO: pthread_rwlock_init(2 arguments).
    // TODO: lck_mtx_init(3 arguments).
    // TODO: lck_mtx_alloc_init(2 arguments) => returns the mutex.
    // TODO: lck_rw_init(3 arguments).
    // TODO: lck_rw_alloc_init(2 arguments) => returns the mutex.

    // Acquire.
    {{"pthread_mutex_lock",        1}, &PthreadLockChecker::AcquirePthreadLock},
    {{"pthread_rwlock_rdlock",     1}, &PthreadLockChecker::AcquirePthreadLock},
    {{"pthread_rwlock_wrlock",     1}, &PthreadLockChecker::AcquirePthreadLock},
    {{"lck_mtx_lock",              1}, &PthreadLockChecker::AcquireXNULock},
    {{"lck_rw_lock_exclusive",     1}, &PthreadLockChecker::AcquireXNULock},
    {{"lck_rw_lock_shared",        1}, &PthreadLockChecker::AcquireXNULock},

    // Try.
    {{"pthread_mutex_trylock",     1}, &PthreadLockChecker::TryPthreadLock},
    {{"pthread_rwlock_tryrdlock",  1}, &PthreadLockChecker::TryPthreadLock},
    {{"pthread_rwlock_trywrlock",  1}, &PthreadLockChecker::TryPthreadLock},
    {{"lck_mtx_try_lock",          1}, &PthreadLockChecker::TryXNULock},
    {{"lck_rw_try_lock_exclusive", 1}, &PthreadLockChecker::TryXNULock},
    {{"lck_rw_try_lock_shared",    1}, &PthreadLockChecker::TryXNULock},

    // Release.
    {{"pthread_mutex_unlock",      1}, &PthreadLockChecker::ReleaseAnyLock},
    {{"pthread_rwlock_unlock",     1}, &PthreadLockChecker::ReleaseAnyLock},
    {{"lck_mtx_unlock",            1}, &PthreadLockChecker::ReleaseAnyLock},
    {{"lck_rw_unlock_exclusive",   1}, &PthreadLockChecker::ReleaseAnyLock},
    {{"lck_rw_unlock_shared",      1}, &PthreadLockChecker::ReleaseAnyLock},
    {{"lck_rw_done",               1}, &PthreadLockChecker::ReleaseAnyLock},

    // Destroy.
    {{"pthread_mutex_destroy",     1}, &PthreadLockChecker::DestroyPthreadLock},
    {{"lck_mtx_destroy",           2}, &PthreadLockChecker::DestroyXNULock},
    // TODO: pthread_rwlock_destroy(1 argument).
    // TODO: lck_rw_destroy(2 arguments).
  };

  ProgramStateRef resolvePossiblyDestroyedMutex(ProgramStateRef state,
                                                const MemRegion *lockR,
                                                const SymbolRef *sym) const;
  void reportUseDestroyedBug(const CallEvent &Call, CheckerContext &C,
                             unsigned ArgNo) const;

  // Init.
  void InitAnyLock(const CallEvent &Call, CheckerContext &C) const;
  void InitLockAux(const CallEvent &Call, CheckerContext &C, unsigned ArgNo,
                   SVal Lock) const;

  // Lock, Try-lock.
  void AcquirePthreadLock(const CallEvent &Call, CheckerContext &C) const;
  void AcquireXNULock(const CallEvent &Call, CheckerContext &C) const;
  void TryPthreadLock(const CallEvent &Call, CheckerContext &C) const;
  void TryXNULock(const CallEvent &Call, CheckerContext &C) const;
  void AcquireLockAux(const CallEvent &Call, CheckerContext &C, unsigned ArgNo,
                      SVal lock, bool isTryLock,
                      enum LockingSemantics semantics) const;

  // Release.
  void ReleaseAnyLock(const CallEvent &Call, CheckerContext &C) const;
  void ReleaseLockAux(const CallEvent &Call, CheckerContext &C, unsigned ArgNo,
                      SVal lock) const;

  // Destroy.
  void DestroyPthreadLock(const CallEvent &Call, CheckerContext &C) const;
  void DestroyXNULock(const CallEvent &Call, CheckerContext &C) const;
  void DestroyLockAux(const CallEvent &Call, CheckerContext &C, unsigned ArgNo,
                      SVal Lock, enum LockingSemantics semantics) const;

public:
  void checkPostCall(const CallEvent &Call, CheckerContext &C) const;
  void checkDeadSymbols(SymbolReaper &SymReaper, CheckerContext &C) const;
  ProgramStateRef
  checkRegionChanges(ProgramStateRef State, const InvalidatedSymbols *Symbols,
                     ArrayRef<const MemRegion *> ExplicitRegions,
                     ArrayRef<const MemRegion *> Regions,
                     const LocationContext *LCtx, const CallEvent *Call) const;
  void printState(raw_ostream &Out, ProgramStateRef State,
                  const char *NL, const char *Sep) const override;
};
} // end anonymous namespace

// A stack of locks for tracking lock-unlock order.
REGISTER_LIST_WITH_PROGRAMSTATE(LockSet, const MemRegion *)

// An entry for tracking lock states.
REGISTER_MAP_WITH_PROGRAMSTATE(LockMap, const MemRegion *, LockState)

// Return values for unresolved calls to pthread_mutex_destroy().
REGISTER_MAP_WITH_PROGRAMSTATE(DestroyRetVal, const MemRegion *, SymbolRef)

void PthreadLockChecker::checkPostCall(const CallEvent &Call,
                                       CheckerContext &C) const {
  // An additional umbrella check that all functions modeled by this checker
  // are global C functions.
  // TODO: Maybe make this the default behavior of CallDescription
  // with exactly one identifier?
  if (!Call.isGlobalCFunction())
    return;

  if (const FnCheck *Callback = Callbacks.lookup(Call))
    (this->**Callback)(Call, C);
}


// When a lock is destroyed, in some semantics(like PthreadSemantics) we are not
// sure if the destroy call has succeeded or failed, and the lock enters one of
// the 'possibly destroyed' state. There is a short time frame for the
// programmer to check the return value to see if the lock was successfully
// destroyed. Before we model the next operation over that lock, we call this
// function to see if the return value was checked by now and set the lock state
// - either to destroyed state or back to its previous state.

// In PthreadSemantics, pthread_mutex_destroy() returns zero if the lock is
// successfully destroyed and it returns a non-zero value otherwise.
ProgramStateRef PthreadLockChecker::resolvePossiblyDestroyedMutex(
    ProgramStateRef state, const MemRegion *lockR, const SymbolRef *sym) const {
  const LockState *lstate = state->get<LockMap>(lockR);
  // Existence in DestroyRetVal ensures existence in LockMap.
  // Existence in Destroyed also ensures that the lock state for lockR is either
  // UntouchedAndPossiblyDestroyed or UnlockedAndPossiblyDestroyed.
  assert(lstate->isUntouchedAndPossiblyDestroyed() ||
         lstate->isUnlockedAndPossiblyDestroyed());

  ConstraintManager &CMgr = state->getConstraintManager();
  ConditionTruthVal retZero = CMgr.isNull(state, *sym);
  if (retZero.isConstrainedFalse()) {
    if (lstate->isUntouchedAndPossiblyDestroyed())
      state = state->remove<LockMap>(lockR);
    else if (lstate->isUnlockedAndPossiblyDestroyed())
      state = state->set<LockMap>(lockR, LockState::getUnlocked());
  } else
    state = state->set<LockMap>(lockR, LockState::getDestroyed());

  // Removing the map entry (lockR, sym) from DestroyRetVal as the lock state is
  // now resolved.
  state = state->remove<DestroyRetVal>(lockR);
  return state;
}

void PthreadLockChecker::printState(raw_ostream &Out, ProgramStateRef State,
                                    const char *NL, const char *Sep) const {
  LockMapTy LM = State->get<LockMap>();
  if (!LM.isEmpty()) {
    Out << Sep << "Mutex states:" << NL;
    for (auto I : LM) {
      I.first->dumpToStream(Out);
      if (I.second.isLocked())
        Out << ": locked";
      else if (I.second.isUnlocked())
        Out << ": unlocked";
      else if (I.second.isDestroyed())
        Out << ": destroyed";
      else if (I.second.isUntouchedAndPossiblyDestroyed())
        Out << ": not tracked, possibly destroyed";
      else if (I.second.isUnlockedAndPossiblyDestroyed())
        Out << ": unlocked, possibly destroyed";
      Out << NL;
    }
  }

  LockSetTy LS = State->get<LockSet>();
  if (!LS.isEmpty()) {
    Out << Sep << "Mutex lock order:" << NL;
    for (auto I: LS) {
      I->dumpToStream(Out);
      Out << NL;
    }
  }

  // TODO: Dump destroyed mutex symbols?
}

void PthreadLockChecker::AcquirePthreadLock(const CallEvent &Call,
                                            CheckerContext &C) const {
  AcquireLockAux(Call, C, 0, Call.getArgSVal(0), false, PthreadSemantics);
}

void PthreadLockChecker::AcquireXNULock(const CallEvent &Call,
                                           CheckerContext &C) const {
  AcquireLockAux(Call, C, 0, Call.getArgSVal(0), false, XNUSemantics);
}

void PthreadLockChecker::TryPthreadLock(const CallEvent &Call,
                                        CheckerContext &C) const {
  AcquireLockAux(Call, C, 0, Call.getArgSVal(0), true, PthreadSemantics);
}

void PthreadLockChecker::TryXNULock(const CallEvent &Call,
                                        CheckerContext &C) const {
  AcquireLockAux(Call, C, 0, Call.getArgSVal(0), true, PthreadSemantics);
}

void PthreadLockChecker::AcquireLockAux(const CallEvent &Call,
                                        CheckerContext &C, unsigned ArgNo,
                                        SVal lock, bool isTryLock,
                                        enum LockingSemantics semantics) const {

  const MemRegion *lockR = lock.getAsRegion();
  if (!lockR)
    return;

  ProgramStateRef state = C.getState();
  const SymbolRef *sym = state->get<DestroyRetVal>(lockR);
  if (sym)
    state = resolvePossiblyDestroyedMutex(state, lockR, sym);

  if (const LockState *LState = state->get<LockMap>(lockR)) {
    if (LState->isLocked()) {
      ExplodedNode *N = C.generateErrorNode();
      if (!N)
        return;
      auto report = std::make_unique<PathSensitiveBugReport>(
          BT_doublelock, "This lock has already been acquired", N);
      report->addRange(Call.getArgExpr(ArgNo)->getSourceRange());
      C.emitReport(std::move(report));
      return;
    } else if (LState->isDestroyed()) {
      reportUseDestroyedBug(Call, C, ArgNo);
      return;
    }
  }

  ProgramStateRef lockSucc = state;
  if (isTryLock) {
    // Bifurcate the state, and allow a mode where the lock acquisition fails.
    SVal RetVal = Call.getReturnValue();
    if (auto DefinedRetVal = RetVal.getAs<DefinedSVal>()) {
      ProgramStateRef lockFail;
      switch (semantics) {
      case PthreadSemantics:
        std::tie(lockFail, lockSucc) = state->assume(*DefinedRetVal);
        break;
      case XNUSemantics:
        std::tie(lockSucc, lockFail) = state->assume(*DefinedRetVal);
        break;
      default:
        llvm_unreachable("Unknown tryLock locking semantics");
      }
      assert(lockFail && lockSucc);
      C.addTransition(lockFail);
    }
    // We might want to handle the case when the mutex lock function was inlined
    // and returned an Unknown or Undefined value.
  } else if (semantics == PthreadSemantics) {
    // Assume that the return value was 0.
    SVal RetVal = Call.getReturnValue();
    if (auto DefinedRetVal = RetVal.getAs<DefinedSVal>()) {
      // FIXME: If the lock function was inlined and returned true,
      // we need to behave sanely - at least generate sink.
      lockSucc = state->assume(*DefinedRetVal, false);
      assert(lockSucc);
    }
    // We might want to handle the case when the mutex lock function was inlined
    // and returned an Unknown or Undefined value.
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

void PthreadLockChecker::ReleaseAnyLock(const CallEvent &Call,
                                        CheckerContext &C) const {
  ReleaseLockAux(Call, C, 0, Call.getArgSVal(0));
}

void PthreadLockChecker::ReleaseLockAux(const CallEvent &Call,
                                        CheckerContext &C, unsigned ArgNo,
                                        SVal lock) const {

  const MemRegion *lockR = lock.getAsRegion();
  if (!lockR)
    return;

  ProgramStateRef state = C.getState();
  const SymbolRef *sym = state->get<DestroyRetVal>(lockR);
  if (sym)
    state = resolvePossiblyDestroyedMutex(state, lockR, sym);

  if (const LockState *LState = state->get<LockMap>(lockR)) {
    if (LState->isUnlocked()) {
      ExplodedNode *N = C.generateErrorNode();
      if (!N)
        return;
      auto Report = std::make_unique<PathSensitiveBugReport>(
          BT_doubleunlock, "This lock has already been unlocked", N);
      Report->addRange(Call.getArgExpr(ArgNo)->getSourceRange());
      C.emitReport(std::move(Report));
      return;
    } else if (LState->isDestroyed()) {
      reportUseDestroyedBug(Call, C, ArgNo);
      return;
    }
  }

  LockSetTy LS = state->get<LockSet>();

  if (!LS.isEmpty()) {
    const MemRegion *firstLockR = LS.getHead();
    if (firstLockR != lockR) {
      ExplodedNode *N = C.generateErrorNode();
      if (!N)
        return;
      auto report = std::make_unique<PathSensitiveBugReport>(
          BT_lor, "This was not the most recently acquired lock. Possible "
                  "lock order reversal", N);
      report->addRange(Call.getArgExpr(ArgNo)->getSourceRange());
      C.emitReport(std::move(report));
      return;
    }
    // Record that the lock was released.
    state = state->set<LockSet>(LS.getTail());
  }

  state = state->set<LockMap>(lockR, LockState::getUnlocked());
  C.addTransition(state);
}

void PthreadLockChecker::DestroyPthreadLock(const CallEvent &Call,
                                            CheckerContext &C) const {
  DestroyLockAux(Call, C, 0, Call.getArgSVal(0), PthreadSemantics);
}

void PthreadLockChecker::DestroyXNULock(const CallEvent &Call,
                                            CheckerContext &C) const {
  DestroyLockAux(Call, C, 0, Call.getArgSVal(0), XNUSemantics);
}

void PthreadLockChecker::DestroyLockAux(const CallEvent &Call,
                                        CheckerContext &C, unsigned ArgNo,
                                        SVal Lock,
                                        enum LockingSemantics semantics) const {

  const MemRegion *LockR = Lock.getAsRegion();
  if (!LockR)
    return;

  ProgramStateRef State = C.getState();

  const SymbolRef *sym = State->get<DestroyRetVal>(LockR);
  if (sym)
    State = resolvePossiblyDestroyedMutex(State, LockR, sym);

  const LockState *LState = State->get<LockMap>(LockR);
  // Checking the return value of the destroy method only in the case of
  // PthreadSemantics
  if (semantics == PthreadSemantics) {
    if (!LState || LState->isUnlocked()) {
      SymbolRef sym = Call.getReturnValue().getAsSymbol();
      if (!sym) {
        State = State->remove<LockMap>(LockR);
        C.addTransition(State);
        return;
      }
      State = State->set<DestroyRetVal>(LockR, sym);
      if (LState && LState->isUnlocked())
        State = State->set<LockMap>(
            LockR, LockState::getUnlockedAndPossiblyDestroyed());
      else
        State = State->set<LockMap>(
            LockR, LockState::getUntouchedAndPossiblyDestroyed());
      C.addTransition(State);
      return;
    }
  } else {
    if (!LState || LState->isUnlocked()) {
      State = State->set<LockMap>(LockR, LockState::getDestroyed());
      C.addTransition(State);
      return;
    }
  }
  StringRef Message;

  if (LState->isLocked()) {
    Message = "This lock is still locked";
  } else {
    Message = "This lock has already been destroyed";
  }

  ExplodedNode *N = C.generateErrorNode();
  if (!N)
    return;
  auto Report =
      std::make_unique<PathSensitiveBugReport>(BT_destroylock, Message, N);
  Report->addRange(Call.getArgExpr(ArgNo)->getSourceRange());
  C.emitReport(std::move(Report));
}

void PthreadLockChecker::InitAnyLock(const CallEvent &Call,
                                     CheckerContext &C) const {
  InitLockAux(Call, C, 0, Call.getArgSVal(0));
}

void PthreadLockChecker::InitLockAux(const CallEvent &Call, CheckerContext &C,
                                     unsigned ArgNo, SVal Lock) const {

  const MemRegion *LockR = Lock.getAsRegion();
  if (!LockR)
    return;

  ProgramStateRef State = C.getState();

  const SymbolRef *sym = State->get<DestroyRetVal>(LockR);
  if (sym)
    State = resolvePossiblyDestroyedMutex(State, LockR, sym);

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

  ExplodedNode *N = C.generateErrorNode();
  if (!N)
    return;
  auto Report =
      std::make_unique<PathSensitiveBugReport>(BT_initlock, Message, N);
  Report->addRange(Call.getArgExpr(ArgNo)->getSourceRange());
  C.emitReport(std::move(Report));
}

void PthreadLockChecker::reportUseDestroyedBug(const CallEvent &Call,
                                               CheckerContext &C,
                                               unsigned ArgNo) const {
  ExplodedNode *N = C.generateErrorNode();
  if (!N)
    return;
  auto Report = std::make_unique<PathSensitiveBugReport>(
      BT_destroylock, "This lock has already been destroyed", N);
  Report->addRange(Call.getArgExpr(ArgNo)->getSourceRange());
  C.emitReport(std::move(Report));
}

void PthreadLockChecker::checkDeadSymbols(SymbolReaper &SymReaper,
                                          CheckerContext &C) const {
  ProgramStateRef State = C.getState();

  for (auto I : State->get<DestroyRetVal>()) {
    // Once the return value symbol dies, no more checks can be performed
    // against it. See if the return value was checked before this point.
    // This would remove the symbol from the map as well.
    if (SymReaper.isDead(I.second))
      State = resolvePossiblyDestroyedMutex(State, I.first, &I.second);
  }

  for (auto I : State->get<LockMap>()) {
    // Stop tracking dead mutex regions as well.
    if (!SymReaper.isLiveRegion(I.first))
      State = State->remove<LockMap>(I.first);
  }

  // TODO: We probably need to clean up the lock stack as well.
  // It is tricky though: even if the mutex cannot be unlocked anymore,
  // it can still participate in lock order reversal resolution.

  C.addTransition(State);
}

ProgramStateRef PthreadLockChecker::checkRegionChanges(
    ProgramStateRef State, const InvalidatedSymbols *Symbols,
    ArrayRef<const MemRegion *> ExplicitRegions,
    ArrayRef<const MemRegion *> Regions, const LocationContext *LCtx,
    const CallEvent *Call) const {

  bool IsLibraryFunction = false;
  if (Call && Call->isGlobalCFunction()) {
    // Avoid invalidating mutex state when a known supported function is called.
    if (Callbacks.lookup(*Call))
        return State;

    if (Call->isInSystemHeader())
      IsLibraryFunction = true;
  }

  for (auto R : Regions) {
    // We assume that system library function wouldn't touch the mutex unless
    // it takes the mutex explicitly as an argument.
    // FIXME: This is a bit quadratic.
    if (IsLibraryFunction &&
        std::find(ExplicitRegions.begin(), ExplicitRegions.end(), R) ==
            ExplicitRegions.end())
      continue;

    State = State->remove<LockMap>(R);
    State = State->remove<DestroyRetVal>(R);

    // TODO: We need to invalidate the lock stack as well. This is tricky
    // to implement correctly and efficiently though, because the effects
    // of mutex escapes on lock order may be fairly varied.
  }

  return State;
}

void ento::registerPthreadLockChecker(CheckerManager &mgr) {
  mgr.registerChecker<PthreadLockChecker>();
}

bool ento::shouldRegisterPthreadLockChecker(const LangOptions &LO) {
  return true;
}

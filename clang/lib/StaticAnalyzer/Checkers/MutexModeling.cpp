//===--- MutexModeling.cpp - Modeling of mutexes --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines modeling checker for tracking mutex states.
//
//===----------------------------------------------------------------------===//

#include "MutexModeling/MutexModelingAPI.h"
#include "MutexModeling/MutexModelingDomain.h"
#include "MutexModeling/MutexRegionExtractor.h"

#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerHelpers.h"
#include "clang/StaticAnalyzer/Frontend/CheckerRegistry.h"
#include <memory>

using namespace clang;
using namespace ento;
using namespace mutex_modeling;

namespace {

// When a lock is destroyed, in some semantics(like PthreadSemantics) we are not
// sure if the destroy call has succeeded or failed, and the lock enters one of
// the 'possibly destroyed' state. There is a short time frame for the
// programmer to check the return value to see if the lock was successfully
// destroyed. Before we model the next operation over that lock, we call this
// function to see if the return value was checked by now and set the lock state
// - either to destroyed state or back to its previous state.

// In PthreadSemantics, pthread_mutex_destroy() returns zero if the lock is
// successfully destroyed and it returns a non-zero value otherwise.
ProgramStateRef resolvePossiblyDestroyedMutex(ProgramStateRef State,
                                              const MemRegion *LockReg,
                                              const SymbolRef *LockReturnSym) {
  const LockStateKind *LockState = State->get<LockStates>(LockReg);
  // Existence in DestroyRetVal ensures existence in LockMap.
  // Existence in Destroyed also ensures that the lock state for lockR is either
  // UntouchedAndPossiblyDestroyed or UnlockedAndPossiblyDestroyed.
  assert(LockState);
  assert(*LockState == LockStateKind::UntouchedAndPossiblyDestroyed ||
         *LockState == LockStateKind::UnlockedAndPossiblyDestroyed);

  ConstraintManager &CMgr = State->getConstraintManager();
  ConditionTruthVal RetZero = CMgr.isNull(State, *LockReturnSym);
  if (RetZero.isConstrainedFalse()) {
    switch (*LockState) {
    case LockStateKind::UntouchedAndPossiblyDestroyed: {
      State = State->remove<LockStates>(LockReg);
      break;
    }
    case LockStateKind::UnlockedAndPossiblyDestroyed: {
      State = State->set<LockStates>(LockReg, LockStateKind::Unlocked);
      break;
    }
    default:
      llvm_unreachable("Unknown lock state for a lock inside DestroyRetVal");
    }
  } else {
    State = State->set<LockStates>(LockReg, LockStateKind::Destroyed);
  }

  // Removing the map entry (LockReg, sym) from DestroyRetVal as the lock
  // state is now resolved.
  return State->remove<DestroyedRetVals>(LockReg);
}

ProgramStateRef doResolvePossiblyDestroyedMutex(ProgramStateRef State,
                                                const MemRegion *MTX) {
  assert(MTX && "should only be called with a mutex region");

  if (const SymbolRef *Sym = State->get<DestroyedRetVals>(MTX))
    return resolvePossiblyDestroyedMutex(State, MTX, Sym);
  return State;
}

class MutexModeling : public Checker<check::PostCall, check::DeadSymbols,
                                     check::RegionChanges> {
public:
  void checkPostCall(const CallEvent &Call, CheckerContext &C) const;

  void checkDeadSymbols(SymbolReaper &SymReaper, CheckerContext &C) const;

  ProgramStateRef
  checkRegionChanges(ProgramStateRef State, const InvalidatedSymbols *Symbols,
                     ArrayRef<const MemRegion *> ExplicitRegions,
                     ArrayRef<const MemRegion *> Regions,
                     const LocationContext *LCtx, const CallEvent *Call) const;

private:
  mutable std::unique_ptr<BugType> BT_initlock;

  // When handling events, NoteTags can be placed on ProgramPoints. This struct
  // supports returning both the resulting ProgramState and a NoteTag.
  struct ModelingResult {
    ProgramStateRef State;
    const NoteTag *Note = nullptr;
  };

  ModelingResult handleInit(const EventDescriptor &Event, const MemRegion *MTX,
                            const CallEvent &Call, ProgramStateRef State,
                            CheckerContext &C) const;
  ModelingResult onSuccessfulAcquire(const MemRegion *MTX,
                                     const CallEvent &Call,
                                     ProgramStateRef State,
                                     CheckerContext &C) const;
  ModelingResult markCritSection(ModelingResult InputState,
                                 const MemRegion *MTX, const CallEvent &Call,
                                 CheckerContext &C) const;
  ModelingResult handleAcquire(const EventDescriptor &Event,
                               const MemRegion *MTX, const CallEvent &Call,
                               ProgramStateRef State, CheckerContext &C) const;
  ModelingResult handleTryAcquire(const EventDescriptor &Event,
                                  const MemRegion *MTX, const CallEvent &Call,
                                  ProgramStateRef State,
                                  CheckerContext &C) const;
  ModelingResult handleRelease(const EventDescriptor &Event,
                               const MemRegion *MTX, const CallEvent &Call,
                               ProgramStateRef State, CheckerContext &C) const;
  ModelingResult handleDestroy(const EventDescriptor &Event,
                               const MemRegion *MTX, const CallEvent &Call,
                               ProgramStateRef State, CheckerContext &C) const;
  ModelingResult handleEvent(const EventDescriptor &Event, const MemRegion *MTX,
                             const CallEvent &Call, ProgramStateRef State,
                             CheckerContext &C) const;
};

} // namespace

MutexModeling::ModelingResult
MutexModeling::handleInit(const EventDescriptor &Event, const MemRegion *MTX,
                          const CallEvent &Call, ProgramStateRef State,
                          CheckerContext &C) const {
  ModelingResult Result{State->set<LockStates>(MTX, LockStateKind::Unlocked)};

  const LockStateKind *LockState = State->get<LockStates>(MTX);

  if (!LockState)
    return Result;

  switch (*LockState) {
  case (LockStateKind::Destroyed): {
    Result.State = State->set<LockStates>(MTX, LockStateKind::Unlocked);
    break;
  }
  case (LockStateKind::Locked): {
    Result.State =
        State->set<LockStates>(MTX, LockStateKind::Error_DoubleInitWhileLocked);
    break;
  }
  default: {
    Result.State = State->set<LockStates>(MTX, LockStateKind::Error_DoubleInit);
    break;
  }
  }

  return Result;
}

MutexModeling::ModelingResult
MutexModeling::markCritSection(MutexModeling::ModelingResult InputState,
                               const MemRegion *MTX, const CallEvent &Call,
                               CheckerContext &C) const {
  const CritSectionMarker MarkToAdd{Call.getOriginExpr(), MTX};
  return {InputState.State->add<CritSections>(MarkToAdd),
          CreateMutexCritSectionNote(MarkToAdd, C)};
}

MutexModeling::ModelingResult
MutexModeling::onSuccessfulAcquire(const MemRegion *MTX, const CallEvent &Call,
                                   ProgramStateRef State,
                                   CheckerContext &C) const {
  ModelingResult Result{State};

  const LockStateKind *LockState = State->get<LockStates>(MTX);

  if (!LockState) {
    Result.State = Result.State->set<LockStates>(MTX, LockStateKind::Locked);
  } else {
    switch (*LockState) {
    case LockStateKind::Unlocked:
      Result.State = Result.State->set<LockStates>(MTX, LockStateKind::Locked);
      break;
    case LockStateKind::Locked:
      Result.State =
          Result.State->set<LockStates>(MTX, LockStateKind::Error_DoubleLock);
      break;
    case LockStateKind::Destroyed:
      Result.State = Result.State->set<LockStates>(
          MTX, LockStateKind::Error_LockDestroyed);
      break;
    default:
      break;
    }
  }

  Result = markCritSection(Result, MTX, Call, C);
  return Result;
}

MutexModeling::ModelingResult
MutexModeling::handleAcquire(const EventDescriptor &Event, const MemRegion *MTX,
                             const CallEvent &Call, ProgramStateRef State,
                             CheckerContext &C) const {

  switch (Event.Semantics) {
  case SemanticsKind::PthreadSemantics: {
    // Assume that the return value was 0.
    SVal RetVal = Call.getReturnValue();
    if (auto DefinedRetVal = RetVal.getAs<DefinedSVal>()) {
      // FIXME: If the lock function was inlined and returned true,
      // we need to behave sanely - at least generate sink.
      State = State->assume(*DefinedRetVal, false);
      assert(State);
    }
    // We might want to handle the case when the mutex lock function was
    // inlined and returned an Unknown or Undefined value.
    break;
  }
  case SemanticsKind::XNUSemantics:
    // XNU semantics return void on non-try locks.
    break;
  default:
    llvm_unreachable(
        "Acquire events should have either Pthread or XNU semantics");
  }

  return onSuccessfulAcquire(MTX, Call, State, C);
}

MutexModeling::ModelingResult MutexModeling::handleTryAcquire(
    const EventDescriptor &Event, const MemRegion *MTX, const CallEvent &Call,
    ProgramStateRef State, CheckerContext &C) const {

  ProgramStateRef LockSucc{State};
  // Bifurcate the state, and allow a mode where the lock acquisition fails.
  SVal RetVal = Call.getReturnValue();
  std::optional<DefinedSVal> DefinedRetVal = RetVal.getAs<DefinedSVal>();
  // Bifurcating the state is only meaningful if the call was not inlined, but
  // we can still reason about the return value.
  if (!C.wasInlined && DefinedRetVal) {
    ProgramStateRef LockFail;
    switch (Event.Semantics) {
    case SemanticsKind::PthreadSemantics:
      // For PthreadSemantics, a non-zero return value indicates success
      std::tie(LockFail, LockSucc) = State->assume(*DefinedRetVal);
      break;
    case SemanticsKind::XNUSemantics:
      // For XNUSemantics, a zero return value indicates success
      std::tie(LockSucc, LockFail) = State->assume(*DefinedRetVal);
      break;
    default:
      llvm_unreachable("Unknown TryLock locking semantics");
    }

    // This is the bifurcation point in the ExplodedGraph, we do not need to
    // return the new ExplodedGraph node because we do not plan on building this
    // lock-failed case path in this checker.
    C.addTransition(LockFail);
  }

  if (!LockSucc)
    LockSucc = State;

  // Pass the state where the locking succeeded onwards.
  return onSuccessfulAcquire(MTX, Call, LockSucc, C);
}

MutexModeling::ModelingResult
MutexModeling::handleRelease(const EventDescriptor &Event, const MemRegion *MTX,
                             const CallEvent &Call, ProgramStateRef State,
                             CheckerContext &C) const {

  ModelingResult Result{State};

  const LockStateKind *LockState = Result.State->get<LockStates>(MTX);

  if (!LockState) {
    Result.State = Result.State->set<LockStates>(MTX, LockStateKind::Unlocked);
    return Result;
  }

  if (*LockState == LockStateKind::Unlocked) {
    Result.State =
        State->set<LockStates>(MTX, LockStateKind::Error_DoubleUnlock);
    return Result;
  }

  if (*LockState == LockStateKind::Destroyed) {
    Result.State =
        State->set<LockStates>(MTX, LockStateKind::Error_UnlockDestroyed);
    return Result;
  }

  const auto ActiveSections = State->get<CritSections>();
  const auto MostRecentLockForMTX =
      llvm::find_if(ActiveSections,
                    [MTX](auto &&Marker) { return Marker.MutexRegion == MTX; });

  // In a non-empty critical section list, if the most recent lock is for
  // another mutex, then there is a lock reversal.
  bool IsLockInversion = MostRecentLockForMTX != ActiveSections.begin();

  // NOTE: IsLockInversion -> !ActiveSections.isEmpty()
  assert((!IsLockInversion || !ActiveSections.isEmpty()) &&
         "The existence of an inversion implies that the list is not empty");

  if (IsLockInversion) {
    Result.State =
        State->set<LockStates>(MTX, LockStateKind::Error_LockReversal);
    // Build a new ImmutableList without this element.
    auto &Factory = Result.State->get_context<CritSections>();
    llvm::ImmutableList<CritSectionMarker> WithoutThisLock =
        Factory.getEmptyList();
    for (auto It = ActiveSections.begin(), End = ActiveSections.end();
         It != End; ++It) {
      if (It != MostRecentLockForMTX)
        WithoutThisLock = Factory.add(*It, WithoutThisLock);
    }
    Result.State = Result.State->set<CritSections>(WithoutThisLock);
    return Result;
  }

  Result.State = Result.State->set<LockStates>(MTX, LockStateKind::Unlocked);
  // If there is no lock inversion, we can just remove the last crit section.
  // NOTE: It should be safe to call getTail on an empty list
  Result.State = Result.State->set<CritSections>(ActiveSections.getTail());

  return Result;
}

MutexModeling::ModelingResult
MutexModeling::handleDestroy(const EventDescriptor &Event, const MemRegion *MTX,
                             const CallEvent &Call, ProgramStateRef State,
                             CheckerContext &C) const {
  ModelingResult Result{State};

  const LockStateKind *LockState = Result.State->get<LockStates>(MTX);

  // PthreadSemantics handles destroy differently due to its return value
  // semantics
  if (Event.Semantics == SemanticsKind::PthreadSemantics) {
    if (!LockState || *LockState == LockStateKind::Unlocked) {
      SymbolRef Sym = Call.getReturnValue().getAsSymbol();
      if (!Sym) {
        Result.State = Result.State->remove<LockStates>(MTX);
        return Result;
      }
      Result.State = Result.State->set<DestroyedRetVals>(MTX, Sym);
      Result.State = Result.State->set<LockStates>(
          MTX, LockState && *LockState == LockStateKind::Unlocked
                   ? LockStateKind::UnlockedAndPossiblyDestroyed
                   : LockStateKind::UntouchedAndPossiblyDestroyed);
      return Result;
    }
  } else {
    // For non-PthreadSemantics, we assume destroy always succeeds
    if (!LockState || *LockState == LockStateKind::Unlocked) {
      Result.State =
          Result.State->set<LockStates>(MTX, LockStateKind::Destroyed);
      return Result;
    }
  }

  if (*LockState == LockStateKind::Locked) {
    Result.State =
        Result.State->set<LockStates>(MTX, LockStateKind::Error_DestroyLocked);
    return Result;
  }

  if (*LockState == LockStateKind::Destroyed) {
    Result.State =
        Result.State->set<LockStates>(MTX, LockStateKind::Error_DoubleDestroy);
    return Result;
  }

  assert(LockState && *LockState != LockStateKind::Unlocked &&
         *LockState != LockStateKind::Locked &&
         *LockState != LockStateKind::Destroyed &&
         "We can only get here if we came from an error-state to begin with");

  return Result;
}

MutexModeling::ModelingResult
MutexModeling::handleEvent(const EventDescriptor &Event, const MemRegion *MTX,
                           const CallEvent &Call, ProgramStateRef State,
                           CheckerContext &C) const {
  assert(MTX && "should only be called with a mutex region");

  State = State->add<MutexEvents>(
      EventMarker{Event.Kind, Event.Semantics, Event.Library,
                  Call.getCalleeIdentifier(), Call.getOriginExpr(), MTX});

  switch (Event.Kind) {
  case EventKind::Init:
    return handleInit(Event, MTX, Call, State, C);
  case EventKind::Acquire:
    return handleAcquire(Event, MTX, Call, State, C);
  case EventKind::TryAcquire:
    return handleTryAcquire(Event, MTX, Call, State, C);
  case EventKind::Release:
    return handleRelease(Event, MTX, Call, State, C);
  case EventKind::Destroy:
    return handleDestroy(Event, MTX, Call, State, C);
  default:
    llvm_unreachable("Unknown event kind");
  }
}

static const MemRegion *skipStdBaseClassRegion(const MemRegion *Reg) {
  while (Reg) {
    const auto *BaseClassRegion = dyn_cast<CXXBaseObjectRegion>(Reg);
    if (!BaseClassRegion || !isWithinStdNamespace(BaseClassRegion->getDecl()))
      break;
    Reg = BaseClassRegion->getSuperRegion();
  }
  return Reg;
}

void MutexModeling::checkPostCall(const CallEvent &Call,
                                  CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  for (auto &&Event : RegisteredEvents) {
    if (matches(Event.Trigger, Call)) {
      // Apply skipStdBaseClassRegion to canonicalize the mutex region
      const MemRegion *MTX =
          skipStdBaseClassRegion(getRegion(Event.Trigger, Call));
      if (!MTX)
        continue;
      State = doResolvePossiblyDestroyedMutex(State, MTX);
      ModelingResult Result = handleEvent(Event, MTX, Call, State, C);
      C.addTransition(Result.State, Result.Note);
    }
  }
}

void MutexModeling::checkDeadSymbols(SymbolReaper &SymReaper,
                                     CheckerContext &C) const {
  ProgramStateRef State = C.getState();

  for (auto I : State->get<DestroyedRetVals>()) {
    // Once the return value symbol dies, no more checks can be performed
    // against it. See if the return value was checked before this point.
    // This would remove the symbol from the map as well.
    if (SymReaper.isDead(I.second))
      State = resolvePossiblyDestroyedMutex(State, I.first, &I.second);
  }

  for (auto I : State->get<LockStates>()) {
    // Stop tracking dead mutex regions as well.
    if (!SymReaper.isLiveRegion(I.first)) {
      State = State->remove<LockStates>(I.first);
      State = State->remove<DestroyedRetVals>(I.first);
    }
  }

  // TODO: We probably need to clean up the lock stack as well.
  // It is tricky though: even if the mutex cannot be unlocked anymore,
  // it can still participate in lock order reversal resolution.

  C.addTransition(State);
}

ProgramStateRef MutexModeling::checkRegionChanges(
    ProgramStateRef State, const InvalidatedSymbols *Symbols,
    ArrayRef<const MemRegion *> ExplicitRegions,
    ArrayRef<const MemRegion *> Regions, const LocationContext *LCtx,
    const CallEvent *Call) const {

  bool IsLibraryFunction = false;
  if (Call) {
    // Avoid invalidating mutex state when a known supported function is
    // called.
    for (auto &&Event : RegisteredEvents) {
      if (matches(Event.Trigger, *Call)) {
        return State;
      }
    }

    if (Call->isInSystemHeader())
      IsLibraryFunction = true;
  }

  for (auto R : Regions) {
    // We treat system library functions differently because we assume they
    // won't modify mutex state unless the mutex is explicitly passed as an
    // argument
    if (IsLibraryFunction && !llvm::is_contained(ExplicitRegions, R))
      continue;

    State = State->remove<LockStates>(R);
    State = State->remove<DestroyedRetVals>(R);

    // TODO: We need to invalidate the lock stack as well. This is tricky
    // to implement correctly and efficiently though, because the effects
    // of mutex escapes on lock order may be fairly varied.
  }

  return State;
}

namespace clang {
namespace ento {
// Checker registration
void registerMutexModeling(CheckerManager &CM) {
  CM.registerChecker<MutexModeling>();

  // The following RegisterEvent calls set up the checker to recognize various
  // mutex-related function calls across different libraries and semantics.
  // Each RegisterEvent associates a specific function with an event type
  // (Init, Acquire, TryAcquire, Release, Destroy) and specifies how to
  // extract the mutex argument from the function call.

  // clang-format off
  // Pthread-related events
  // Init
  RegisterEvent(EventDescriptor{
    MakeFirstArgExtractor(
      {"pthread_mutex_init"}, 2),
      EventKind::Init,
      LibraryKind::Pthread
  });

  // Acquire
  RegisterEvent(EventDescriptor{
    MakeFirstArgExtractor(
      {"pthread_mutex_lock"}),
      EventKind::Acquire,
      LibraryKind::Pthread,
      SemanticsKind::PthreadSemantics
  });
  RegisterEvent(EventDescriptor{
    MakeFirstArgExtractor({"pthread_rwlock_rdlock"}),
    EventKind::Acquire,
    LibraryKind::Pthread,
    SemanticsKind::PthreadSemantics
  });
  RegisterEvent(EventDescriptor{
    MakeFirstArgExtractor({"pthread_rwlock_wrlock"}),
    EventKind::Acquire,
    LibraryKind::Pthread,
    SemanticsKind::PthreadSemantics
  });
  RegisterEvent(EventDescriptor{
    MakeFirstArgExtractor({"lck_mtx_lock"}),
    EventKind::Acquire,
    LibraryKind::Pthread,
    SemanticsKind::XNUSemantics
  });
  RegisterEvent(EventDescriptor{
    MakeFirstArgExtractor({"lck_rw_lock_exclusive"}),
    EventKind::Acquire,
    LibraryKind::Pthread,
    SemanticsKind::XNUSemantics
  });
  RegisterEvent(EventDescriptor{
    MakeFirstArgExtractor({"lck_rw_lock_shared"}),
    EventKind::Acquire,
    LibraryKind::Pthread,
    SemanticsKind::XNUSemantics
  });

  // TryAcquire
  RegisterEvent(EventDescriptor{
    MakeFirstArgExtractor({"pthread_mutex_trylock"}),
    EventKind::TryAcquire,
    LibraryKind::Pthread,
    SemanticsKind::PthreadSemantics
  });
  RegisterEvent(EventDescriptor{
    MakeFirstArgExtractor({"pthread_rwlock_tryrdlock"}),
    EventKind::TryAcquire,
    LibraryKind::Pthread,
    SemanticsKind::PthreadSemantics
  });
  RegisterEvent(EventDescriptor{
    MakeFirstArgExtractor({"pthread_rwlock_trywrlock"}),
    EventKind::TryAcquire,
    LibraryKind::Pthread,
    SemanticsKind::PthreadSemantics
  });
  RegisterEvent(EventDescriptor{
    MakeFirstArgExtractor({"lck_mtx_try_lock"}),
    EventKind::TryAcquire,
    LibraryKind::Pthread,
    SemanticsKind::XNUSemantics
  });
  RegisterEvent(EventDescriptor{
    MakeFirstArgExtractor({"lck_rw_try_lock_exclusive"}),
    EventKind::TryAcquire,
    LibraryKind::Pthread,
    SemanticsKind::XNUSemantics
  });
  RegisterEvent(EventDescriptor{
    MakeFirstArgExtractor({"lck_rw_try_lock_shared"}),
    EventKind::TryAcquire,
    LibraryKind::Pthread,
    SemanticsKind::XNUSemantics
  });

  // Release
  RegisterEvent(EventDescriptor{
    MakeFirstArgExtractor({"pthread_mutex_unlock"}),
    EventKind::Release,
    LibraryKind::Pthread
  });
  RegisterEvent(EventDescriptor{
    MakeFirstArgExtractor({"pthread_rwlock_unlock"}),
    EventKind::Release,
    LibraryKind::Pthread
  });
  RegisterEvent(EventDescriptor{
    MakeFirstArgExtractor({"lck_mtx_unlock"}),
    EventKind::Release,
    LibraryKind::Pthread
  });
  RegisterEvent(EventDescriptor{
    MakeFirstArgExtractor({"lck_rw_unlock_exclusive"}),
    EventKind::Release,
    LibraryKind::Pthread
  });
  RegisterEvent(EventDescriptor{
    MakeFirstArgExtractor({"lck_rw_unlock_shared"}),
    EventKind::Release,
    LibraryKind::Pthread
  });
  RegisterEvent(EventDescriptor{
    MakeFirstArgExtractor({"lck_rw_done"}),
    EventKind::Release,
    LibraryKind::Pthread
  });

  // Destroy
  RegisterEvent(EventDescriptor{
    MakeFirstArgExtractor({"pthread_mutex_destroy"}),
    EventKind::Destroy,
    LibraryKind::Pthread,
    SemanticsKind::PthreadSemantics
  });
  RegisterEvent(EventDescriptor{
    MakeFirstArgExtractor({"lck_mtx_destroy"}, 2),
    EventKind::Destroy,
    LibraryKind::Pthread,
    SemanticsKind::XNUSemantics
  });

  // Fuchsia-related events
  // Init
  RegisterEvent(EventDescriptor{
    MakeFirstArgExtractor({"spin_lock_init"}),
    EventKind::Init,
    LibraryKind::Fuchsia
  });

  // Acquire
  RegisterEvent(EventDescriptor{
    MakeFirstArgExtractor({"spin_lock"}),
    EventKind::Acquire,
    LibraryKind::Fuchsia,
    SemanticsKind::PthreadSemantics
  });
  RegisterEvent(EventDescriptor{
    MakeFirstArgExtractor({"spin_lock_save"}, 3),
    EventKind::Acquire,
    LibraryKind::Fuchsia,
    SemanticsKind::PthreadSemantics
  });
  RegisterEvent(EventDescriptor{
    MakeFirstArgExtractor({"sync_mutex_lock"}),
    EventKind::Acquire,
    LibraryKind::Fuchsia,
    SemanticsKind::PthreadSemantics
  });
  RegisterEvent(EventDescriptor{
    MakeFirstArgExtractor({"sync_mutex_lock_with_waiter"}),
    EventKind::Acquire,
    LibraryKind::Fuchsia,
    SemanticsKind::PthreadSemantics
  });

  // TryAcquire
  RegisterEvent(EventDescriptor{
    MakeFirstArgExtractor({"spin_trylock"}),
    EventKind::TryAcquire,
    LibraryKind::Fuchsia,
    SemanticsKind::PthreadSemantics
  });
  RegisterEvent(EventDescriptor{
    MakeFirstArgExtractor({"sync_mutex_trylock"}),
    EventKind::TryAcquire,
    LibraryKind::Fuchsia,
    SemanticsKind::PthreadSemantics
  });
  RegisterEvent(EventDescriptor{
    MakeFirstArgExtractor({"sync_mutex_timedlock"}, 2),
    EventKind::TryAcquire,
    LibraryKind::Fuchsia,
    SemanticsKind::PthreadSemantics
  });

  // Release
  RegisterEvent(EventDescriptor{
    MakeFirstArgExtractor({"spin_unlock"}),
    EventKind::Release,
    LibraryKind::Fuchsia
  });
  RegisterEvent(EventDescriptor{
    MakeFirstArgExtractor({"spin_unlock_restore"}, 3),
    EventKind::Release,
    LibraryKind::Fuchsia
  });
  RegisterEvent(EventDescriptor{
    MakeFirstArgExtractor({"sync_mutex_unlock"}),
    EventKind::Release,
    LibraryKind::Fuchsia
  });

  // C11-related events
  // Init
  RegisterEvent(EventDescriptor{
    MakeFirstArgExtractor({"mtx_init"}, 2),
    EventKind::Init,
    LibraryKind::C11
  });

  // Acquire
  RegisterEvent(EventDescriptor{
    MakeFirstArgExtractor({"mtx_lock"}),
    EventKind::Acquire,
    LibraryKind::C11,
    SemanticsKind::PthreadSemantics
  });

  // TryAcquire
  RegisterEvent(EventDescriptor{
    MakeFirstArgExtractor({"mtx_trylock"}),
    EventKind::TryAcquire,
    LibraryKind::C11,
    SemanticsKind::PthreadSemantics
  });
  RegisterEvent(EventDescriptor{
    MakeFirstArgExtractor({"mtx_timedlock"}, 2),
    EventKind::TryAcquire,
    LibraryKind::C11,
    SemanticsKind::PthreadSemantics
  });

  // Release
  RegisterEvent(EventDescriptor{
    MakeFirstArgExtractor({"mtx_unlock"}),
    EventKind::Release,
    LibraryKind::C11
  });

  // Destroy
  RegisterEvent(EventDescriptor{
    MakeFirstArgExtractor({"mtx_destroy"}),
    EventKind::Destroy,
    LibraryKind::C11,
    SemanticsKind::PthreadSemantics
  });
  // clang-format on
}
bool shouldRegisterMutexModeling(const CheckerManager &) { return true; }
} // namespace ento
} // namespace clang

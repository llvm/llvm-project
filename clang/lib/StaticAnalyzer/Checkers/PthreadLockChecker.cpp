//===--- PthreadLockChecker.cpp - Check for locking problems ---*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines:
//  * PthreadLockChecker, a simple lock -> unlock checker.
//    Which also checks for XNU locks, which behave similarly enough to share
//    code.
//  * FuchsiaLocksChecker, which is also rather similar.
//  * C11LockChecker which also closely follows Pthread semantics.
//
//  TODO: Path notes.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"

#include "MutexModeling/MutexModelingAPI.h"

using namespace clang;
using namespace ento;
using namespace mutex_modeling;

namespace {

class PthreadLockChecker : public Checker<check::PostCall> {
public:
  // Enum to represent different types of lock checkers
  enum CheckerKind {
    CK_PthreadLockChecker,
    CK_FuchsiaLockChecker,
    CK_C11LockChecker,
    CK_NumCheckKinds
  };

  bool ChecksEnabled[CK_NumCheckKinds] = {false};
  CheckerNameRef CheckNames[CK_NumCheckKinds];

private:
  void reportBug(CheckerContext &C, std::unique_ptr<BugType> BT[],
                 const Expr *MtxExpr, CheckerKind CheckKind,
                 StringRef Desc) const;

public:
  void checkPostCall(const CallEvent &Call, CheckerContext &C) const;
  void printState(raw_ostream &Out, ProgramStateRef State, const char *NL,
                  const char *Sep) const override;

private:
  // Bug types for different lock-related issues
  mutable std::unique_ptr<BugType> BT_doublelock[CK_NumCheckKinds];
  mutable std::unique_ptr<BugType> BT_doubleunlock[CK_NumCheckKinds];
  mutable std::unique_ptr<BugType> BT_destroylock[CK_NumCheckKinds];
  mutable std::unique_ptr<BugType> BT_initlock[CK_NumCheckKinds];
  mutable std::unique_ptr<BugType> BT_initlockPthread;
  mutable std::unique_ptr<BugType> BT_lor[CK_NumCheckKinds];

  // Initialize bug types for a specific checker kind
  void initBugType(CheckerKind CheckKind) const {
    if (BT_doublelock[CheckKind])
      return;
    BT_doublelock[CheckKind].reset(
        new BugType{CheckNames[CheckKind], "Double locking", "Lock checker"});
    BT_doubleunlock[CheckKind].reset(
        new BugType{CheckNames[CheckKind], "Double unlocking", "Lock checker"});
    BT_destroylock[CheckKind].reset(new BugType{
        CheckNames[CheckKind], "Use destroyed lock", "Lock checker"});
    BT_initlock[CheckKind].reset(new BugType{
        CheckNames[CheckKind], "Init invalid lock", "Lock checker"});
    BT_lor[CheckKind].reset(new BugType{CheckNames[CheckKind],
                                        "Lock order reversal", "Lock checker"});
  }

  // Detect which checker kind should be used based on the event
  [[nodiscard]] constexpr std::optional<PthreadLockChecker::CheckerKind>
  detectCheckerKind(mutex_modeling::EventMarker EV) const noexcept {
    switch (EV.Library) {
    default:
      return std::nullopt;
    case mutex_modeling::LibraryKind::Pthread:
      return {PthreadLockChecker::CK_PthreadLockChecker};
    case mutex_modeling::LibraryKind::Fuchsia:
      return {PthreadLockChecker::CK_FuchsiaLockChecker};
    case mutex_modeling::LibraryKind::C11:
      return {PthreadLockChecker::CK_C11LockChecker};
    }
  }

  // Check methods for different types of lock events
  void checkInitEvent(const EventMarker &LastEvent, CheckerKind Checker,
                      CheckerContext &C) const;
  void checkAcquireEvent(const EventMarker &LastEvent, CheckerKind Checker,
                         CheckerContext &C) const;
  void checkReleaseEvent(const EventMarker &LastEvent, CheckerKind Checker,
                         CheckerContext &C) const;
  void checkDestroyEvent(const EventMarker &LastEvent, CheckerKind Checker,
                         CheckerContext &C) const;
};
} // end anonymous namespace

void PthreadLockChecker::checkInitEvent(const EventMarker &LastEvent,
                                        CheckerKind Checker,
                                        CheckerContext &C) const {
  ProgramStateRef State = C.getState();

  const LockStateKind *const LockState =
      State->get<LockStates>(LastEvent.MutexRegion);

  if (!LockState) {
    return;
  }

  if (*LockState == LockStateKind::Error_DoubleInit) {
    reportBug(C, BT_initlock, LastEvent.EventExpr, Checker,
              "This lock has already been initialized");
  } else if (*LockState == LockStateKind::Error_DoubleInitWhileLocked) {
    reportBug(C, BT_initlock, LastEvent.EventExpr, Checker,
              "This lock is still being held");
  }
}

void PthreadLockChecker::checkAcquireEvent(const EventMarker &LastEvent,
                                           CheckerKind Checker,
                                           CheckerContext &C) const {
  ProgramStateRef State = C.getState();

  const LockStateKind *const LockState =
      State->get<LockStates>(LastEvent.MutexRegion);

  if (!LockState)
    return;

  if (*LockState == LockStateKind::Error_DoubleLock) {
    reportBug(C, BT_doublelock, LastEvent.EventExpr, Checker,
              "This lock has already been acquired");
  } else if (*LockState == LockStateKind::Error_LockDestroyed) {
    reportBug(C, BT_destroylock, LastEvent.EventExpr, Checker,
              "This lock has already been destroyed");
  }
}

void PthreadLockChecker::checkReleaseEvent(const EventMarker &LastEvent,
                                           CheckerKind Checker,
                                           CheckerContext &C) const {
  ProgramStateRef State = C.getState();

  const LockStateKind *const LockState =
      State->get<LockStates>(LastEvent.MutexRegion);

  if (!LockState)
    return;

  if (*LockState == LockStateKind::Error_DoubleUnlock) {
    reportBug(C, BT_doubleunlock, LastEvent.EventExpr, Checker,
              "This lock has already been unlocked");
  } else if (*LockState == LockStateKind::Error_UnlockDestroyed) {
    reportBug(C, BT_destroylock, LastEvent.EventExpr, Checker,
              "This lock has already been destroyed");
  } else if (*LockState == LockStateKind::Error_LockReversal) {
    reportBug(C, BT_lor, LastEvent.EventExpr, Checker,
              "This was not the most recently acquired lock. Possible lock "
              "order reversal");
  }
}

void PthreadLockChecker::checkDestroyEvent(const EventMarker &LastEvent,
                                           CheckerKind Checker,
                                           CheckerContext &C) const {
  ProgramStateRef State = C.getState();

  const LockStateKind *const LockState =
      State->get<LockStates>(LastEvent.MutexRegion);

  if (!LockState || *LockState == LockStateKind::Destroyed)
    return;

  if (*LockState == LockStateKind::Error_DestroyLocked) {
    reportBug(C, BT_destroylock, LastEvent.EventExpr, Checker,
              "This lock is still locked");
  } else if (*LockState == LockStateKind::Error_DoubleDestroy)
    reportBug(C, BT_destroylock, LastEvent.EventExpr, Checker,
              "This lock has already been destroyed");
}

void PthreadLockChecker::checkPostCall(const CallEvent &Call,
                                       CheckerContext &C) const {

  ProgramStateRef State = C.getState();

  const auto &MTXEvents = State->get<MutexEvents>();

  if (MTXEvents.isEmpty())
    return;

  const auto &LastEvent = MTXEvents.getHead();

  std::optional<CheckerKind> Checker = detectCheckerKind(LastEvent);

  if (!Checker || !ChecksEnabled[*Checker])
    return;

  switch (LastEvent.Kind) {
  case EventKind::Init:
    checkInitEvent(LastEvent, *Checker, C);
    break;
  case EventKind::Acquire:
  case EventKind::TryAcquire:
    checkAcquireEvent(LastEvent, *Checker, C);
    break;
  case EventKind::Release:
    checkReleaseEvent(LastEvent, *Checker, C);
    break;
  case EventKind::Destroy:
    checkDestroyEvent(LastEvent, *Checker, C);
    break;
  }
}

void PthreadLockChecker::reportBug(CheckerContext &C,
                                   std::unique_ptr<BugType> BT[],
                                   const Expr *MtxExpr, CheckerKind CheckKind,
                                   StringRef Desc) const {
  ExplodedNode *N = C.generateErrorNode();
  if (!N)
    return;
  initBugType(CheckKind);
  auto Report =
      std::make_unique<PathSensitiveBugReport>(*BT[CheckKind], Desc, N);
  Report->addRange(MtxExpr->getSourceRange());
  C.emitReport(std::move(Report));
}

void PthreadLockChecker::printState(raw_ostream &Out, ProgramStateRef State,
                                    const char *NL, const char *Sep) const {
  mutex_modeling::printState(Out, State, NL, Sep);
}

void ento::registerPthreadLockBase(CheckerManager &mgr) {
  mgr.registerChecker<PthreadLockChecker>();
}

bool ento::shouldRegisterPthreadLockBase(const CheckerManager &mgr) {
  return true;
}

void ento::registerPthreadLockChecker(CheckerManager &CM) {
  PthreadLockChecker *ImplChecker = CM.getChecker<PthreadLockChecker>();
  ImplChecker->ChecksEnabled[PthreadLockChecker::CK_PthreadLockChecker] = true;
  ImplChecker->CheckNames[PthreadLockChecker::CK_PthreadLockChecker] =
      CM.getCurrentCheckerName();
}

bool ento::shouldRegisterPthreadLockChecker(const CheckerManager &CM) {
  return true;
}

void ento::registerFuchsiaLockChecker(CheckerManager &CM) {
  PthreadLockChecker *ImplChecker = CM.getChecker<PthreadLockChecker>();
  ImplChecker->ChecksEnabled[PthreadLockChecker::CK_FuchsiaLockChecker] = true;
  ImplChecker->CheckNames[PthreadLockChecker::CK_FuchsiaLockChecker] =
      CM.getCurrentCheckerName();
}

bool ento::shouldRegisterFuchsiaLockChecker(const CheckerManager &CM) {
  return true;
}

void ento::registerC11LockChecker(CheckerManager &CM) {
  PthreadLockChecker *ImplChecker = CM.getChecker<PthreadLockChecker>();
  ImplChecker->ChecksEnabled[PthreadLockChecker::CK_C11LockChecker] = true;
  ImplChecker->CheckNames[PthreadLockChecker::CK_C11LockChecker] =
      CM.getCurrentCheckerName();
}

bool ento::shouldRegisterC11LockChecker(const CheckerManager &CM) {
  return true;
}

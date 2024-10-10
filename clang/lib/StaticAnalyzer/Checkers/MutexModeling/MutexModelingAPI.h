//===--- MutexModelingAPI.h - API for modeling mutexes --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines inter-checker API for tracking and manipulating the
// modeled state of locked mutexes in the GDM.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_STATICANALYZER_CHECKERS_MUTEXMODELINGAPI_H
#define LLVM_CLANG_LIB_STATICANALYZER_CHECKERS_MUTEXMODELINGAPI_H

#include "MutexModelingDomain.h"
#include "MutexModelingGDM.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"

namespace clang {

namespace ento {
class BugType;
namespace mutex_modeling {

// Set of registered bug types for mutex modeling
inline llvm::SmallSet<const BugType *, 0> RegisteredBugTypes{};

// Register a bug type for mutex modeling
inline void RegisterBugTypeForMutexModeling(const BugType *BT) {
  RegisteredBugTypes.insert(BT);
}

// Check if a bug type is registered for mutex modeling
inline bool IsBugTypeRegisteredForMutexModeling(const BugType *BT) {
  return RegisteredBugTypes.contains(BT);
}

// Vector of registered event descriptors
inline llvm::SmallVector<EventDescriptor, 0> RegisteredEvents{};

// Register an event descriptor
inline auto RegisterEvent(EventDescriptor Event) {
  RegisteredEvents.push_back(Event);
}

// Helper functions to create common types of mutex extractors
inline auto
MakeFirstArgExtractor(ArrayRef<StringRef> NameParts, int NumArgsRequired = 1,
                      CallDescription::Mode MatchAs = CDM::CLibrary) {
  return FirstArgMutexExtractor{
      CallDescription{MatchAs, NameParts, NumArgsRequired}};
}

inline auto
MakeMemberExtractor(ArrayRef<StringRef> NameParts, int NumArgsRequired = 0,
                    CallDescription::Mode MatchAs = CDM::CXXMethod) {
  return MemberMutexExtractor{
      CallDescription{MatchAs, NameParts, NumArgsRequired}};
}

inline auto MakeRAIILockExtractor(StringRef GuardObjectName) {
  return RAIILockExtractor{GuardObjectName};
}

inline auto MakeRAIIReleaseExtractor(StringRef GuardObjectName) {
  return RAIIReleaseExtractor{GuardObjectName};
}

// Check if any critical sections are currently active
inline bool AreAnyCritsectionsActive(CheckerContext &C) {
  return !C.getState()->get<CritSections>().isEmpty();
}

// Create a note tag for a mutex critical section
inline const NoteTag *CreateMutexCritSectionNote(CritSectionMarker M,
                                                 CheckerContext &C) {
  return C.getNoteTag([M](const PathSensitiveBugReport &BR,
                          llvm::raw_ostream &OS) {
    if (!IsBugTypeRegisteredForMutexModeling(&BR.getBugType()))
      return;
    const auto CritSectionBegins =
        BR.getErrorNode()->getState()->get<CritSections>();
    llvm::SmallVector<CritSectionMarker, 4> LocksForMutex;
    llvm::copy_if(CritSectionBegins, std::back_inserter(LocksForMutex),
                  [M](const auto &Marker) {
                    return Marker.MutexRegion == M.MutexRegion;
                  });
    if (LocksForMutex.empty())
      return;

    // As the ImmutableList builds the locks by prepending them, we
    // reverse the list to get the correct order.
    std::reverse(LocksForMutex.begin(), LocksForMutex.end());

    // Find the index of the lock expression in the list of all locks for a
    // given mutex (in acquisition order).
    const CritSectionMarker *const Position =
        llvm::find_if(std::as_const(LocksForMutex), [M](const auto &Marker) {
          return Marker.BeginExpr == M.BeginExpr;
        });
    if (Position == LocksForMutex.end())
      return;

    // If there is only one lock event, we don't need to specify how many times
    // the critical section was entered.
    if (LocksForMutex.size() == 1) {
      OS << "Entering critical section here";
      return;
    }

    const auto IndexOfLock =
        std::distance(std::as_const(LocksForMutex).begin(), Position);

    const auto OrdinalOfLock = IndexOfLock + 1;
    OS << "Entering critical section for the " << OrdinalOfLock
       << llvm::getOrdinalSuffix(OrdinalOfLock) << " time here";
  });
}

// Print the current state of mutex events, lock states, and destroyed return
// values
inline void printState(raw_ostream &Out, ProgramStateRef State, const char *NL,
                       const char *Sep) {

  const MutexEventsTy &ME = State->get<MutexEvents>();
  if (!ME.isEmpty()) {
    Out << Sep << "Mutex event:" << NL;
    for (auto I : ME) {
      Out << Sep << "Kind: " << ": ";
      switch (I.Kind) {
      case (EventKind::Init):
        Out << "Init";
        break;
      case (EventKind::Acquire):
        Out << "Acquire";
        break;
      case (EventKind::TryAcquire):
        Out << "TryAcquire";
        break;
      case (EventKind::Release):
        Out << "Release";
        break;
      case (EventKind::Destroy):
        Out << "Destroy";
        break;
      }
      Out << NL;
      Out << Sep << "Semantics: ";
      switch (I.Semantics) {
      case (SemanticsKind::NotApplicable):
        Out << "NotApplicable";
        break;
      case (SemanticsKind::PthreadSemantics):
        Out << "PthreadSemantics";
        break;
      case (SemanticsKind::XNUSemantics):
        Out << "XNUSemantics";
        break;
      }
      Out << NL;
      Out << Sep << "Library: ";
      switch (I.Library) {
      case (LibraryKind::NotApplicable):
        Out << "NotApplicable";
        break;
      case (LibraryKind::Pthread):
        Out << "Pthread";
        break;
      case (LibraryKind::Fuchsia):
        Out << "Fuchsia";
        break;
      case (LibraryKind::C11):
        Out << "C11";
        break;
      default:
        llvm_unreachable("Unknown library");
      }
      Out << NL;

      // Omit MutexExpr and EventExpr

      Out << Sep << "Mutex region: ";
      I.MutexRegion->dumpToStream(Out);
      Out << NL;
    }

    const LockStatesTy &LM = State->get<LockStates>();
    if (!LM.isEmpty()) {
      Out << Sep << "Mutex states:" << NL;
      for (auto I : LM) {
        I.first->dumpToStream(Out);
        switch (I.second) {
        case (LockStateKind::Locked):
          Out << ": locked";
          break;
        case (LockStateKind::Unlocked):
          Out << ": unlocked";
          break;
        case (LockStateKind::Destroyed):
          Out << ": destroyed";
          break;
        case (LockStateKind::UntouchedAndPossiblyDestroyed):
          Out << ": not tracked, possibly destroyed";
          break;
        case (LockStateKind::UnlockedAndPossiblyDestroyed):
          Out << ": unlocked, possibly destroyed";
          break;
        case (LockStateKind::Error_DoubleInit):
          Out << ": error: double init";
          break;
        case (LockStateKind::Error_DoubleInitWhileLocked):
          Out << ": error: double init while locked";
          break;
        default:
          llvm_unreachable("Unknown lock state");
        }
        Out << NL;
      }
    }

    const DestroyedRetValsTy &DRV = State->get<DestroyedRetVals>();
    if (!DRV.isEmpty()) {
      Out << Sep << "Mutexes in unresolved possibly destroyed state:" << NL;
      for (auto I : DRV) {
        I.first->dumpToStream(Out);
        Out << ": ";
        I.second->dumpToStream(Out);
        Out << NL;
      }
    }
  }
}

} // namespace mutex_modeling
} // namespace ento
} // namespace clang

#endif

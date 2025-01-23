//===--- MutexModelingDomain.h - Common vocabulary for modeling mutexes ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines common types and related functions used in the mutex modeling domain.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_STATICANALYZER_CHECKERS_MUTEXMODELINGDOMAIN_H
#define LLVM_CLANG_LIB_STATICANALYZER_CHECKERS_MUTEXMODELINGDOMAIN_H

#include "MutexRegionExtractor.h"

// Forwad declarations.
namespace clang {
class Expr;
class IdentifierInfo;
namespace ento {
class MemRegion;
} // namespace ento
} // namespace clang

namespace clang::ento::mutex_modeling {

enum class EventKind { Init, Acquire, TryAcquire, Release, Destroy };

// TODO: Ideally the modeling should not know about which checkers consume the
// modeling information. This enum is here to make a correspondence between the
// checked mutex event the library that event came from. In order to keep the
// external API of multiple distinct checkers (PthreadLockChecker,
// FuchsiaLockChecker and C11LockChecker), this mapping is done here, but if
// more consumers of this modeling arise, adding all of them here may not be
// feasible and we may need to make this modeling more flexible.
enum class LibraryKind { NotApplicable = 0, Pthread, Fuchsia, C11 };

enum class SemanticsKind { NotApplicable = 0, PthreadSemantics, XNUSemantics };

enum class LockStateKind {
  Unlocked,
  Locked,
  Destroyed,
  UntouchedAndPossiblyDestroyed,
  UnlockedAndPossiblyDestroyed,
  Error_DoubleInit,
  Error_DoubleInitWhileLocked,
  Error_DoubleLock,
  Error_LockDestroyed,
  Error_DoubleUnlock,
  Error_UnlockDestroyed,
  Error_LockReversal,
  Error_DestroyLocked,
  Error_DoubleDestroy
};

/// This class is intended for describing the list of events to detect.
/// This list of events is the configuration of the MutexModeling checker.
struct EventDescriptor {
  MutexRegionExtractor Trigger;
  EventKind Kind{};
  LibraryKind Library{};
  SemanticsKind Semantics{};

  // TODO: Modernize to spaceship when C++20 is available.
  [[nodiscard]] bool operator!=(const EventDescriptor &Other) const noexcept {
    return !(Trigger == Other.Trigger) || Library != Other.Library ||
           Kind != Other.Kind || Semantics != Other.Semantics;
  }
  [[nodiscard]] bool operator==(const EventDescriptor &Other) const noexcept {
    return !(*this != Other);
  }
};

/// This class is used in the GDM to describe the events that were detected.
/// As instances of this class can appear many times in the ExplodedGraph, it
/// best to keep it as simple and small as possible.
struct EventMarker {
  EventKind Kind{};
  SemanticsKind Semantics{};
  LibraryKind Library{};
  const IdentifierInfo *EventII;
  const clang::Expr *EventExpr{};
  const clang::ento::MemRegion *MutexRegion{};

  // TODO: Modernize to spaceship when C++20 is available.
  [[nodiscard]] constexpr bool
  operator!=(const EventMarker &Other) const noexcept {
    return EventII != Other.EventII || Kind != Other.Kind ||
           Semantics != Other.Semantics || Library != Other.Library ||
           EventExpr != Other.EventExpr || MutexRegion != Other.MutexRegion;
  }
  [[nodiscard]] constexpr bool
  operator==(const EventMarker &Other) const noexcept {
    return !(*this != Other);
  }
};

struct CritSectionMarker {
  const clang::Expr *BeginExpr;
  const clang::ento::MemRegion *MutexRegion;

  explicit CritSectionMarker(const clang::Expr *BeginExpr,
                             const clang::ento::MemRegion *MutexRegion)
      : BeginExpr(BeginExpr), MutexRegion(MutexRegion) {}

  // TODO: Modernize to spaceship when C++20 is available.
  [[nodiscard]] constexpr bool
  operator!=(const CritSectionMarker &Other) const noexcept {
    return BeginExpr != Other.BeginExpr || MutexRegion != Other.MutexRegion;
  }
  [[nodiscard]] constexpr bool
  operator==(const CritSectionMarker &Other) const noexcept {
    return !(*this != Other);
  }
};

} // namespace clang::ento::mutex_modeling

#endif

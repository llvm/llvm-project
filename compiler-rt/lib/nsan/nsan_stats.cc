//===-- nsan_stats.cc -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of NumericalStabilitySanitizer.
//
// NumericalStabilitySanitizer statistics.
//===----------------------------------------------------------------------===//

#include "nsan/nsan_stats.h"

#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_placement_new.h"
#include "sanitizer_common/sanitizer_stackdepot.h"
#include "sanitizer_common/sanitizer_stacktrace.h"
#include "sanitizer_common/sanitizer_symbolizer.h"

#include <assert.h>
#include <stdio.h>

namespace __nsan {

using namespace __sanitizer;

Stats::Stats() {
  CheckAndWarnings.Initialize(0);
  TrackedLoads.Initialize(0);
}

Stats::~Stats() { Printf("deleting nsan stats\n"); }

static uptr key(CheckTypeT CheckType, u32 StackId) {
  return static_cast<uptr>(CheckType) +
         StackId * static_cast<uptr>(CheckTypeT::kMaxCheckType);
}

template <typename MapT, typename VectorT, typename Fn>
void UpdateEntry(CheckTypeT CheckTy, uptr PC, uptr BP, MapT *Map,
                 VectorT *Vector, Mutex *Mutex, Fn F) {
  BufferedStackTrace Stack;
  Stack.Unwind(PC, BP, nullptr, false);
  u32 StackId = StackDepotPut(Stack);
  typename MapT::Handle Handle(Map, key(CheckTy, StackId));
  Lock L(Mutex);
  if (Handle.created()) {
    typename VectorT::value_type Entry;
    Entry.StackId = StackId;
    Entry.CheckTy = CheckTy;
    F(Entry);
    Vector->push_back(Entry);
  } else {
    auto &Entry = (*Vector)[*Handle];
    F(Entry);
  }
}

void Stats::addCheck(CheckTypeT CheckTy, uptr PC, uptr BP, double RelErr) {
  UpdateEntry(CheckTy, PC, BP, &CheckAndWarningsMap, &CheckAndWarnings,
              &CheckAndWarningsMutex, [RelErr](CheckAndWarningsValue &Entry) {
                ++Entry.NumChecks;
                if (RelErr > Entry.MaxRelativeError) {
                  Entry.MaxRelativeError = RelErr;
                }
              });
}

void Stats::addWarning(CheckTypeT CheckTy, uptr PC, uptr BP, double RelErr) {
  UpdateEntry(CheckTy, PC, BP, &CheckAndWarningsMap, &CheckAndWarnings,
              &CheckAndWarningsMutex, [RelErr](CheckAndWarningsValue &Entry) {
                ++Entry.NumWarnings;
                if (RelErr > Entry.MaxRelativeError) {
                  Entry.MaxRelativeError = RelErr;
                }
              });
}

void Stats::addInvalidLoadTrackingEvent(uptr PC, uptr BP) {
  UpdateEntry(CheckTypeT::kLoad, PC, BP, &LoadTrackingMap, &TrackedLoads,
              &TrackedLoadsMutex,
              [](LoadTrackingValue &Entry) { ++Entry.NumInvalid; });
}

void Stats::addUnknownLoadTrackingEvent(uptr PC, uptr BP) {
  UpdateEntry(CheckTypeT::kLoad, PC, BP, &LoadTrackingMap, &TrackedLoads,
              &TrackedLoadsMutex,
              [](LoadTrackingValue &Entry) { ++Entry.NumUnknown; });
}

static const char *CheckTypeDisplay(CheckTypeT CheckType) {
  switch (CheckType) {
  case CheckTypeT::kUnknown:
    return "unknown";
  case CheckTypeT::kRet:
    return "return";
  case CheckTypeT::kArg:
    return "argument";
  case CheckTypeT::kLoad:
    return "load";
  case CheckTypeT::kStore:
    return "store";
  case CheckTypeT::kInsert:
    return "vector insert";
  case CheckTypeT::kUser:
    return "user-initiated";
  case CheckTypeT::kFcmp:
    return "fcmp";
  case CheckTypeT::kMaxCheckType:
    return "[max]";
  }
  assert(false && "unknown CheckType case");
  return "";
}

void Stats::print() const {
  {
    Lock L(&CheckAndWarningsMutex);
    for (const auto &Entry : CheckAndWarnings) {
      Printf("warned %llu times out of %llu %s checks ", Entry.NumWarnings,
             Entry.NumChecks, CheckTypeDisplay(Entry.CheckTy));
      if (Entry.NumWarnings > 0) {
        char RelErrBuf[64];
        snprintf(RelErrBuf, sizeof(RelErrBuf) - 1, "%f",
                 Entry.MaxRelativeError * 100.0);
        Printf("(max relative error: %s%%) ", RelErrBuf);
      }
      Printf("at:\n");
      StackDepotGet(Entry.StackId).Print();
    }
  }

  {
    Lock L(&TrackedLoadsMutex);
    u64 TotalInvalidLoadTracking = 0;
    u64 TotalUnknownLoadTracking = 0;
    for (const auto &Entry : TrackedLoads) {
      TotalInvalidLoadTracking += Entry.NumInvalid;
      TotalUnknownLoadTracking += Entry.NumUnknown;
      Printf("invalid/unknown type for %llu/%llu loads at:\n", Entry.NumInvalid,
             Entry.NumUnknown);
      StackDepotGet(Entry.StackId).Print();
    }
    Printf(
        "There were %llu/%llu floating-point loads where the shadow type was "
        "invalid/unknown.\n",
        TotalInvalidLoadTracking, TotalUnknownLoadTracking);
  }
}

ALIGNED(64) static char StatsPlaceholder[sizeof(Stats)];
Stats *nsan_stats = nullptr;

void initializeStats() { nsan_stats = new (StatsPlaceholder) Stats(); }

} // namespace __nsan

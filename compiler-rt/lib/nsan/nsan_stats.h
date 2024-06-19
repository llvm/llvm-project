//===-- nsan_stats.h --------------------------------------------*- C++- *-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of NumericalStabilitySanitizer.
//
// NSan statistics. This class counts the number of checks per code location,
// and is used to output statistics (typically when using
// `disable_warnings=1,enable_check_stats=1,enable_warning_stats=1`).
//===----------------------------------------------------------------------===//

#ifndef NSAN_STATS_H
#define NSAN_STATS_H

#include "sanitizer_common/sanitizer_addrhashmap.h"
#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_mutex.h"

namespace __nsan {

enum class CheckTypeT {
  kUnknown = 0,
  kRet,
  kArg,
  kLoad,
  kStore,
  kInsert,
  kUser, // User initiated.
  kFcmp,
  kMaxCheckType,
};

class Stats {
public:
  Stats();
  ~Stats();

  // Signal that we checked the instruction at the given address.
  void addCheck(CheckTypeT CheckType, __sanitizer::uptr PC,
                __sanitizer::uptr BP, double RelErr);
  // Signal that we warned for the instruction at the given address.
  void addWarning(CheckTypeT CheckType, __sanitizer::uptr PC,
                  __sanitizer::uptr BP, double RelErr);

  // Signal that we detected a floating-point load where the shadow type was
  // invalid.
  void addInvalidLoadTrackingEvent(__sanitizer::uptr PC, __sanitizer::uptr BP);
  // Signal that we detected a floating-point load where the shadow type was
  // unknown but the value was nonzero.
  void addUnknownLoadTrackingEvent(__sanitizer::uptr PC, __sanitizer::uptr BP);

  void print() const;

private:
  using IndexMap = __sanitizer::AddrHashMap<__sanitizer::uptr, 11>;

  struct CheckAndWarningsValue {
    CheckTypeT CheckTy;
    __sanitizer::u32 StackId = 0;
    __sanitizer::u64 NumChecks = 0;
    __sanitizer::u64 NumWarnings = 0;
    // This is a bitcasted double. Doubles have the nice idea to be ordered as
    // ints.
    double MaxRelativeError = 0;
  };
  // Maps key(CheckType, StackId) to indices in CheckAndWarnings.
  IndexMap CheckAndWarningsMap;
  __sanitizer::InternalMmapVectorNoCtor<CheckAndWarningsValue> CheckAndWarnings;
  mutable __sanitizer::Mutex CheckAndWarningsMutex;

  struct LoadTrackingValue {
    CheckTypeT CheckTy;
    __sanitizer::u32 StackId = 0;
    __sanitizer::u64 NumInvalid = 0;
    __sanitizer::u64 NumUnknown = 0;
  };
  // Maps key(CheckTypeT::kLoad, StackId) to indices in TrackedLoads.
  IndexMap LoadTrackingMap;
  __sanitizer::InternalMmapVectorNoCtor<LoadTrackingValue> TrackedLoads;
  mutable __sanitizer::Mutex TrackedLoadsMutex;
};

extern Stats *nsan_stats;
void initializeStats();

} // namespace __nsan

#endif // NSAN_STATS_H

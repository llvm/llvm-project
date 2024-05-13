//===--- timeout linux implementation (type-only) ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_TIME_LINUX_TIMEOUT_BASIC_H
#define LLVM_LIBC_SRC___SUPPORT_TIME_LINUX_TIMEOUT_BASIC_H

#include "hdr/time_macros.h"
#include "hdr/types/struct_timespec.h"
#include "src/__support/CPP/expected.h"
#include "src/__support/time/units.h"

namespace LIBC_NAMESPACE {
namespace internal {
// We use AbsTimeout to remind ourselves that the timeout is an absolute time.
// This is a simple wrapper around the timespec struct that also keeps track of
// whether the time is in realtime or monotonic time.

// This struct is going to be used in timed locks. Pthread generally uses
// realtime clocks for timeouts. However, due to non-monotoncity, realtime
// clocks reportedly lead to undesired behaviors. Therefore, we also provide a
// method to convert the timespec to a monotonic clock relative to the time of
// function call.
class AbsTimeout {
  timespec timeout;
  bool realtime_flag;
  LIBC_INLINE constexpr explicit AbsTimeout(timespec ts, bool realtime)
      : timeout(ts), realtime_flag(realtime) {}

public:
  enum class Error { Invalid, BeforeEpoch };
  LIBC_INLINE const timespec &get_timespec() const { return timeout; }
  LIBC_INLINE bool is_realtime() const { return realtime_flag; }
  LIBC_INLINE static constexpr cpp::expected<AbsTimeout, Error>
  from_timespec(timespec ts, bool realtime) {
    using namespace time_units;
    if (ts.tv_nsec < 0 || ts.tv_nsec >= 1_s_ns)
      return cpp::unexpected<Error>(Error::Invalid);

    // POSIX allows tv_sec to be negative. We interpret this as an expired
    // timeout.
    if (ts.tv_sec < 0)
      return cpp::unexpected<Error>(Error::BeforeEpoch);

    return AbsTimeout{ts, realtime};
  }
  // The implementation of this function is separated to timeout.h.
  // This function pulls in the dependency to clock_conversion.h,
  // which may transitively depend on vDSO hence futex. However, this structure
  // would be passed to futex, so we need to avoid cyclic dependencies.
  LIBC_INLINE void ensure_monotonic();
};
} // namespace internal
} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC___SUPPORT_TIME_LINUX_TIMEOUT_BASIC_H

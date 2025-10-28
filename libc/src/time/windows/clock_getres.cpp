//===-- Windows implementation of clock_getres ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/time_macros.h"
#include "hdr/types/clockid_t.h"
#include "hdr/types/struct_timespec.h"

#include "src/__support/CPP/limits.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/time/units.h"
#include "src/__support/time/windows/performance_counter.h"
#include "src/time/clock_getres.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>

// add in dependencies for GetSystemTimeAdjustmentPrecise
#pragma comment(lib, "mincore.lib")

namespace LIBC_NAMESPACE_DECL {
LLVM_LIBC_FUNCTION(int, clock_getres, (clockid_t id, struct timespec *res)) {
  using namespace time_units;
  // POSIX allows nullptr to be passed as res, in which case the function should
  // do nothing.
  if (res == nullptr)
    return 0;
  constexpr unsigned long long HNS_PER_SEC = 1_s_ns / 100ULL;
  constexpr unsigned long long SEC_LIMIT =
      cpp::numeric_limits<decltype(res->tv_sec)>::max();
  // For CLOCK_MONOTONIC, we are using performance counter
  // https://learn.microsoft.com/en-us/windows/win32/sysinfo/acquiring-high-resolution-time-stamps
  // Hence, the resolution is given by the performance counter frequency.
  // For CLOCK_REALTIME, the precision is given by
  // GetSystemTimeAdjustmentPrecise
  // (https://learn.microsoft.com/en-us/windows/win32/api/sysinfoapi/nf-sysinfoapi-getsystemtimeadjustmentprecise)
  // For CLOCK_PROCESS_CPUTIME_ID, CLOCK_THREAD_CPUTIME_ID, the precision is
  // given by GetSystemTimeAdjustment
  // (https://learn.microsoft.com/en-us/windows/win32/api/sysinfoapi/nf-sysinfoapi-getsystemtimeadjustment)
  switch (id) {
  default:
    libc_errno = EINVAL;
    return -1;

  case CLOCK_MONOTONIC: {
    long long freq = performance_counter::get_ticks_per_second();
    __builtin_assume(freq != 0);
    // division of 1 second by frequency, rounded up.
    long long tv_sec = static_cast<long long>(freq == 1);
    long long tv_nsec =
        LIBC_LIKELY(freq != 1) ? 1ll + ((1_s_ns - 1ll) / freq) : 0ll;
    // not possible to overflow tv_sec, tv_nsec
    res->tv_sec = static_cast<decltype(res->tv_sec)>(tv_sec);
    res->tv_nsec = static_cast<decltype(res->tv_nsec)>(tv_nsec);
    break;
  }

  case CLOCK_REALTIME: {
    [[clang::uninitialized]] DWORD64 time_adjustment;
    [[clang::uninitialized]] DWORD64 time_increment;
    [[clang::uninitialized]] BOOL time_adjustment_disabled;
    if (!::GetSystemTimeAdjustmentPrecise(&time_adjustment, &time_increment,
                                          &time_adjustment_disabled)) {
      libc_errno = EINVAL;
      return -1;
    }
    DWORD64 tv_sec = time_increment / HNS_PER_SEC;
    DWORD64 tv_nsec = (time_increment % HNS_PER_SEC) * 100ULL;
    if (LIBC_UNLIKELY(tv_sec > SEC_LIMIT)) {
      libc_errno = EOVERFLOW;
      return -1;
    }
    res->tv_sec = static_cast<decltype(res->tv_sec)>(tv_sec);
    res->tv_nsec = static_cast<decltype(res->tv_nsec)>(tv_nsec);
    break;
  }
  case CLOCK_PROCESS_CPUTIME_ID:
  case CLOCK_THREAD_CPUTIME_ID: {
    [[clang::uninitialized]] DWORD time_adjustment;
    [[clang::uninitialized]] DWORD time_increment;
    [[clang::uninitialized]] BOOL time_adjustment_disabled;
    if (!::GetSystemTimeAdjustment(&time_adjustment, &time_increment,
                                   &time_adjustment_disabled)) {
      libc_errno = EINVAL;
      return -1;
    }
    DWORD hns_per_sec = static_cast<DWORD>(HNS_PER_SEC);
    DWORD sec_limit = static_cast<DWORD>(SEC_LIMIT);
    DWORD tv_sec = time_increment / hns_per_sec;
    DWORD tv_nsec = (time_increment % hns_per_sec) * 100UL;
    if (LIBC_UNLIKELY(tv_sec > sec_limit)) {
      libc_errno = EOVERFLOW;
      return -1;
    }
    res->tv_sec = static_cast<decltype(res->tv_sec)>(tv_sec);
    res->tv_nsec = static_cast<decltype(res->tv_nsec)>(tv_nsec);
    break;
  }
  }
  return 0;
}
} // namespace LIBC_NAMESPACE_DECL

//===--- clock_gettime windows implementation -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/time/clock_gettime.h"
#include "include/llvm-libc-macros/windows/time-macros-ext.h"
#include "src/__support/CPP/atomic.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/time/units.h"
#include <Windows.h>

#ifdef __clang__
#define UNINITIALIZED [[clang::uninitialized]]
#else
#define UNINITIALIZED
#endif

namespace LIBC_NAMESPACE_DECL {
namespace internal {
static long long get_ticks_per_second() {
  static cpp::Atomic<long long> frequency = 0;
  auto freq = frequency.load(cpp::MemoryOrder::RELAXED);
  if (!freq) {
    UNINITIALIZED LARGE_INTEGER buffer;
    // On systems that run Windows XP or later, the function will always
    // succeed and will thus never return zero.
    ::QueryPerformanceFrequency(&buffer);
    frequency.store(buffer.QuadPart, cpp::MemoryOrder::RELAXED);
    return buffer.QuadPart;
  }
  return freq;
}

ErrorOr<int> clock_gettime(clockid_t clockid, timespec *ts) {
  using namespace time_units;
  ErrorOr<int> ret = 0;
  switch (clockid) {
  default:
    ret = cpp::unexpected(EINVAL);
    break;

  case CLOCK_MONOTONIC: {
    // see
    // https://learn.microsoft.com/en-us/windows/win32/sysinfo/acquiring-high-resolution-time-stamps
    // Is the performance counter monotonic (non-decreasing)?
    // Yes. QPC does not go backward.
    UNINITIALIZED LARGE_INTEGER buffer;
    // On systems that run Windows XP or later, the function will always
    // succeed and will thus never return zero.
    ::QueryPerformanceCounter(&buffer);
    long long freq = get_ticks_per_second();
    long long ticks = buffer.QuadPart;
    long long tv_sec = ticks / freq;
    long long tv_nsec = (ticks % freq) * 1_s_ns / freq;
    ts->tv_sec = static_cast<decltype(ts->tv_sec)>(tv_sec);
    ts->tv_nsec = static_cast<decltype(ts->tv_nsec)>(tv_nsec);
    break;
  }
  case CLOCK_REALTIME: {
    // https://learn.microsoft.com/en-us/windows/win32/api/sysinfoapi/nf-sysinfoapi-getsystemtimepreciseasfiletime
    // GetSystemTimePreciseAsFileTime
    // This function is best suited for high-resolution time-of-day
    // measurements, or time stamps that are synchronized to UTC
    UNINITIALIZED FILETIME file_time;
    UNINITIALIZED ULARGE_INTEGER time;
    ::GetSystemTimePreciseAsFileTime(&file_time);
    time.LowPart = file_time.dwLowDateTime;
    time.HighPart = file_time.dwHighDateTime;

    // adjust to POSIX epoch (from Jan 1, 1601 to Jan 1, 1970)
    constexpr unsigned long long HNS_PER_SEC = 1_s_ns / 100ULL;
    time.QuadPart -= (11644473600ULL * HNS_PER_SEC);
    unsigned long long tv_sec = time.QuadPart / HNS_PER_SEC;
    unsigned long long tv_nsec = (time.QuadPart % HNS_PER_SEC) * 100ULL;
    ts->tv_sec = static_cast<decltype(ts->tv_sec)>(tv_sec);
    ts->tv_nsec = static_cast<decltype(ts->tv_nsec)>(tv_nsec);
    break;
  }
  }
  return ret;
}
} // namespace internal
} // namespace LIBC_NAMESPACE_DECL

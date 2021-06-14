//===-- runtime/time-intrinsic.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Implements time-related intrinsic subroutines.

#include "time-intrinsic.h"

#include <ctime>

// CPU_TIME (Fortran 2018 16.9.57)
// We can use std::clock() from the <ctime> header as a fallback implementation
// that should be available everywhere. This may not provide the best resolution
// and is particularly troublesome on (some?) POSIX systems where CLOCKS_PER_SEC
// is defined as 10^6 regardless of the actual precision of std::clock().
// Therefore, we will usually prefer platform-specific alternatives when they
// are available.
//
// We can use SFINAE to choose a platform-specific alternative. To do so, we
// introduce a helper function template, whose overload set will contain only
// implementations relying on interfaces which are actually available. Each
// overload will have a dummy parameter whose type indicates whether or not it
// should be preferred. Any other parameters required for SFINAE should have
// default values provided.
namespace {
// Types for the dummy parameter indicating the priority of a given overload.
// We will invoke our helper with an integer literal argument, so the overload
// with the highest priority should have the type int.
using fallback_implementation = double;
using preferred_implementation = int;

// This is the fallback implementation, which should work everywhere.
template <typename Unused = void> double getCpuTime(fallback_implementation) {
  std::clock_t timestamp{std::clock()};
  if (timestamp != std::clock_t{-1}) {
    return static_cast<double>(timestamp) / CLOCKS_PER_SEC;
  }

  // Return some negative value to represent failure.
  return -1.0;
}

// POSIX implementation using clock_gettime. This is only enabled if
// clock_gettime is available.
template <typename T = int, typename U = struct timespec>
double getCpuTime(preferred_implementation,
    // We need some dummy parameters to pass to decltype(clock_gettime).
    T ClockId = 0, U *Timespec = nullptr,
    decltype(clock_gettime(ClockId, Timespec)) *Enabled = nullptr) {
#if defined CLOCK_THREAD_CPUTIME_ID
#define CLOCKID CLOCK_THREAD_CPUTIME_ID
#elif defined CLOCK_PROCESS_CPUTIME_ID
#define CLOCKID CLOCK_PROCESS_CPUTIME_ID
#elif defined CLOCK_MONOTONIC
#define CLOCKID CLOCK_MONOTONIC
#else
#define CLOCKID CLOCK_REALTIME
#endif
  struct timespec tspec;
  if (clock_gettime(CLOCKID, &tspec) == 0) {
    return tspec.tv_nsec * 1.0e-9 + tspec.tv_sec;
  }

  // Return some negative value to represent failure.
  return -1.0;
}
} // anonymous namespace

namespace Fortran::runtime {
extern "C" {

double RTNAME(CpuTime)() { return getCpuTime(0); }
} // extern "C"
} // namespace Fortran::runtime

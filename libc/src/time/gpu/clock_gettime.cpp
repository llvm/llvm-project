//===---------- GPU implementation of the POSIX clock_gettime function ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/clock_gettime.h"

#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "time_utils.h"

namespace LIBC_NAMESPACE_DECL {

constexpr uint64_t TICKS_PER_SEC = 1000000000UL;

LLVM_LIBC_FUNCTION(int, clock_gettime, (clockid_t clockid, timespec *ts)) {
  if (clockid != CLOCK_MONOTONIC || !ts)
    return -1;

  uint64_t ns_per_tick = TICKS_PER_SEC / GPU_CLOCKS_PER_SEC;
  uint64_t ticks = gpu::fixed_frequency_clock();

  ts->tv_nsec = (ticks * ns_per_tick) % TICKS_PER_SEC;
  ts->tv_sec = (ticks * ns_per_tick) / TICKS_PER_SEC;

  return 0;
}

} // namespace LIBC_NAMESPACE_DECL

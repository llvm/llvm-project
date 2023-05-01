//===-- Linux implementation of the clock function ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/clock.h"

#include "src/__support/CPP/limits.h"
#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"
#include "src/errno/libc_errno.h"

#include <sys/syscall.h> // For syscall numbers.
#include <time.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(clock_t, clock, ()) {
  struct timespec ts;
  long ret_val = __llvm_libc::syscall_impl(
      SYS_clock_gettime, CLOCK_PROCESS_CPUTIME_ID, reinterpret_cast<long>(&ts));
  if (ret_val < 0) {
    libc_errno = -ret_val;
    return clock_t(-1);
  }

  // The above syscall gets the CPU time in seconds plus nanoseconds.
  // The standard requires that we return clock_t(-1) if we cannot represent
  // clocks as a clock_t value.
  constexpr clock_t CLOCK_SECS_MAX =
      cpp::numeric_limits<clock_t>::max() / CLOCKS_PER_SEC;
  if (ts.tv_sec > CLOCK_SECS_MAX)
    return clock_t(-1);
  if (ts.tv_nsec / 1000000000 > CLOCK_SECS_MAX - ts.tv_sec)
    return clock_t(-1);

  // For the integer computation converting tv_nsec to clocks to work
  // correctly, we want CLOCKS_PER_SEC to be less than 1000000000.
  static_assert(1000000000 > CLOCKS_PER_SEC,
                "Expected CLOCKS_PER_SEC to be less than 1000000000.");
  return clock_t(ts.tv_sec * CLOCKS_PER_SEC +
                 ts.tv_nsec / (1000000000 / CLOCKS_PER_SEC));
}

} // namespace __llvm_libc

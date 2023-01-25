//===-- Linux implementation of the time function -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/time_func.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"

#include <errno.h>
#include <sys/syscall.h> // For syscall numbers.
#include <time.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(time_t, time, (time_t * tp)) {
  // TODO: Use the Linux VDSO to fetch the time and avoid the syscall.
  struct timespec ts;
  long ret_val = __llvm_libc::syscall_impl(SYS_clock_gettime, CLOCK_REALTIME,
                                           reinterpret_cast<long>(&ts));
  if (ret_val < 0) {
    errno = -ret_val;
    return -1;
  }

  if (tp != nullptr)
    *tp = time_t(ts.tv_sec);
  return time_t(ts.tv_sec);
}

} // namespace __llvm_libc

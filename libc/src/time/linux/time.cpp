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
#include "src/errno/libc_errno.h"
#include "src/time/linux/clockGetTimeImpl.h"

#include <sys/syscall.h> // For syscall numbers.
#include <time.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(time_t, time, (time_t * tp)) {
  // TODO: Use the Linux VDSO to fetch the time and avoid the syscall.
  struct timespec ts;
  auto result = internal::clock_gettimeimpl(CLOCK_REALTIME, &ts);
  if (!result.has_value()) {
    libc_errno = result.error();
    return -1;
  }

  if (tp != nullptr)
    *tp = time_t(ts.tv_sec);
  return time_t(ts.tv_sec);
}

} // namespace __llvm_libc

//===-- Linux implementation of the time function -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/time_macros.h"
#include "src/__support/common.h"
#include "src/__support/time/linux/clock_gettime.h"
#include "src/errno/libc_errno.h"
#include "src/time/time_func.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(time_t, time, (time_t * tp)) {
  // TODO: Use the Linux VDSO to fetch the time and avoid the syscall.
  struct timespec ts;
  auto result = internal::clock_gettime(CLOCK_REALTIME, &ts);
  if (!result.has_value()) {
    libc_errno = result.error();
    return -1;
  }

  if (tp != nullptr)
    *tp = time_t(ts.tv_sec);
  return time_t(ts.tv_sec);
}

} // namespace LIBC_NAMESPACE

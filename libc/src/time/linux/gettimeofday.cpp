//===-- Implementation of gettimeofday function ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/gettimeofday.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"
#include "src/errno/libc_errno.h"
#include "src/time/linux/clockGetTimeImpl.h"

#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE {

// TODO(michaelrj): Move this into time/linux with the other syscalls.
LLVM_LIBC_FUNCTION(int, gettimeofday,
                   (struct timeval * tv, [[maybe_unused]] void *unused)) {
  if (tv == nullptr)
    return 0;

  struct timespec ts;
  auto result = internal::clock_gettimeimpl(CLOCK_REALTIME, &ts);

  // A negative return value indicates an error with the magnitude of the
  // value being the error code.
  if (!result.has_value()) {
    libc_errno = result.error();
    return -1;
  }

  tv->tv_sec = ts.tv_sec;
  tv->tv_usec = static_cast<suseconds_t>(ts.tv_nsec / 1000);
  return 0;
}

} // namespace LIBC_NAMESPACE

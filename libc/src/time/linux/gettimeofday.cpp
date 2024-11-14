//===-- Implementation of gettimeofday function ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/gettimeofday.h"
#include "hdr/time_macros.h"
#include "hdr/types/suseconds_t.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/time/linux/clock_gettime.h"
#include "src/__support/time/units.h"
#include "src/errno/libc_errno.h"

namespace LIBC_NAMESPACE_DECL {

// TODO(michaelrj): Move this into time/linux with the other syscalls.
LLVM_LIBC_FUNCTION(int, gettimeofday,
                   (struct timeval * tv, [[maybe_unused]] void *unused)) {
  using namespace time_units;
  if (tv == nullptr)
    return 0;

  struct timespec ts;
  auto result = internal::clock_gettime(CLOCK_REALTIME, &ts);

  // A negative return value indicates an error with the magnitude of the
  // value being the error code.
  if (!result.has_value()) {
    libc_errno = result.error();
    return -1;
  }

  tv->tv_sec = ts.tv_sec;
  tv->tv_usec = static_cast<suseconds_t>(ts.tv_nsec / 1_us_ns);
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL

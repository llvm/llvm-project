//===-- Implementation of timespec_get for Linux --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/timespec_get.h"
#include "hdr/time_macros.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/time/linux/clock_gettime.h"
#include "src/errno/libc_errno.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, timespec_get, (struct timespec * ts, int base)) {
  if (base != TIME_UTC) {
    return 0;
  }

  auto result = internal::clock_gettime(CLOCK_REALTIME, ts);
  if (!result.has_value()) {
    libc_errno = result.error();
    return 0;
  }
  return base;
}

} // namespace LIBC_NAMESPACE_DECL

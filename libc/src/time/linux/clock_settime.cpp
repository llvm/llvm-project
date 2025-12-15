//===---------- Linux implementation of the POSIX clock_settime function --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/clock_settime.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/time/clock_settime.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, clock_settime,
                   (clockid_t clockid, const timespec *ts)) {
  auto result = internal::clock_settime(clockid, ts);

  // A negative return value indicates an error with the magnitude of the
  // value being the error code.
  if (!result.has_value()) {
    libc_errno = result.error();
    return -1;
  }
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL

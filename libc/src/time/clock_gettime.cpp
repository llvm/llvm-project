//===---------- Linux implementation of the POSIX clock_gettime function
//--------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/clock_gettime.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"

#include <errno.h>
#include <sys/syscall.h> // For syscall numbers.
#include <time.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, clock_gettime,
                   (clockid_t clockid, struct timespec *tp)) {
  long ret_val =
      __llvm_libc::syscall(SYS_clock_gettime, static_cast<long>(clockid),
                           reinterpret_cast<long>(tp));
  // A negative return value indicates an error with the magnitude of the
  // value being the error code.
  if (ret_val < 0) {
    errno = -ret_val;
    return -1;
  }

  return 0;
}

} // namespace __llvm_libc

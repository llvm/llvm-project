//===-- Linux implementation of nanosleep function ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/nanosleep.h"
#include "hdr/time_macros.h"
#include "src/__support/OSUtil/syscall.h" // For syscall functions.
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

#include <stdint.h>      // For int64_t.
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, nanosleep,
                   (const struct timespec *req, struct timespec *rem)) {
#if SYS_nanosleep
  int ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_nanosleep, req, rem);
#elif defined(SYS_clock_nanosleep_time64)
  static_assert(
      sizeof(time_t) == sizeof(int64_t),
      "SYS_clock_gettime64 requires struct timespec with 64-bit members.");
  int ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_clock_nanosleep_time64,
                                              CLOCK_REALTIME, 0, req, rem);
#else
#error "SYS_nanosleep and SYS_clock_nanosleep_time64 syscalls not available."
#endif

  if (ret < 0) {
    libc_errno = -ret;
    return -1;
  }
  return ret;
}

} // namespace LIBC_NAMESPACE_DECL

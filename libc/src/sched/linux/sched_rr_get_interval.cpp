//===-- Implementation of sched_rr_get_interval ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sched/sched_rr_get_interval.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

#include "hdr/types/pid_t.h"
#include "hdr/types/struct_timespec.h"
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, sched_rr_get_interval,
                   (pid_t tid, struct timespec *tp)) {
#if defined(SYS_sched_rr_get_interval_time64)
  int ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_sched_rr_get_interval_time64,
                                              tid, tp);
#elif defined(SYS_sched_rr_get_interval)
  static_assert(
      sizeof(timespec::tv_nsec) == sizeof(long),
      "This legacy syscall fallback is only safe on platforms where tv_nsec "
      "matches the register size (long). It is unsafe on 32-bit platforms "
      "with 64-bit tv_nsec.");
  int ret =
      LIBC_NAMESPACE::syscall_impl<int>(SYS_sched_rr_get_interval, tid, tp);
#else
#error                                                                         \
    "sched_rr_get_interval and sched_rr_get_interval_time64 syscalls not available."
#endif
  if (ret < 0) {
    libc_errno = -ret;
    return -1;
  }
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL

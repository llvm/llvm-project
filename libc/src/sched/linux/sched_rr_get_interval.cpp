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
#include "src/__support/macros/config.h"
#include "src/errno/libc_errno.h"

#include <sys/syscall.h> // For syscall numbers.

#ifdef SYS_sched_rr_get_interval_time64
#include <linux/time_types.h> // For __kernel_timespec.
#endif

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, sched_rr_get_interval,
                   (pid_t tid, struct timespec *tp)) {
#ifdef SYS_sched_rr_get_interval
  int ret =
      LIBC_NAMESPACE::syscall_impl<int>(SYS_sched_rr_get_interval, tid, tp);
#elif defined(SYS_sched_rr_get_interval_time64)
  // The difference between the  and SYS_sched_rr_get_interval
  // SYS_sched_rr_get_interval_time64 syscalls is the data type used for the
  // time interval parameter: the latter takes a struct __kernel_timespec
  int ret;
  if (tp) {
    struct __kernel_timespec ts32;
    ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_sched_rr_get_interval_time64,
                                            tid, &ts32);
    if (ret == 0) {
      tp->tv_sec = ts32.tv_sec;
      tp->tv_nsec = static_cast<long int>(ts32.tv_nsec);
    }
  } else
    // When tp is a nullptr, we still do the syscall to set ret and errno
    ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_sched_rr_get_interval_time64,
                                            tid, nullptr);
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

//===- Linux implementation of the POSIX clock_gettime function -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_TIME_LINUX_CLOCKGETTIMEIMPL_H
#define LLVM_LIBC_SRC_TIME_LINUX_CLOCKGETTIMEIMPL_H

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/errno/libc_errno.h"

#include <sys/syscall.h> // For syscall numbers.
#include <time.h>

namespace __llvm_libc {
namespace internal {

LIBC_INLINE ErrorOr<int> clock_gettimeimpl(clockid_t clockid,
                                           struct timespec *ts) {
#if SYS_clock_gettime
  int ret = __llvm_libc::syscall_impl<int>(SYS_clock_gettime,
                                           static_cast<long>(clockid),
                                           reinterpret_cast<long>(ts));
#elif defined(SYS_clock_gettime64)
  struct timespec64 ts64;
  int ret = __llvm_libc::syscall_impl<int>(SYS_clock_gettime64,
                                           static_cast<long>(clockid),
                                           reinterpret_cast<long>(&ts64));
  ts->tv_sec = static_cast<time_t>(ts64.tv_sec);
  ts->tv_nsec = static_cast<long>(ts64.tv_nsec);
#else
#error "SYS_clock_gettime and SYS_clock_gettime64 syscalls not available."
#endif
  if (ret < 0)
    return Error(-ret);
  return ret;
}

} // namespace internal
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_TIME_LINUX_CLOCKGETTIMEIMPL_H

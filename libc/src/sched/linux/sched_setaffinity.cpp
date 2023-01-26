//===-- Implementation of sched_setaffinity -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sched/sched_setaffinity.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"

#include <errno.h>
#include <sched.h>
#include <sys/syscall.h> // For syscall numbers.

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, sched_setaffinity,
                   (pid_t tid, size_t cpuset_size, const cpu_set_t *mask)) {
  long ret =
      __llvm_libc::syscall_impl(SYS_sched_setaffinity, tid, cpuset_size, mask);
  if (ret < 0) {
    errno = -ret;
    return -1;
  }
  return 0;
}

} // namespace __llvm_libc

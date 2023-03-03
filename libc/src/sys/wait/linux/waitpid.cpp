//===-- Linux implementation of waitpid -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/wait/waitpid.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"

#include "src/errno/libc_errno.h"
#include <sys/syscall.h> // For syscall numbers.
#include <sys/wait.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(pid_t, waitpid, (pid_t pid, int *wait_status, int options)) {
  pid = __llvm_libc::syscall_impl(SYS_wait4, pid, wait_status, options, 0);
  if (pid < 0) {
    libc_errno = -pid;
    return -1;
  }
  return pid;
}

} // namespace __llvm_libc

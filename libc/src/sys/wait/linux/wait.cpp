//===-- Linux implementation of wait --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/wait/wait.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"

#include <errno.h>
#include <sys/syscall.h> // For syscall numbers.
#include <sys/wait.h>

namespace __llvm_libc {

// The implementation of wait here is very minimal. We will add more
// functionality and standard compliance in future.

LLVM_LIBC_FUNCTION(pid_t, wait, (int *wait_status)) {
  pid_t pid = __llvm_libc::syscall_impl(SYS_wait4, -1, wait_status, 0, 0);
  if (pid < 0) {
    // Error case, a child process was not created.
    errno = -pid;
    return -1;
  }

  return pid;
}

} // namespace __llvm_libc

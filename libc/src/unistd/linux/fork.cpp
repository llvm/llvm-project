//===-- Linux implementation of fork --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/fork.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"
#include "src/__support/threads/thread.h" // For thread self object

#include <errno.h>
#include <sys/syscall.h> // For syscall numbers.

namespace __llvm_libc {

// The implementation of fork here is very minimal. We will add more
// functionality and standard compliance in future.

LLVM_LIBC_FUNCTION(pid_t, fork, (void)) {
#ifdef SYS_fork
  pid_t ret = __llvm_libc::syscall_impl(SYS_fork);
#elif defined(SYS_clone)
  pid_t ret = __llvm_libc::syscall_impl(SYS_clone, SIGCHLD, 0);
#else
#error "SYS_fork or SYS_clone not available."
#endif
  if (ret == 0) {
    // Return value is 0 in the child process.
    // The child is created with a single thread whose self object will be a
    // copy of parent process' thread which called fork. So, we have to fix up
    // the child process' self object with the new process' tid.
    self.attrib->tid = __llvm_libc::syscall_impl(SYS_gettid);
    return 0;
  }

  if (ret < 0) {
    // Error case, a child process was not created.
    errno = -ret;
    return -1;
  }

  return ret;
}

} // namespace __llvm_libc

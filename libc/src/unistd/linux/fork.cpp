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
#include "src/__support/macros/config.h"
#include "src/__support/threads/fork_callbacks.h"
#include "src/__support/threads/identifier.h"
#include "src/__support/threads/thread.h" // For thread self object

#include "src/__support/libc_errno.h"
#include <signal.h>      // For SIGCHLD
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE_DECL {

// The implementation of fork here is very minimal. We will add more
// functionality and standard compliance in future.

LLVM_LIBC_FUNCTION(pid_t, fork, (void)) {
  invoke_prepare_callbacks();
  pid_t parent_tid = internal::gettid();
  // Invalidate parent's tid cache before forking. We cannot do this in child
  // process because in the post-fork instruction windows, there may be a signal
  // handler triggered which may get the wrong tid.
  internal::force_set_tid(0);
#ifdef SYS_fork
  pid_t ret = syscall_impl<pid_t>(SYS_fork);
#elif defined(SYS_clone)
  pid_t ret = syscall_impl<pid_t>(SYS_clone, SIGCHLD, 0);
#else
#error "fork and clone syscalls not available."
#endif

  if (ret == 0) {
    // Return value is 0 in the child process.
    // The child is created with a single thread whose self object will be a
    // copy of parent process' thread which called fork. So, we have to fix up
    // the child process' self object with the new process' tid.
    internal::force_set_tid(syscall_impl<pid_t>(SYS_gettid));
    invoke_child_callbacks();
    return 0;
  }

  if (ret < 0) {
    // Error case, a child process was not created.
    libc_errno = static_cast<int>(-ret);
    return -1;
  }
  // recover parent's tid.
  internal::force_set_tid(parent_tid);
  invoke_parent_callbacks();
  return ret;
}

} // namespace LIBC_NAMESPACE_DECL

//===-- Linux implementation of fork --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/fork.h"

#include "src/__support/OSUtil/pid.h"
#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/threads/fork_callbacks.h"
#include "src/__support/threads/thread.h" // For thread self object
#include "src/errno/libc_errno.h"

#include <signal.h>      // For SIGCHLD
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE_DECL {

// The implementation of fork here is very minimal. We will add more
// functionality and standard compliance in future.

LLVM_LIBC_FUNCTION(pid_t, fork, (void)) {
  invoke_prepare_callbacks();

  // Invalidate tid/pid cache before fork to avoid post fork signal handler from
  // getting wrong values. gettid() is not async-signal-safe, but let's provide
  // our best efforts here.
  pid_t parent_tid = self.get_tid();
  self.invalidate_tid();
  ProcessIdentity::start_fork();

#ifdef SYS_fork
  pid_t ret = LIBC_NAMESPACE::syscall_impl<pid_t>(SYS_fork);
#elif defined(SYS_clone)
  pid_t ret = LIBC_NAMESPACE::syscall_impl<pid_t>(SYS_clone, SIGCHLD, 0);
#else
#error "fork and clone syscalls not available."
#endif

  if (ret < 0) {
    // Error case, a child process was not created.
    libc_errno = static_cast<int>(-ret);
    return -1;
  }

  // Child process
  if (ret == 0) {
    self.refresh_tid();
    ProcessIdentity::refresh_cache();
    ProcessIdentity::end_fork();
    invoke_child_callbacks();
    return 0;
  }

  // Parent process
  self.refresh_tid(parent_tid);
  ProcessIdentity::end_fork();
  invoke_parent_callbacks();
  return ret;
}

} // namespace LIBC_NAMESPACE_DECL

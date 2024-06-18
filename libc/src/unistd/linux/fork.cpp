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
#include "src/__support/threads/fork_callbacks.h"
#include "src/__support/threads/thread.h" // For thread self object
#include "src/errno/libc_errno.h"

#include <signal.h>      // For SIGCHLD
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE {

// The implementation of fork here is very minimal. We will add more
// functionality and standard compliance in future.

LLVM_LIBC_FUNCTION(pid_t, fork, (void)) {
  invoke_prepare_callbacks();

  // Invalidate tid/pid cache before fork to avoid post fork signal handler from
  // getting wrong values. gettid() is not async-signal-safe, but let's provide
  // our best efforts here.
  self.invalidate_tid();
  ProcessIdentity::invalidate_cache();

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

  // Refresh tid/pid cache after fork
  self.refresh_tid();
  ProcessIdentity::refresh_cache();

  if (ret == 0) {
    invoke_child_callbacks();
    return 0;
  }

  invoke_parent_callbacks();
  return ret;
}

} // namespace LIBC_NAMESPACE

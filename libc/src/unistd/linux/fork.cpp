//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of fork.
///
//===----------------------------------------------------------------------===//

#include "src/unistd/fork.h"

#include "hdr/signal_macros.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/clone3.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/gettid.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/threads/fork_callbacks.h"
#include "src/__support/threads/identifier.h"
#include "src/__support/threads/thread.h" // For thread self object

#include "src/__support/libc_errno.h"

#include <linux/sched.h> // For struct clone_args

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
  struct clone_args args{};
  args.exit_signal = SIGCHLD;
  auto result = linux_syscalls::clone3(&args, sizeof(args));

  if (result && result.value() == 0) {
    // Return value is 0 in the child process.
    // The child is created with a single thread whose self object will be a
    // copy of parent process' thread which called fork. So, we have to fix up
    // the child process' self object with the new process' tid.
    internal::force_set_tid(linux_syscalls::gettid());
    invoke_child_callbacks();
    return 0;
  }

  if (!result) {
    // Error case, a child process was not created.
    libc_errno = result.error();
    return -1;
  }
  // recover parent's tid.
  internal::force_set_tid(parent_tid);
  invoke_parent_callbacks();
  return result.value();
}

} // namespace LIBC_NAMESPACE_DECL

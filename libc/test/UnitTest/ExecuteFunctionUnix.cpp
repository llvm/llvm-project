//===-- ExecuteFunction implementation for Unix-like Systems --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ExecuteFunction.h"
#include "src/__support/libc_assert.h"
#include "src/__support/macros/config.h"
#include "test/UnitTest/ExecuteFunction.h" // FunctionCaller

#ifdef LIBC_FULL_BUILD
#include "include/llvm-libc-macros/poll-macros.h"
#include "include/llvm-libc-macros/signal-macros.h"
#include "include/llvm-libc-macros/sys-wait-macros.h"
#include "include/llvm-libc-types/pid_t.h"
#include "include/llvm-libc-types/struct_pollfd.h"
#include "src/__support/libc_assert.h"
#include "src/poll/poll.h"
#include "src/signal/kill.h"
#include "src/stdio/fflush.h"
#include "src/stdio/stderr.h"
#include "src/stdio/stdout.h"
#include "src/stdlib/exit.h"
#include "src/string/strsignal.h"
#include "src/sys/wait/waitpid.h"
#include "src/unistd/close.h"
#include "src/unistd/fork.h"
#include "src/unistd/pipe.h"

#define LIBC_IMPL LIBC_NAMESPACE

#else // Overlay mode
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#define LIBC_IMPL
#endif

namespace LIBC_NAMESPACE_DECL {
namespace testutils {

bool ProcessStatus::exited_normally() { return WIFEXITED(platform_defined); }

int ProcessStatus::get_exit_code() {
  LIBC_ASSERT(exited_normally() && "Abnormal termination, no exit code");
  return WEXITSTATUS(platform_defined);
}

int ProcessStatus::get_fatal_signal() {
  if (exited_normally())
    return 0;
  return WTERMSIG(platform_defined);
}

ProcessStatus invoke_in_subprocess(FunctionCaller *func, int timeout_ms) {
  int pipe_fds[2];
  if (LIBC_IMPL::pipe(pipe_fds) == -1) {
    delete func;
    return ProcessStatus::error("pipe(2) failed");
  }

  // Don't copy the buffers into the child process and print twice.
  LIBC_IMPL::fflush(stderr);
  LIBC_IMPL::fflush(stdout);
  pid_t pid = LIBC_IMPL::fork();
  if (pid == -1) {
    delete func;
    return ProcessStatus::error("fork(2) failed");
  }

  if (!pid) {
    (*func)();
    delete func;
    LIBC_IMPL::exit(0);
  }
  LIBC_IMPL::close(pipe_fds[1]);

  pollfd poll_fd{pipe_fds[0], POLLIN, 0};
  // No events requested so this call will only return after the timeout or if
  // the pipes peer was closed, signaling the process exited.
  if (LIBC_IMPL::poll(&poll_fd, 1, timeout_ms) == -1) {
    delete func;
    return ProcessStatus::error("poll(2) failed");
  }
  // If the pipe wasn't closed by the child yet then timeout has expired.
  if (!(poll_fd.revents & POLLHUP)) {
    LIBC_IMPL::kill(pid, SIGKILL);
    delete func;
    return ProcessStatus::timed_out_ps();
  }

  int wstatus = 0;
  // Wait on the pid of the subprocess here so it gets collected by the system
  // and doesn't turn into a zombie.
  pid_t status = LIBC_IMPL::waitpid(pid, &wstatus, 0);
  if (status == -1) {
    delete func;
    return ProcessStatus::error("waitpid(2) failed");
  }
  LIBC_ASSERT(status == pid);
  delete func;
  return {wstatus};
}

const char *signal_as_string(int signum) {
  return LIBC_IMPL::strsignal(signum);
}

} // namespace testutils
} // namespace LIBC_NAMESPACE_DECL

//===-- ExecuteFunction implementation for Unix-like Systems --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifdef LIBC_HERMETIC_TEST_FRAMEWORK
// new goes first to provide needed operators
#include "src/__support/CPP/new.h"
#include "ExecuteFunction.h"

// TODO: for now, we use epoll directly. Use poll for unix compatibility when it
// is available.
#include "hdr/sys_epoll_macros.h"
#include "hdr/sys_wait_macros.h"
#include "src/__support/libc_assert.h"
#include "src/__support/macros/config.h"
#include "src/signal/kill.h"
#include "src/stdio/fflush.h"
#include "src/stdlib/exit.h"
#include "src/string/strsignal.h"
#include "src/sys/epoll/epoll_create1.h"
#include "src/sys/epoll/epoll_ctl.h"
#include "src/sys/epoll/epoll_wait.h"
#include "src/sys/wait/waitpid.h"
#include "src/unistd/close.h"
#include "src/unistd/fork.h"
#include "src/unistd/pipe.h"
#else
#include "ExecuteFunction.h"
#include <assert.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/epoll.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
namespace LIBC_NAMESPACE_DECL {
using ::close;
using ::exit;
using ::fflush;
using ::fork;
using ::kill;
using ::pipe;
using ::waitpid;
using ::epoll_create1;
using ::epoll_ctl;
using ::epoll_wait;
using ::strsignal;
#define LIBC_ASSERT(...) assert(__VA_ARGS__)
} // namespace LIBC_NAMESPACE_DECL
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

template <typename T> struct DeleteGuard {
  T *ptr;
  DeleteGuard(T *p) : ptr(p) {}
  ~DeleteGuard() { delete ptr; }
};
// deduction guide
template <typename T> DeleteGuard(T *) -> DeleteGuard<T>;

ProcessStatus invoke_in_subprocess(FunctionCaller *func, unsigned timeout_ms) {
  DeleteGuard guard(func);

  int pipe_fds[2];
  if (LIBC_NAMESPACE::pipe(pipe_fds) == -1)
    return ProcessStatus::error("pipe(2) failed");

  // Don't copy the buffers into the child process and print twice.
  LIBC_NAMESPACE::fflush(stdout);
  LIBC_NAMESPACE::fflush(stderr);
  pid_t pid = fork();
  if (pid == -1)
    return ProcessStatus::error("fork(2) failed");

  if (!pid) {
    (*func)();
    LIBC_NAMESPACE::exit(0);
  }

  LIBC_NAMESPACE::close(pipe_fds[1]);

  // Create an epoll instance.
  int epfd = LIBC_NAMESPACE::epoll_create1(0);
  if (epfd == -1) {
    return ProcessStatus::error("epoll_create1(2) failed");
  }

  // Register the pipe FD with epoll. We monitor for reads (EPOLLIN)
  // plus any half-close/hangup events (EPOLLRDHUP). epoll will set
  // EPOLLHUP automatically if the peer closes, but typically you also
  // include EPOLLIN or EPOLLRDHUP in the event mask.
  epoll_event ev = {};
  ev.events = EPOLLIN | EPOLLRDHUP;
  ev.data.fd = pipe_fds[0];

  if (LIBC_NAMESPACE::epoll_ctl(epfd, EPOLL_CTL_ADD, pipe_fds[0], &ev) == -1) {
    LIBC_NAMESPACE::close(epfd);
    return ProcessStatus::error("epoll_ctl(2) failed");
  }

  // Block until epoll signals an event or times out.
  epoll_event result = {};
  int nfds = LIBC_NAMESPACE::epoll_wait(epfd, &result, 1, timeout_ms);
  LIBC_NAMESPACE::close(
      epfd); // We’re done with the epoll FD regardless of outcome.

  if (nfds == -1) {
    return ProcessStatus::error("epoll_wait(2) failed");
  }

  // If no FDs became “ready,” the child didn’t close the pipe before the
  // timeout.
  if (nfds == 0) {
    while (LIBC_NAMESPACE::kill(pid, SIGKILL) == 0)
      ;
    return ProcessStatus::timed_out_ps();
  }

  // If we did get an event, check for EPOLLHUP (or EPOLLRDHUP).
  // If those are not set, the pipe wasn't closed in the manner we expected.
  if (!(result.events & (EPOLLHUP | EPOLLRDHUP))) {
    while (LIBC_NAMESPACE::kill(pid, SIGKILL) == 0)
      ;
    return ProcessStatus::timed_out_ps();
  }

  int wstatus = 0;
  // Wait on the pid of the subprocess here so it gets collected by the system
  // and doesn't turn into a zombie.
  pid_t status = LIBC_NAMESPACE::waitpid(pid, &wstatus, 0);
  if (status == -1)
    return ProcessStatus::error("waitpid(2) failed");
  LIBC_ASSERT(status == pid);
  return {wstatus};
}

const char *signal_as_string(int signum) {
  return LIBC_NAMESPACE::strsignal(signum);
}

} // namespace testutils
} // namespace LIBC_NAMESPACE_DECL

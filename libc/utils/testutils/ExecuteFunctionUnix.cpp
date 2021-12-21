//===-- ExecuteFunction implementation for Unix-like Systems --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ExecuteFunction.h"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

namespace __llvm_libc {
namespace testutils {

bool ProcessStatus::exited_normally() const {
  return WIFEXITED(platform_defined);
}

int ProcessStatus::get_exit_code() const {
  assert(exited_normally() && "Abnormal termination, no exit code");
  return WEXITSTATUS(platform_defined);
}

int ProcessStatus::get_fatal_signal() const {
  if (exited_normally())
    return 0;
  return WTERMSIG(platform_defined);
}

ProcessStatus invoke_in_subprocess(FunctionCaller *Func, unsigned timeoutMS) {
  std::unique_ptr<FunctionCaller> X(Func);
  int pipeFDs[2];
  if (::pipe(pipeFDs) == -1)
    return ProcessStatus::error("pipe(2) failed");

  // Don't copy the buffers into the child process and print twice.
  std::cout.flush();
  std::cerr.flush();
  pid_t Pid = ::fork();
  if (Pid == -1)
    return ProcessStatus::error("fork(2) failed");

  if (!Pid) {
    (*Func)();
    std::exit(0);
  }
  ::close(pipeFDs[1]);

  struct pollfd pollFD {
    pipeFDs[0], 0, 0
  };
  // No events requested so this call will only return after the timeout or if
  // the pipes peer was closed, signaling the process exited.
  if (::poll(&pollFD, 1, timeoutMS) == -1)
    return ProcessStatus::error("poll(2) failed");
  // If the pipe wasn't closed by the child yet then timeout has expired.
  if (!(pollFD.revents & POLLHUP)) {
    ::kill(Pid, SIGKILL);
    return ProcessStatus::timed_out_ps();
  }

  int WStatus = 0;
  // Wait on the pid of the subprocess here so it gets collected by the system
  // and doesn't turn into a zombie.
  pid_t status = ::waitpid(Pid, &WStatus, 0);
  if (status == -1)
    return ProcessStatus::error("waitpid(2) failed");
  assert(status == Pid);
  (void)status;
  return {WStatus};
}

const char *signal_as_string(int Signum) { return ::strsignal(Signum); }

} // namespace testutils
} // namespace __llvm_libc

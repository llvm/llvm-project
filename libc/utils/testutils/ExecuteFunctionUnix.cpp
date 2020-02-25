//===------- ExecuteFunction implementation for Unix-like Systems ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ExecuteFunction.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdlib>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

namespace __llvm_libc {
namespace testutils {

bool ProcessStatus::exitedNormally() { return WIFEXITED(PlatformDefined); }

int ProcessStatus::getExitCode() {
  assert(exitedNormally() && "Abnormal termination, no exit code");
  return WEXITSTATUS(PlatformDefined);
}

int ProcessStatus::getFatalSignal() {
  if (exitedNormally())
    return 0;
  return WTERMSIG(PlatformDefined);
}

ProcessStatus invokeInSubprocess(FunctionCaller *Func) {
  // Don't copy the buffers into the child process and print twice.
  llvm::outs().flush();
  llvm::errs().flush();
  pid_t Pid = ::fork();
  if (!Pid) {
    (*Func)();
    std::exit(0);
  }

  int WStatus;
  ::waitpid(Pid, &WStatus, 0);
  delete Func;
  return {WStatus};
}

const char *signalAsString(int Signum) { return ::strsignal(Signum); }

} // namespace testutils
} // namespace __llvm_libc

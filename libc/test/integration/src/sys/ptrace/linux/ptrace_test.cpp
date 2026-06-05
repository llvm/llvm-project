//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Integration test for ptrace.
///
//===----------------------------------------------------------------------===//

#include "hdr/signal_macros.h"
#include "src/signal/kill.h"
#include "src/signal/raise.h"
#include "src/stdlib/exit.h"
#include "src/sys/ptrace/ptrace.h"
#include "src/sys/wait/waitpid.h"
#include "src/unistd/fork.h"
#include "test/IntegrationTest/test.h"
#include <errno.h>
#include <sys/ptrace.h>

void trace_me_test() {
  pid_t pid = LIBC_NAMESPACE::fork();
  ASSERT_TRUE(pid >= 0);

  if (pid == 0) {
    // Child process
    long ret = LIBC_NAMESPACE::ptrace(PTRACE_TRACEME, 0, nullptr, nullptr);
    if (ret != 0)
      LIBC_NAMESPACE::internal::exit(errno);
    LIBC_NAMESPACE::raise(SIGSTOP);
    LIBC_NAMESPACE::internal::exit(0);
  }

  // Parent process
  int status;
  pid_t wait_ret = LIBC_NAMESPACE::waitpid(pid, &status, 0);
  ASSERT_EQ(wait_ret, pid);
  ASSERT_TRUE(WIFSTOPPED(status));
  ASSERT_EQ(WSTOPSIG(status), SIGSTOP);

  // Kill the child
  ASSERT_EQ(LIBC_NAMESPACE::kill(pid, SIGKILL), 0);
  wait_ret = LIBC_NAMESPACE::waitpid(pid, &status, 0);
  ASSERT_EQ(wait_ret, pid);
  ASSERT_TRUE(WIFSIGNALED(status));
  ASSERT_EQ(WTERMSIG(status), SIGKILL);
}

void errno_test() {
  errno = 0;
  ASSERT_EQ(LIBC_NAMESPACE::ptrace(-1, 0, nullptr, nullptr), -1L);
  ASSERT_ERRNO_EQ(ESRCH);
}

TEST_MAIN() {
  trace_me_test();
  errno_test();
  return 0;
}

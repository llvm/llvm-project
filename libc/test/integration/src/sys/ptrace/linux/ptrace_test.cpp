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
#include <sys/wait.h>

void basic_ptrace_test() {
  // This variable will be read and written to via ptrace.
  volatile long test_variable = 0;
  pid_t pid = LIBC_NAMESPACE::fork();
  ASSERT_TRUE(pid >= 0);

  if (pid == 0) {
    // Child process
    test_variable = 0xdeadbeef;
    long ret = LIBC_NAMESPACE::ptrace(PTRACE_TRACEME, 0, nullptr, nullptr);
    if (ret != 0)
      LIBC_NAMESPACE::internal::exit(errno);
    LIBC_NAMESPACE::raise(SIGSTOP);
    LIBC_NAMESPACE::internal::exit(static_cast<int>(test_variable));
  }

  // Parent process
  int status;
  pid_t wait_ret = LIBC_NAMESPACE::waitpid(pid, &status, 0);
  ASSERT_EQ(wait_ret, pid);
  ASSERT_TRUE(WIFSTOPPED(status));
  ASSERT_EQ(WSTOPSIG(status), SIGSTOP);

  // Try reading from an invalid address, and check for failure.
  ASSERT_EQ(-1,
            LIBC_NAMESPACE::ptrace(PTRACE_PEEKDATA, pid,
                                   reinterpret_cast<void *>(0x470), nullptr));
  ASSERT_ERRNO_EQ(EIO);

  errno = 0;
  // We're reading the value of test_variable from the child process. As the
  // child is a fork of ourselves, the variable will have the same address in
  // both processes.
  long value =
      LIBC_NAMESPACE::ptrace(PTRACE_PEEKDATA, pid, &test_variable, nullptr);
  ASSERT_EQ(value, 0xdeadbeef);
  ASSERT_ERRNO_SUCCESS();

  // Now modify the variable
  ASSERT_EQ(0,
            LIBC_NAMESPACE::ptrace(PTRACE_POKEDATA, pid, &test_variable, 47));
  ASSERT_ERRNO_SUCCESS();

  // Resume the child, it should read back the modified value.
  ASSERT_EQ(0, LIBC_NAMESPACE::ptrace(PTRACE_CONT, pid, nullptr, nullptr));
  ASSERT_ERRNO_SUCCESS();
  wait_ret = LIBC_NAMESPACE::waitpid(pid, &status, 0);
  ASSERT_EQ(wait_ret, pid);
  ASSERT_TRUE(WIFEXITED(status));
  ASSERT_EQ(WEXITSTATUS(status), 47);
}

void errno_test() {
  errno = 0;
  ASSERT_EQ(LIBC_NAMESPACE::ptrace(-1, 0, nullptr, nullptr), -1L);
  ASSERT_ERRNO_EQ(ESRCH);
}

TEST_MAIN() {
  basic_ptrace_test();
  errno_test();
  return 0;
}

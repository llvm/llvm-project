//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Integration test for execvp.
///
//===----------------------------------------------------------------------===//

#include "src/sys/wait/waitpid.h"
#include "src/unistd/execvp.h"
#include "src/unistd/fork.h"

#include "hdr/signal_macros.h"
#include "hdr/sys_wait_macros.h"
#include "test/IntegrationTest/test.h"

void fork_and_execvp_path_search() {
  pid_t pid = LIBC_NAMESPACE::fork();
  if (pid == 0) {
    char *const argv[] = {
        const_cast<char *>("libc_execv_test_normal_exit"),
        nullptr,
    };
    LIBC_NAMESPACE::execvp("libc_execv_test_normal_exit", argv);
  }
  ASSERT_TRUE(pid > 0);
  int status;
  pid_t cpid = LIBC_NAMESPACE::waitpid(pid, &status, 0);
  ASSERT_TRUE(cpid > 0);
  ASSERT_EQ(cpid, pid);
  ASSERT_TRUE(WIFEXITED(status));
}

void fork_and_execvp_slash_path() {
  pid_t pid = LIBC_NAMESPACE::fork();
  if (pid == 0) {
    char *const argv[] = {
        const_cast<char *>("libc_execv_test_normal_exit"),
        nullptr,
    };
    LIBC_NAMESPACE::execvp("./libc_execv_test_normal_exit", argv);
  }
  ASSERT_TRUE(pid > 0);
  int status;
  pid_t cpid = LIBC_NAMESPACE::waitpid(pid, &status, 0);
  ASSERT_TRUE(cpid > 0);
  ASSERT_EQ(cpid, pid);
  ASSERT_TRUE(WIFEXITED(status));
}

void fork_and_execvp_signal_exit() {
  pid_t pid = LIBC_NAMESPACE::fork();
  if (pid == 0) {
    char *const argv[] = {
        const_cast<char *>("libc_exec_test_signal_exit"),
        nullptr,
    };
    LIBC_NAMESPACE::execvp("libc_exec_test_signal_exit", argv);
  }
  ASSERT_TRUE(pid > 0);
  int status;
  pid_t cpid = LIBC_NAMESPACE::waitpid(pid, &status, 0);
  ASSERT_TRUE(cpid > 0);
  ASSERT_EQ(cpid, pid);
  ASSERT_FALSE(WIFEXITED(status));
  ASSERT_TRUE(WTERMSIG(status) == SIGUSR1);
}

TEST_MAIN([[maybe_unused]] int argc, [[maybe_unused]] char **argv,
          [[maybe_unused]] char **envp) {
  fork_and_execvp_path_search();
  fork_and_execvp_slash_path();
  fork_and_execvp_signal_exit();
  return 0;
}

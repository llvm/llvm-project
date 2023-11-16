//===-- Unittests for execve ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/wait/waitpid.h"
#include "src/unistd/execve.h"
#include "src/unistd/fork.h"

#include "test/IntegrationTest/test.h"

#include <signal.h>
#include <sys/wait.h>

void fork_and_execv_normal_exit(char **envp) {
  pid_t pid = LIBC_NAMESPACE::fork();
  if (pid == 0) {
    const char *path = "libc_execv_test_normal_exit";
    char *const argv[] = {
        const_cast<char *>("execv_test_normal_exit"),
        nullptr,
    };
    LIBC_NAMESPACE::execve(path, argv, envp);
  }
  ASSERT_TRUE(pid > 0);
  int status;
  pid_t cpid = LIBC_NAMESPACE::waitpid(pid, &status, 0);
  ASSERT_TRUE(cpid > 0);
  ASSERT_EQ(cpid, pid);
  ASSERT_TRUE(WIFEXITED(status));
}

void fork_and_execv_signal_exit(char **envp) {
  pid_t pid = LIBC_NAMESPACE::fork();
  if (pid == 0) {
    const char *path = "libc_execv_test_signal_exit";
    char *const argv[] = {
        const_cast<char *>("execv_test_normal_exit"),
        nullptr,
    };
    LIBC_NAMESPACE::execve(path, argv, envp);
  }
  ASSERT_TRUE(pid > 0);
  int status;
  pid_t cpid = LIBC_NAMESPACE::waitpid(pid, &status, 0);
  ASSERT_TRUE(cpid > 0);
  ASSERT_EQ(cpid, pid);
  ASSERT_FALSE(WIFEXITED(status));
  ASSERT_TRUE(WTERMSIG(status) == SIGUSR1);
}

TEST_MAIN(int argc, char **argv, char **envp) {
  fork_and_execv_normal_exit(envp);
  fork_and_execv_signal_exit(envp);
  return 0;
}

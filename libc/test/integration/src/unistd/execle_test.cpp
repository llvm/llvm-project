//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Unittests for execle
///
//===----------------------------------------------------------------------===//

#include "src/sys/wait/waitpid.h"
#include "src/unistd/execle.h"
#include "src/unistd/fork.h"

#include "test/IntegrationTest/test.h"

#include <signal.h>
#include <sys/wait.h>

void fork_and_execle_normal_exit(char **envp) {
  pid_t pid = LIBC_NAMESPACE::fork();
  if (pid == 0) {
    const char *path = "libc_execle_test_normal_exit";
    LIBC_NAMESPACE::execle(path, const_cast<char *>("execle_test_normal_exit"),
                           const_cast<char *>("first"),
                           const_cast<char *>("second"),
                           static_cast<char *>(nullptr), envp);
  }

  ASSERT_TRUE(pid > 0);
  int status;
  pid_t cpid = LIBC_NAMESPACE::waitpid(pid, &status, 0);
  ASSERT_TRUE(cpid > 0);
  ASSERT_EQ(cpid, pid);
  ASSERT_TRUE(WIFEXITED(status));
}

void fork_and_execle_signal_exit(char **envp) {
  pid_t pid = LIBC_NAMESPACE::fork();
  if (pid == 0) {
    const char *path = "libc_exec_test_signal_exit";
    LIBC_NAMESPACE::execle(path, const_cast<char *>("exec_test_signal_exit"),
                           static_cast<char *>(nullptr), envp);
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
          char **envp) {
  fork_and_execle_normal_exit(envp);
  fork_and_execle_signal_exit(envp);
  return 0;
}

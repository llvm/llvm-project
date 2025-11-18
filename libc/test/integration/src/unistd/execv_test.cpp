//===-- Unittests for execv -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/wait/waitpid.h"
#include "src/unistd/execv.h"
#include "src/unistd/fork.h"

#include "test/IntegrationTest/test.h"

#include <signal.h>
#include <sys/wait.h>

void fork_and_execv_normal_exit() {
  pid_t pid = LIBC_NAMESPACE::fork();
  if (pid == 0) {
    const char *path = "libc_execv_test_normal_exit";
    char *const argv[] = {
        const_cast<char *>("execv_test_normal_exit"),
        nullptr,
    };
    LIBC_NAMESPACE::execv(path, argv);
  }
  ASSERT_TRUE(pid > 0);
  int status;
  pid_t cpid = LIBC_NAMESPACE::waitpid(pid, &status, 0);
  ASSERT_TRUE(cpid > 0);
  ASSERT_EQ(cpid, pid);
  ASSERT_TRUE(WIFEXITED(status));
}

void fork_and_execv_signal_exit() {
  pid_t pid = LIBC_NAMESPACE::fork();
  if (pid == 0) {
    const char *path = "libc_execv_test_signal_exit";
    char *const argv[] = {
        const_cast<char *>("execv_test_normal_exit"),
        nullptr,
    };
    LIBC_NAMESPACE::execv(path, argv);
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
  fork_and_execv_normal_exit();
  fork_and_execv_signal_exit();
  return 0;
}

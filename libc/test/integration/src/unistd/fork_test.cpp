//===-- Unittests for fork ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/signal/raise.h"
#include "src/sys/wait/wait.h"
#include "src/unistd/fork.h"

#include "utils/IntegrationTest/test.h"

#include <errno.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

void fork_and_wait_normal_exit() {
  pid_t pid = __llvm_libc::fork();
  if (pid == 0)
    return; // Just end without any thing special.
  ASSERT_TRUE(pid > 0);
  int status;
  pid_t cpid = __llvm_libc::wait(&status);
  ASSERT_TRUE(cpid > 0);
  ASSERT_EQ(cpid, pid);
  ASSERT_TRUE(WIFEXITED(status));
}

void fork_and_wait_signal_exit() {
  pid_t pid = __llvm_libc::fork();
  if (pid == 0)
    __llvm_libc::raise(SIGUSR1);
  ASSERT_TRUE(pid > 0);
  int status;
  pid_t cpid = __llvm_libc::wait(&status);
  ASSERT_TRUE(cpid > 0);
  ASSERT_EQ(cpid, pid);
  ASSERT_FALSE(WIFEXITED(status));
  ASSERT_TRUE(WTERMSIG(status) == SIGUSR1);
}

TEST_MAIN(int argc, char **argv, char **envp) {
  fork_and_wait_normal_exit();
  fork_and_wait_signal_exit();
  return 0;
}

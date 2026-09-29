//===--- Stack smashing test to check stack canary set up  ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/string.h"
#include "src/__support/OSUtil/io.h"
#include "src/pthread/pthread_atfork.h"
#include "src/signal/raise.h"
#include "src/sys/wait/wait.h"
#include "src/sys/wait/wait4.h"
#include "src/sys/wait/waitpid.h"
#include "src/unistd/fork.h"

#include "test/IntegrationTest/test.h"

#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

void no_stack_smashing_normal_exit() {
  pid_t pid = LIBC_NAMESPACE::fork();
  if (pid == 0) {
    // Child process
    char foo[30];
    for (int i = 0; i < 30; i++)
      foo[i] = (foo[i] != 42) ? 42 : 24;
    return;
  }
  ASSERT_TRUE(pid > 0);
  int status;
  pid_t cpid = LIBC_NAMESPACE::wait(&status);
  ASSERT_TRUE(cpid > 0);
  ASSERT_EQ(cpid, pid);
  ASSERT_TRUE(WIFEXITED(status));
}

void stack_smashing_abort() {
  pid_t pid = LIBC_NAMESPACE::fork();
  if (pid == 0) {
    // Child process
    char foo[30];
    char *frame_ptr = static_cast<char *>(__builtin_frame_address(0));
    char *cur_ptr = &foo[0];
    // Corrupt the stack
    while (cur_ptr != frame_ptr) {
      *cur_ptr = (*cur_ptr != 42) ? 42 : 24;
      cur_ptr++;
    }
    return;
  }
  ASSERT_TRUE(pid > 0);
  int status;
  pid_t cpid = LIBC_NAMESPACE::wait(&status);
  ASSERT_TRUE(cpid > 0);
  ASSERT_EQ(cpid, pid);
  ASSERT_TRUE(WTERMSIG(status) == SIGABRT);
}

TEST_MAIN(int argc, char **argv, char **envp) {
  no_stack_smashing_normal_exit();
  stack_smashing_abort();
  return 0;
}

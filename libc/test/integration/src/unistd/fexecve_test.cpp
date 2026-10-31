//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Integration test for fexecve.
///
//===----------------------------------------------------------------------===//

#include "src/fcntl/open.h"
#include "src/sys/wait/waitpid.h"
#include "src/unistd/close.h"
#include "src/unistd/fexecve.h"
#include "src/unistd/fork.h"

#include "hdr/fcntl_macros.h"
#include "hdr/signal_macros.h"
#include "hdr/sys_wait_macros.h"
#include "test/IntegrationTest/test.h"

void fork_and_fexecve_normal_exit(char **envp) {
  int fd = LIBC_NAMESPACE::open("libc_execv_test_normal_exit", O_RDONLY);
  ASSERT_TRUE(fd > 0);
  pid_t pid = LIBC_NAMESPACE::fork();
  if (pid == 0) {
    char *const argv[] = {
        const_cast<char *>("execv_test_normal_exit"),
        nullptr,
    };
    LIBC_NAMESPACE::fexecve(fd, argv, envp);
  }
  LIBC_NAMESPACE::close(fd);
  ASSERT_TRUE(pid > 0);
  int status;
  pid_t cpid = LIBC_NAMESPACE::waitpid(pid, &status, 0);
  ASSERT_TRUE(cpid > 0);
  ASSERT_EQ(cpid, pid);
  ASSERT_TRUE(WIFEXITED(status));
}

void fork_and_fexecve_signal_exit(char **envp) {
  int fd = LIBC_NAMESPACE::open("libc_exec_test_signal_exit", O_RDONLY);
  ASSERT_TRUE(fd > 0);
  pid_t pid = LIBC_NAMESPACE::fork();
  if (pid == 0) {
    char *const argv[] = {
        const_cast<char *>("exec_test_signal_exit"),
        nullptr,
    };
    LIBC_NAMESPACE::fexecve(fd, argv, envp);
  }
  LIBC_NAMESPACE::close(fd);
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
  fork_and_fexecve_normal_exit(envp);
  fork_and_fexecve_signal_exit(envp);
  return 0;
}

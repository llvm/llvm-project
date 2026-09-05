//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Integration tests for execl
///
//===----------------------------------------------------------------------===//

#include "src/sys/wait/waitpid.h"
#include "src/unistd/environ.h"
#include "src/unistd/execl.h"
#include "src/unistd/fork.h"

#include "hdr/signal_macros.h"
#include "hdr/sys_wait_macros.h"
#include "test/IntegrationTest/test.h"

void fork_and_execl_normal_exit(char **envp) {
  pid_t pid = LIBC_NAMESPACE::fork();
  if (pid == 0) {
    LIBC_NAMESPACE::environ = envp;
    const char *path = "libc_execl_test_normal_exit";
    LIBC_NAMESPACE::execl(path, const_cast<char *>("execl_test_normal_exit"),
                          const_cast<char *>("first"),
                          const_cast<char *>("second"),
                          static_cast<char *>(nullptr));
  }

  ASSERT_TRUE(pid > 0);
  int status;
  pid_t cpid = LIBC_NAMESPACE::waitpid(pid, &status, 0);
  ASSERT_EQ(cpid, pid);
  ASSERT_TRUE(WIFEXITED(status));
}

void fork_and_execl_custom_env() {
  pid_t pid = LIBC_NAMESPACE::fork();
  if (pid == 0) {
    char *custom_env[] = {const_cast<char *>("EXECL_TEST=PASS"), nullptr};
    LIBC_NAMESPACE::environ = custom_env;
    const char *path = "libc_execl_test_normal_exit";
    LIBC_NAMESPACE::execl(path, const_cast<char *>("execl_test_normal_exit"),
                          const_cast<char *>("first"),
                          const_cast<char *>("second"),
                          static_cast<char *>(nullptr));
  }

  ASSERT_TRUE(pid > 0);
  int status;
  pid_t cpid = LIBC_NAMESPACE::waitpid(pid, &status, 0);
  ASSERT_EQ(cpid, pid);
  ASSERT_TRUE(WIFEXITED(status));
}

void fork_and_execl_missing_env_fails() {
  pid_t pid = LIBC_NAMESPACE::fork();
  if (pid == 0) {
    char *missing_env[] = {const_cast<char *>("OTHER_VAR=1"), nullptr};
    LIBC_NAMESPACE::environ = missing_env;
    const char *path = "libc_execl_test_normal_exit";
    LIBC_NAMESPACE::execl(path, const_cast<char *>("execl_test_normal_exit"),
                          const_cast<char *>("first"),
                          const_cast<char *>("second"),
                          static_cast<char *>(nullptr));
  }

  ASSERT_TRUE(pid > 0);
  int status;
  pid_t cpid = LIBC_NAMESPACE::waitpid(pid, &status, 0);
  ASSERT_EQ(cpid, pid);
  ASSERT_FALSE(WIFEXITED(status));
  ASSERT_TRUE(WTERMSIG(status) == SIGUSR1);
}

void fork_and_execl_signal_exit() {
  pid_t pid = LIBC_NAMESPACE::fork();
  if (pid == 0) {
    const char *path = "libc_exec_test_signal_exit";
    LIBC_NAMESPACE::execl(path, const_cast<char *>("exec_test_signal_exit"),
                          static_cast<char *>(nullptr));
  }
  ASSERT_TRUE(pid > 0);
  int status;
  pid_t cpid = LIBC_NAMESPACE::waitpid(pid, &status, 0);
  ASSERT_EQ(cpid, pid);
  ASSERT_FALSE(WIFEXITED(status));
  ASSERT_TRUE(WTERMSIG(status) == SIGUSR1);
}

TEST_MAIN([[maybe_unused]] int argc, [[maybe_unused]] char **argv,
          char **envp) {
  fork_and_execl_normal_exit(envp);
  fork_and_execl_custom_env();
  fork_and_execl_missing_env_fails();
  fork_and_execl_signal_exit();
  return 0;
}

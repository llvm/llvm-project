//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Integration tests for execlp
///
//===----------------------------------------------------------------------===//

#include "src/stdlib/setenv.h"
#include "src/sys/wait/waitpid.h"
#include "src/unistd/_exit.h"
#include "src/unistd/execlp.h"
#include "src/unistd/fork.h"

#include "hdr/signal_macros.h"
#include "hdr/sys_wait_macros.h"
#include "test/IntegrationTest/test.h"

#include <errno.h>

void fork_and_execlp_slash_path() {
  pid_t pid = LIBC_NAMESPACE::fork();
  if (pid == 0) {
    const char *path = "./libc_execv_test_normal_exit";
    LIBC_NAMESPACE::execlp(path, "execv_test_normal_exit",
                           static_cast<char *>(nullptr));
    LIBC_NAMESPACE::_exit(errno);
  }
  ASSERT_TRUE(pid > 0);
  int status;
  pid_t cpid = LIBC_NAMESPACE::waitpid(pid, &status, 0);
  ASSERT_TRUE(cpid > 0);
  ASSERT_EQ(cpid, pid);
  ASSERT_TRUE(WIFEXITED(status));
  ASSERT_EQ(WEXITSTATUS(status), 0);
}

void fork_and_execlp_path_search() {
  pid_t pid = LIBC_NAMESPACE::fork();
  if (pid == 0) {
    ASSERT_EQ(LIBC_NAMESPACE::setenv(
                  "PATH", "/nonexistent_dir_for_execlp_test:testdata", 1),
              0);
    const char *file = "libc_execvp_test_normal_exit";
    LIBC_NAMESPACE::execlp(file, "execvp_test_normal_exit",
                           static_cast<char *>(nullptr));
    LIBC_NAMESPACE::_exit(errno);
  }
  ASSERT_TRUE(pid > 0);
  int status;
  pid_t cpid = LIBC_NAMESPACE::waitpid(pid, &status, 0);
  ASSERT_TRUE(cpid > 0);
  ASSERT_EQ(cpid, pid);
  ASSERT_TRUE(WIFEXITED(status));
  ASSERT_EQ(WEXITSTATUS(status), 0);
}

void execlp_missing_in_path() {
  ASSERT_EQ(
      LIBC_NAMESPACE::setenv("PATH", "/nonexistent_dir_for_execlp_test", 1), 0);
  const char *file = "libc_execvp_test_normal_exit";
  errno = 0;
  ASSERT_EQ(LIBC_NAMESPACE::execlp(file, "execvp_test_normal_exit",
                                   static_cast<char *>(nullptr)),
            -1);
  ASSERT_ERRNO_EQ(ENOENT);
}

void execlp_nonexistent_file() {
  ASSERT_EQ(LIBC_NAMESPACE::setenv("PATH", "testdata", 1), 0);
  const char *file = "libc_execlp_nonexistent_file";
  errno = 0;
  ASSERT_EQ(LIBC_NAMESPACE::execlp(file, "execlp_nonexistent_file",
                                   static_cast<char *>(nullptr)),
            -1);
  ASSERT_ERRNO_EQ(ENOENT);
}

// testdata/noexec contains a file with the same name as the helper binary but
// with no execute permission bits, so execve on it fails with EACCES.
void fork_and_execlp_skips_eacces() {
  pid_t pid = LIBC_NAMESPACE::fork();
  if (pid == 0) {
    ASSERT_EQ(LIBC_NAMESPACE::setenv("PATH", "testdata/noexec:testdata", 1), 0);
    const char *file = "libc_execvp_test_normal_exit";
    LIBC_NAMESPACE::execlp(file, "execvp_test_normal_exit",
                           static_cast<char *>(nullptr));
    LIBC_NAMESPACE::_exit(errno);
  }
  ASSERT_TRUE(pid > 0);
  int status;
  pid_t cpid = LIBC_NAMESPACE::waitpid(pid, &status, 0);
  ASSERT_TRUE(cpid > 0);
  ASSERT_EQ(cpid, pid);
  ASSERT_TRUE(WIFEXITED(status));
  ASSERT_EQ(WEXITSTATUS(status), 0);
}

void execlp_reports_eacces() {
  ASSERT_EQ(LIBC_NAMESPACE::setenv(
                "PATH", "testdata/noexec:/nonexistent_dir_for_execlp_test", 1),
            0);
  const char *file = "libc_execvp_test_normal_exit";
  errno = 0;
  ASSERT_EQ(LIBC_NAMESPACE::execlp(file, "execvp_test_normal_exit",
                                   static_cast<char *>(nullptr)),
            -1);
  ASSERT_ERRNO_EQ(EACCES);
}

void fork_and_execlp_signal_exit() {
  pid_t pid = LIBC_NAMESPACE::fork();
  if (pid == 0) {
    const char *path = "./libc_exec_test_signal_exit";
    LIBC_NAMESPACE::execlp(path, "exec_test_signal_exit",
                           static_cast<char *>(nullptr));
    LIBC_NAMESPACE::_exit(errno);
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
  fork_and_execlp_slash_path();
  fork_and_execlp_path_search();
  fork_and_execlp_skips_eacces();
  execlp_missing_in_path();
  execlp_nonexistent_file();
  execlp_reports_eacces();
  fork_and_execlp_signal_exit();
  return 0;
}

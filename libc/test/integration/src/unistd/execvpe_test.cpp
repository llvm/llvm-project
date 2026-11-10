//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Integration tests for execvpe
///
//===----------------------------------------------------------------------===//

#include "src/stdlib/setenv.h"
#include "src/sys/wait/waitpid.h"
#include "src/unistd/_exit.h"
#include "src/unistd/execvpe.h"
#include "src/unistd/fork.h"

#include "hdr/signal_macros.h"
#include "hdr/sys_wait_macros.h"
#include "test/IntegrationTest/test.h"

#include <errno.h>

void fork_and_execvpe_slash_path() {
  pid_t pid = LIBC_NAMESPACE::fork();
  if (pid == 0) {
    const char *path = "./libc_execv_test_normal_exit";
    char *const argv[] = {
        const_cast<char *>("execv_test_normal_exit"),
        nullptr,
    };
    char *const envp[] = {
        const_cast<char *>("EXECV_TEST=PASS"),
        nullptr,
    };
    LIBC_NAMESPACE::execvpe(path, argv, envp);
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

void fork_and_execvpe_path_search() {
  pid_t pid = LIBC_NAMESPACE::fork();
  if (pid == 0) {
    ASSERT_EQ(LIBC_NAMESPACE::setenv(
                  "PATH", "/nonexistent_dir_for_execvp_test:testdata", 1),
              0);
    const char *file = "libc_execvp_test_normal_exit";
    char *const argv[] = {
        const_cast<char *>("execvp_test_normal_exit"),
        nullptr,
    };
    char *const envp[] = {
        const_cast<char *>("EXECV_TEST=PASS"),
        nullptr,
    };
    LIBC_NAMESPACE::execvpe(file, argv, envp);
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

// Linux / GNU execvpe() searches for the program using the value of PATH
// from the caller's environment, not from the envp argument (also standardized
// in POSIX.1-2024).
void fork_and_execvpe_caller_path_not_envp() {
  pid_t pid = LIBC_NAMESPACE::fork();
  if (pid == 0) {
    ASSERT_EQ(LIBC_NAMESPACE::setenv("PATH", "testdata", 1), 0);
    const char *file = "libc_execvp_test_normal_exit";
    char *const argv[] = {
        const_cast<char *>("execvp_test_normal_exit"),
        nullptr,
    };
    // envp has a bogus PATH, but execvpe must ignore it and use caller's PATH.
    char *const envp[] = {
        const_cast<char *>("PATH=/nonexistent_dir_for_execvpe_test"),
        const_cast<char *>("EXECV_TEST=PASS"),
        nullptr,
    };
    LIBC_NAMESPACE::execvpe(file, argv, envp);
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

// testdata/noexec contains a file with the same name as the helper binary but
// with no execute permission bits, so execve on it fails with EACCES.
void fork_and_execvpe_skips_eacces() {
  pid_t pid = LIBC_NAMESPACE::fork();
  if (pid == 0) {
    ASSERT_EQ(LIBC_NAMESPACE::setenv("PATH", "testdata/noexec:testdata", 1), 0);
    const char *file = "libc_execvp_test_normal_exit";
    char *const argv[] = {
        const_cast<char *>("execvp_test_normal_exit"),
        nullptr,
    };
    char *const envp[] = {
        const_cast<char *>("EXECV_TEST=PASS"),
        nullptr,
    };
    LIBC_NAMESPACE::execvpe(file, argv, envp);
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

void execvpe_missing_in_path() {
  ASSERT_EQ(
      LIBC_NAMESPACE::setenv("PATH", "/nonexistent_dir_for_execvp_test", 1), 0);
  const char *file = "libc_execvp_test_normal_exit";
  char *const argv[] = {
      const_cast<char *>("execvp_test_normal_exit"),
      nullptr,
  };
  char *const envp[] = {
      nullptr,
  };
  errno = 0;
  ASSERT_EQ(LIBC_NAMESPACE::execvpe(file, argv, envp), -1);
  ASSERT_ERRNO_EQ(ENOENT);
}

void execvpe_nonexistent_file() {
  ASSERT_EQ(LIBC_NAMESPACE::setenv("PATH", "testdata", 1), 0);
  const char *file = "libc_execvp_nonexistent_file";
  char *const argv[] = {
      const_cast<char *>("execvp_nonexistent_file"),
      nullptr,
  };
  char *const envp[] = {
      nullptr,
  };
  errno = 0;
  ASSERT_EQ(LIBC_NAMESPACE::execvpe(file, argv, envp), -1);
  ASSERT_ERRNO_EQ(ENOENT);
}

void execvpe_reports_eacces() {
  ASSERT_EQ(LIBC_NAMESPACE::setenv(
                "PATH", "testdata/noexec:/nonexistent_dir_for_execvp_test", 1),
            0);
  const char *file = "libc_execvp_test_normal_exit";
  char *const argv[] = {
      const_cast<char *>("execvp_test_normal_exit"),
      nullptr,
  };
  char *const envp[] = {
      nullptr,
  };
  errno = 0;
  ASSERT_EQ(LIBC_NAMESPACE::execvpe(file, argv, envp), -1);
  ASSERT_ERRNO_EQ(EACCES);
}

TEST_MAIN([[maybe_unused]] int argc, [[maybe_unused]] char **argv,
          [[maybe_unused]] char **envp) {
  fork_and_execvpe_slash_path();
  fork_and_execvpe_path_search();
  fork_and_execvpe_caller_path_not_envp();
  fork_and_execvpe_skips_eacces();
  execvpe_missing_in_path();
  execvpe_nonexistent_file();
  execvpe_reports_eacces();
  return 0;
}

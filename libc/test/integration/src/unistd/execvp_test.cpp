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

#include "src/fcntl/open.h"
#include "src/sys/wait/waitpid.h"
#include "src/unistd/close.h"
#include "src/unistd/execvp.h"
#include "src/unistd/fork.h"
#include "src/unistd/unlink.h"
#include "src/unistd/write.h"

#include "hdr/errno_macros.h"
#include "hdr/fcntl_macros.h"
#include "hdr/signal_macros.h"
#include "hdr/sys_wait_macros.h"
#include "hdr/types/ssize_t.h"
#include "test/IntegrationTest/test.h"

#include <errno.h>

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

void execvp_eacces() {
  const char *no_exec_file = "libc_execvp_test_no_exec.test";
  int fd =
      LIBC_NAMESPACE::open(no_exec_file, O_CREAT | O_WRONLY | O_TRUNC, 0644);
  ASSERT_TRUE(fd > 0);
  LIBC_NAMESPACE::close(fd);

  char *const argv[] = {
      const_cast<char *>(no_exec_file),
      nullptr,
  };

  errno = 0;
  ASSERT_EQ(LIBC_NAMESPACE::execvp("./libc_execvp_test_no_exec.test", argv),
            -1);
  ASSERT_ERRNO_EQ(EACCES);

  errno = 0;
  ASSERT_EQ(LIBC_NAMESPACE::execvp(no_exec_file, argv), -1);
  ASSERT_ERRNO_EQ(EACCES);

  LIBC_NAMESPACE::unlink(no_exec_file);
}

void fork_and_execvp_enoexec_fallback() {
  const char *script_file = "libc_execvp_test_script.test";
  int fd =
      LIBC_NAMESPACE::open(script_file, O_CREAT | O_WRONLY | O_TRUNC, 0777);
  ASSERT_TRUE(fd > 0);
  const char content[] = "exit 0\n";
  ASSERT_EQ(LIBC_NAMESPACE::write(fd, content, sizeof(content) - 1),
            static_cast<ssize_t>(sizeof(content) - 1));
  LIBC_NAMESPACE::close(fd);

  // Test slash path with fallback to shell.
  pid_t pid = LIBC_NAMESPACE::fork();
  if (pid == 0) {
    char *const argv[] = {
        const_cast<char *>(script_file),
        nullptr,
    };
    LIBC_NAMESPACE::execvp("./libc_execvp_test_script.test", argv);
  }
  ASSERT_TRUE(pid > 0);
  int status;
  pid_t cpid = LIBC_NAMESPACE::waitpid(pid, &status, 0);
  ASSERT_TRUE(cpid > 0);
  ASSERT_EQ(cpid, pid);
  ASSERT_TRUE(WIFEXITED(status));
  ASSERT_EQ(WEXITSTATUS(status), 0);

  // Test PATH search with fallback to shell.
  pid = LIBC_NAMESPACE::fork();
  if (pid == 0) {
    char *const argv[] = {
        const_cast<char *>(script_file),
        nullptr,
    };
    LIBC_NAMESPACE::execvp(script_file, argv);
  }
  ASSERT_TRUE(pid > 0);
  cpid = LIBC_NAMESPACE::waitpid(pid, &status, 0);
  ASSERT_TRUE(cpid > 0);
  ASSERT_EQ(cpid, pid);
  ASSERT_TRUE(WIFEXITED(status));
  ASSERT_EQ(WEXITSTATUS(status), 0);

  LIBC_NAMESPACE::unlink(script_file);
}

TEST_MAIN([[maybe_unused]] int argc, [[maybe_unused]] char **argv,
          [[maybe_unused]] char **envp) {
  fork_and_execvp_path_search();
  fork_and_execvp_slash_path();
  fork_and_execvp_signal_exit();
  execvp_eacces();
  fork_and_execvp_enoexec_fallback();
  return 0;
}

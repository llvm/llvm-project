//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Round trip test for the wait status macros.
///
//===----------------------------------------------------------------------===//

#include "hdr/signal_macros.h"
#include "hdr/sys_wait_macros.h"
#include "hdr/types/pid_t.h"
#include "hdr/types/ssize_t.h"
#include "src/__support/OSUtil/exit.h"
#include "src/signal/kill.h"
#include "src/signal/raise.h"
#include "src/sys/wait/waitpid.h"
#include "src/unistd/close.h"
#include "src/unistd/fork.h"
#include "src/unistd/pipe.h"
#include "src/unistd/read.h"

#include "test/IntegrationTest/test.h"

// Decodes statuses produced by the kernel, and checks that exactly one of the
// predicates is true for each.

void exited() {
  pid_t pid = LIBC_NAMESPACE::fork();
  if (pid == 0)
    LIBC_NAMESPACE::internal::exit(3);
  ASSERT_TRUE(pid > 0);
  int status = 0;
  ASSERT_EQ(LIBC_NAMESPACE::waitpid(pid, &status, 0), pid);
  ASSERT_TRUE(WIFEXITED(status));
  ASSERT_EQ(WEXITSTATUS(status), 3);
  ASSERT_FALSE(WIFSIGNALED(status));
  ASSERT_FALSE(WIFSTOPPED(status));
  ASSERT_FALSE(WIFCONTINUED(status));
}

void signaled() {
  pid_t pid = LIBC_NAMESPACE::fork();
  if (pid == 0) {
    LIBC_NAMESPACE::raise(SIGTERM);
    LIBC_NAMESPACE::internal::exit(0);
  }
  ASSERT_TRUE(pid > 0);
  int status = 0;
  ASSERT_EQ(LIBC_NAMESPACE::waitpid(pid, &status, 0), pid);
  ASSERT_TRUE(WIFSIGNALED(status));
  ASSERT_EQ(WTERMSIG(status), SIGTERM);
  ASSERT_FALSE(WIFEXITED(status));
  ASSERT_FALSE(WIFSTOPPED(status));
  ASSERT_FALSE(WIFCONTINUED(status));
}

void stopped_and_continued() {
  int pipefd[2];
  ASSERT_EQ(LIBC_NAMESPACE::pipe(pipefd), 0);

  pid_t pid = LIBC_NAMESPACE::fork();
  if (pid == 0) {
    ASSERT_EQ(LIBC_NAMESPACE::close(pipefd[1]), 0);
    LIBC_NAMESPACE::raise(SIGSTOP);
    char c = 0;
    ASSERT_EQ(LIBC_NAMESPACE::read(pipefd[0], &c, 1), ssize_t(0));
    ASSERT_EQ(LIBC_NAMESPACE::close(pipefd[0]), 0);
    LIBC_NAMESPACE::internal::exit(0);
  }
  ASSERT_TRUE(pid > 0);
  ASSERT_EQ(LIBC_NAMESPACE::close(pipefd[0]), 0);

  int status = 0;
  ASSERT_EQ(LIBC_NAMESPACE::waitpid(pid, &status, WUNTRACED), pid);
  ASSERT_TRUE(WIFSTOPPED(status));
  ASSERT_EQ(WSTOPSIG(status), SIGSTOP);
  ASSERT_FALSE(WIFEXITED(status));
  ASSERT_FALSE(WIFSIGNALED(status));
  ASSERT_FALSE(WIFCONTINUED(status));

  ASSERT_EQ(LIBC_NAMESPACE::kill(pid, SIGCONT), 0);
  ASSERT_EQ(LIBC_NAMESPACE::waitpid(pid, &status, WCONTINUED), pid);
  ASSERT_TRUE(WIFCONTINUED(status));
  ASSERT_FALSE(WIFEXITED(status));
  ASSERT_FALSE(WIFSIGNALED(status));
  ASSERT_FALSE(WIFSTOPPED(status));

  ASSERT_EQ(LIBC_NAMESPACE::close(pipefd[1]), 0);
  ASSERT_EQ(LIBC_NAMESPACE::waitpid(pid, &status, 0), pid);
  ASSERT_TRUE(WIFEXITED(status));
  ASSERT_EQ(WEXITSTATUS(status), 0);
}

TEST_MAIN([[maybe_unused]] int argc, [[maybe_unused]] char **argv,
          [[maybe_unused]] char **envp) {
  exited();
  signaled();
  stopped_and_continued();
  return 0;
}

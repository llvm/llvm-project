//===-- Integration tests for abort --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/signal/signal.h"
#include "src/stdlib/_Exit.h"
#include "src/stdlib/abort.h"
#include "src/stdlib/linux/abort_utils.h"
#include "src/sys/wait/waitpid.h"
#include "src/unistd/close.h"
#include "src/unistd/fork.h"
#include "src/unistd/pipe.h"
#include "src/unistd/read.h"
#include "src/unistd/write.h"

#include "test/IntegrationTest/test.h"

#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

constexpr char HANDLER_MARKER = 'A';
int handler_pipe_fd = -1;

void expect_child_died_with_signal(pid_t pid, int signal) {
  int status = 0;
  ASSERT_EQ(LIBC_NAMESPACE::waitpid(pid, &status, 0), pid);
  ASSERT_TRUE(WIFSIGNALED(status));
  ASSERT_EQ(WTERMSIG(status), signal);
}

void child_abort() { LIBC_NAMESPACE::abort(); }

void returning_sigabrt_handler(int) {
  if (handler_pipe_fd >= 0)
    LIBC_NAMESPACE::write(handler_pipe_fd, &HANDLER_MARKER, 1);
}

void child_abort_with_returning_handler() {
  auto previous = LIBC_NAMESPACE::signal(SIGABRT, returning_sigabrt_handler);
  ASSERT_NE(previous, SIG_ERR);
  LIBC_NAMESPACE::abort();
}

void abort_kills_child_with_sigabrt() {
  pid_t pid = LIBC_NAMESPACE::fork();
  if (pid == 0)
    child_abort();

  ASSERT_TRUE(pid > 0);
  expect_child_died_with_signal(pid, SIGABRT);
}

void abort_reraises_sigabrt_after_returning_handler() {
  int pipefd[2];
  ASSERT_EQ(LIBC_NAMESPACE::pipe(pipefd), 0);

  pid_t pid = LIBC_NAMESPACE::fork();
  if (pid == 0) {
    LIBC_NAMESPACE::close(pipefd[0]);
    handler_pipe_fd = pipefd[1];
    child_abort_with_returning_handler();
  }

  ASSERT_TRUE(pid > 0);
  ASSERT_EQ(LIBC_NAMESPACE::close(pipefd[1]), 0);

  expect_child_died_with_signal(pid, SIGABRT);

  char marker = 0;
  ASSERT_EQ(LIBC_NAMESPACE::read(pipefd[0], &marker, 1), ssize_t(1));
  ASSERT_EQ(marker, HANDLER_MARKER);
  ASSERT_EQ(LIBC_NAMESPACE::close(pipefd[0]), 0);
}

} // namespace

TEST_MAIN([[maybe_unused]] int argc, [[maybe_unused]] char **argv,
          [[maybe_unused]] char **envp) {
  abort_kills_child_with_sigabrt();
  abort_reraises_sigabrt_after_returning_handler();
  return 0;
}

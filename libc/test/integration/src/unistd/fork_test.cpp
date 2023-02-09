//===-- Unittests for fork ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/pthread/pthread_atfork.h"
#include "src/signal/raise.h"
#include "src/sys/wait/wait.h"
#include "src/sys/wait/wait4.h"
#include "src/sys/wait/waitpid.h"
#include "src/unistd/fork.h"

#include "test/IntegrationTest/test.h"

#include <errno.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

// The tests wait4 and waitpid are present as tests for those functions
// really and not for the fork function. They are here along with the tests
// for fork because it is convenient to invoke and test them after forking
// a child.

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

void fork_and_wait4_normal_exit() {
  pid_t pid = __llvm_libc::fork();
  if (pid == 0)
    return; // Just end without any thing special.
  ASSERT_TRUE(pid > 0);
  int status;
  struct rusage usage;
  usage.ru_utime = {0, 0};
  usage.ru_stime = {0, 0};
  pid_t cpid = __llvm_libc::wait4(pid, &status, 0, &usage);
  ASSERT_TRUE(cpid > 0);
  ASSERT_EQ(cpid, pid);
  ASSERT_TRUE(WIFEXITED(status));
}

void fork_and_waitpid_normal_exit() {
  pid_t pid = __llvm_libc::fork();
  if (pid == 0)
    return; // Just end without any thing special.
  ASSERT_TRUE(pid > 0);
  int status;
  pid_t cpid = __llvm_libc::waitpid(pid, &status, 0);
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

void fork_and_wait4_signal_exit() {
  pid_t pid = __llvm_libc::fork();
  if (pid == 0)
    __llvm_libc::raise(SIGUSR1);
  ASSERT_TRUE(pid > 0);
  int status;
  struct rusage usage;
  usage.ru_utime = {0, 0};
  usage.ru_stime = {0, 0};
  pid_t cpid = __llvm_libc::wait4(pid, &status, 0, &usage);
  ASSERT_TRUE(cpid > 0);
  ASSERT_EQ(cpid, pid);
  ASSERT_FALSE(WIFEXITED(status));
  ASSERT_TRUE(WTERMSIG(status) == SIGUSR1);
}

void fork_and_waitpid_signal_exit() {
  pid_t pid = __llvm_libc::fork();
  if (pid == 0)
    __llvm_libc::raise(SIGUSR1);
  ASSERT_TRUE(pid > 0);
  int status;
  pid_t cpid = __llvm_libc::waitpid(pid, &status, 0);
  ASSERT_TRUE(cpid > 0);
  ASSERT_EQ(cpid, pid);
  ASSERT_FALSE(WIFEXITED(status));
  ASSERT_TRUE(WTERMSIG(status) == SIGUSR1);
}

static int prepare = 0;
static int parent = 0;
static int child = 0;
static constexpr int DONE = 0x600D;

static void prepare_cb() { prepare = DONE; }

static void parent_cb() { parent = DONE; }

static void child_cb() { child = DONE; }

void fork_with_atfork_callbacks() {
  ASSERT_EQ(__llvm_libc::pthread_atfork(&prepare_cb, &parent_cb, &child_cb), 0);
  pid_t pid = __llvm_libc::fork();
  if (pid == 0) {
    // Raise a signal from the child if unexpected at-fork
    // behavior is observed.
    if (child != DONE || prepare != DONE || parent == DONE)
      __llvm_libc::raise(SIGUSR1);
    return;
  }

  ASSERT_TRUE(pid > 0);
  int status;
  pid_t cpid = __llvm_libc::waitpid(pid, &status, 0);
  ASSERT_TRUE(cpid > 0);
  ASSERT_EQ(cpid, pid);
  ASSERT_TRUE(WIFEXITED(status));
  ASSERT_EQ(prepare, DONE);
  ASSERT_EQ(parent, DONE);
  ASSERT_NE(child, DONE);
}

TEST_MAIN(int argc, char **argv, char **envp) {
  fork_and_wait_normal_exit();
  fork_and_wait4_normal_exit();
  fork_and_waitpid_normal_exit();
  fork_and_wait_signal_exit();
  fork_and_wait4_signal_exit();
  fork_and_waitpid_signal_exit();
  fork_with_atfork_callbacks();
  return 0;
}

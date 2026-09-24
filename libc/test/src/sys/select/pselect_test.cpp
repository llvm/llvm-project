//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for pselect.
///
//===----------------------------------------------------------------------===//

#include "hdr/signal_macros.h"
#include "hdr/sys_select_macros.h"
#include "hdr/sys_time_macros.h"
#include "hdr/types/fd_set.h"
#include "hdr/types/sigset_t.h"
#include "hdr/types/ssize_t.h"
#include "hdr/types/struct_itimerval.h"
#include "hdr/types/struct_sigaction.h"
#include "hdr/types/struct_timespec.h"
#include "src/__support/CPP/scope.h"
#include "src/signal/raise.h"
#include "src/signal/sigaction.h"
#include "src/signal/sigaddset.h"
#include "src/signal/sigemptyset.h"
#include "src/signal/sigprocmask.h"
#include "src/sys/select/pselect.h"
#include "src/sys/time/setitimer.h"
#include "src/unistd/close.h"
#include "src/unistd/pipe.h"
#include "src/unistd/read.h"
#include "src/unistd/write.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

using LlvmLibcPSelectTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

static bool sigalrm_handler_called = false;
extern "C" void handle_sigalrm(int) { sigalrm_handler_called = true; }

static bool sigusr1_handler_called = false;
extern "C" void handle_sigusr1(int) { sigusr1_handler_called = true; }

TEST_F(LlvmLibcPSelectTest, SmokeTest) {
  timespec ts{0, 0};
  int ret = LIBC_NAMESPACE::pselect(0, nullptr, nullptr, nullptr, &ts, nullptr);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(0, ret);
}

TEST_F(LlvmLibcPSelectTest, SmokeFailureTest) {
  timespec ts{0, 0};
  int ret =
      LIBC_NAMESPACE::pselect(-1, nullptr, nullptr, nullptr, &ts, nullptr);
  ASSERT_EQ(-1, ret);
  ASSERT_ERRNO_EQ(EINVAL);
}

TEST_F(LlvmLibcPSelectTest, TimeoutNotMutated) {
  sigalrm_handler_called = false;
  struct sigaction sa{};
  sa.sa_handler = handle_sigalrm;
  LIBC_NAMESPACE::sigemptyset(&sa.sa_mask);
  sa.sa_flags = 0;
  struct sigaction old_sa{};
  ASSERT_EQ(LIBC_NAMESPACE::sigaction(SIGALRM, &sa, &old_sa), 0);

  LIBC_NAMESPACE::cpp::scope_exit restore_sa([&] {
    LIBC_NAMESPACE::sigaction(SIGALRM, &old_sa, nullptr);
    struct itimerval disable_timer{};
    LIBC_NAMESPACE::setitimer(ITIMER_REAL, &disable_timer, nullptr);
  });

  struct itimerval timer{};
  timer.it_value.tv_sec = 0;
  timer.it_value.tv_usec = 100000; // 100ms
  ASSERT_EQ(LIBC_NAMESPACE::setitimer(ITIMER_REAL, &timer, nullptr), 0);

  const timespec ORIG_TS{1, 0}; // 1 second
  timespec ts = ORIG_TS;
  int ret = LIBC_NAMESPACE::pselect(0, nullptr, nullptr, nullptr, &ts, nullptr);
  ASSERT_EQ(-1, ret);
  ASSERT_ERRNO_EQ(EINTR);
  ASSERT_TRUE(sigalrm_handler_called);

  // The Linux raw syscall modifies its timeout argument when interrupted by a
  // signal, but POSIX requires that pselect does not modify it. Verify that the
  // timeout argument was not modified.
  ASSERT_EQ(ts.tv_sec, ORIG_TS.tv_sec);
  ASSERT_EQ(ts.tv_nsec, ORIG_TS.tv_nsec);
}

TEST_F(LlvmLibcPSelectTest, WithSigmask) {
  sigusr1_handler_called = false;
  struct sigaction sa{};
  sa.sa_handler = handle_sigusr1;
  LIBC_NAMESPACE::sigemptyset(&sa.sa_mask);
  sa.sa_flags = 0;
  struct sigaction old_sa{};
  ASSERT_EQ(LIBC_NAMESPACE::sigaction(SIGUSR1, &sa, &old_sa), 0);

  sigset_t block_mask{};
  LIBC_NAMESPACE::sigemptyset(&block_mask);
  LIBC_NAMESPACE::sigaddset(&block_mask, SIGUSR1);
  sigset_t orig_mask{};
  ASSERT_EQ(LIBC_NAMESPACE::sigprocmask(SIG_BLOCK, &block_mask, &orig_mask), 0);

  LIBC_NAMESPACE::cpp::scope_exit cleanup([&] {
    LIBC_NAMESPACE::sigprocmask(SIG_SETMASK, &orig_mask, nullptr);
    LIBC_NAMESPACE::sigaction(SIGUSR1, &old_sa, nullptr);
  });

  // Raise SIGUSR1 while it is blocked.
  ASSERT_EQ(LIBC_NAMESPACE::raise(SIGUSR1), 0);
  ASSERT_FALSE(sigusr1_handler_called);

  // Call pselect with a mask that unblocks SIGUSR1.
  sigset_t unblock_mask{};
  LIBC_NAMESPACE::sigemptyset(&unblock_mask);
  timespec ts{1, 0};
  int ret =
      LIBC_NAMESPACE::pselect(0, nullptr, nullptr, nullptr, &ts, &unblock_mask);
  ASSERT_EQ(-1, ret);
  ASSERT_ERRNO_EQ(EINTR);
  ASSERT_TRUE(sigusr1_handler_called);
}

TEST_F(LlvmLibcPSelectTest, PipeReadiness) {
  int pipefd[2];
  ASSERT_EQ(LIBC_NAMESPACE::pipe(pipefd), 0);

  LIBC_NAMESPACE::cpp::scope_exit close_pipe([&] {
    LIBC_NAMESPACE::close(pipefd[0]);
    LIBC_NAMESPACE::close(pipefd[1]);
  });

  fd_set read_set;
  FD_ZERO(&read_set);
  FD_SET(pipefd[0], &read_set);

  timespec zero{0, 0};
  int ret = LIBC_NAMESPACE::pselect(pipefd[0] + 1, &read_set, nullptr, nullptr,
                                    &zero, nullptr);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(0, ret);
  ASSERT_EQ(0, FD_ISSET(pipefd[0], &read_set));

  constexpr char MSG = 'x';
  ASSERT_EQ(LIBC_NAMESPACE::write(pipefd[1], &MSG, 1), ssize_t(1));

  FD_ZERO(&read_set);
  FD_SET(pipefd[0], &read_set);

  ret = LIBC_NAMESPACE::pselect(pipefd[0] + 1, &read_set, nullptr, nullptr,
                                &zero, nullptr);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(1, ret);
  ASSERT_NE(0, FD_ISSET(pipefd[0], &read_set));

  char buf = 0;
  ASSERT_EQ(LIBC_NAMESPACE::read(pipefd[0], &buf, 1), ssize_t(1));
  ASSERT_EQ(buf, MSG);
}

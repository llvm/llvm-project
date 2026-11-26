//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for sigwait.
///
//===----------------------------------------------------------------------===//

#include "src/signal/sigwait.h"

#include "hdr/errno_macros.h"
#include "hdr/signal_macros.h"
#include "hdr/sys_time_macros.h"
#include "hdr/types/sigset_t.h"
#include "hdr/types/struct_itimerval.h"
#include "src/signal/raise.h"
#include "src/signal/sigaddset.h"
#include "src/signal/sigemptyset.h"
#include "src/signal/sigprocmask.h"
#include "src/sys/time/setitimer.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcSigwaitTest, WaitPendingSignal) {
  // Block SIGUSR1 so raising it leaves it pending without terminating.
  sigset_t mask;
  ASSERT_EQ(LIBC_NAMESPACE::sigemptyset(&mask), 0);
  ASSERT_EQ(LIBC_NAMESPACE::sigaddset(&mask, SIGUSR1), 0);

  sigset_t oldset;
  ASSERT_EQ(LIBC_NAMESPACE::sigprocmask(SIG_BLOCK, &mask, &oldset), 0);

  // Raise SIGUSR1. It is now queued/pending for this thread.
  ASSERT_EQ(LIBC_NAMESPACE::raise(SIGUSR1), 0);

  // sigwait should immediately catch and dequeue SIGUSR1.
  int sig = 0;
  ASSERT_EQ(LIBC_NAMESPACE::sigwait(&mask, &sig), 0);
  ASSERT_EQ(sig, SIGUSR1);

  // Restore the original signal mask.
  ASSERT_EQ(LIBC_NAMESPACE::sigprocmask(SIG_SETMASK, &oldset, nullptr), 0);
}

TEST(LlvmLibcSigwaitTest, WaitForSignalDeliveredLater) {
  // Block SIGALRM so that when setitimer fires, it is queued for this thread
  // and does not terminate the process.
  sigset_t mask;
  ASSERT_EQ(LIBC_NAMESPACE::sigemptyset(&mask), 0);
  ASSERT_EQ(LIBC_NAMESPACE::sigaddset(&mask, SIGALRM), 0);

  sigset_t oldset;
  ASSERT_EQ(LIBC_NAMESPACE::sigprocmask(SIG_BLOCK, &mask, &oldset), 0);

  // Set a timer to deliver SIGALRM after 10 milliseconds.
  struct itimerval timer = {};
  timer.it_value.tv_sec = 0;
  timer.it_value.tv_usec = 10000;

  struct itimerval old_timer = {};
  ASSERT_EQ(LIBC_NAMESPACE::setitimer(ITIMER_REAL, &timer, &old_timer), 0);

  // SIGALRM is not yet pending at the time of this call.
  // sigwait should block until the timer fires and delivers the signal.
  int sig = 0;
  ASSERT_EQ(LIBC_NAMESPACE::sigwait(&mask, &sig), 0);
  ASSERT_EQ(sig, SIGALRM);

  // Restore the original timer and signal mask.
  LIBC_NAMESPACE::setitimer(ITIMER_REAL, &old_timer, nullptr);
  ASSERT_EQ(LIBC_NAMESPACE::sigprocmask(SIG_SETMASK, &oldset, nullptr), 0);
}

TEST(LlvmLibcSigwaitTest, InvalidSetReturnsEfault) {
  // We let the kernel verify the pointer value.
  int sig = 0;
  EXPECT_EQ(LIBC_NAMESPACE::sigwait(nullptr, &sig), EFAULT);
}

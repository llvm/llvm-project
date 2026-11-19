//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for pause.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/signal_macros.h"
#include "hdr/sys_time_macros.h"
#include "hdr/types/struct_itimerval.h"
#include "hdr/types/struct_sigaction.h"
#include "src/__support/CPP/scope.h"
#include "src/signal/sigaction.h"
#include "src/signal/sigemptyset.h"
#include "src/sys/time/setitimer.h"
#include "src/unistd/pause.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
using LlvmLibcPauseTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

static volatile bool alarm_fired = false;

extern "C" void pause_sigalrm_handler(int) { alarm_fired = true; }

TEST_F(LlvmLibcPauseTest, InterruptedBySignal) {
  alarm_fired = false;

  struct sigaction sa = {};
  sa.sa_handler = pause_sigalrm_handler;
  ASSERT_THAT(LIBC_NAMESPACE::sigemptyset(&sa.sa_mask), Succeeds(0));
  sa.sa_flags = 0;

  struct sigaction old_sa = {};
  ASSERT_THAT(LIBC_NAMESPACE::sigaction(SIGALRM, &sa, &old_sa), Succeeds(0));
  auto restore_sigaction = LIBC_NAMESPACE::cpp::scope_exit(
      [&] { LIBC_NAMESPACE::sigaction(SIGALRM, &old_sa, nullptr); });

  // Use a short repeating interval timer so that even if a signal arrives
  // before pause() blocks, the subsequent tick wakes pause() immediately.
  struct itimerval timer = {};
  timer.it_value.tv_sec = 0;
  timer.it_value.tv_usec = 10000;
  timer.it_interval.tv_sec = 0;
  timer.it_interval.tv_usec = 10000;

  struct itimerval old_timer = {};
  ASSERT_THAT(LIBC_NAMESPACE::setitimer(ITIMER_REAL, &timer, &old_timer),
              Succeeds(0));
  auto restore_timer = LIBC_NAMESPACE::cpp::scope_exit(
      [&] { LIBC_NAMESPACE::setitimer(ITIMER_REAL, &old_timer, nullptr); });

  EXPECT_THAT(LIBC_NAMESPACE::pause(), Fails(EINTR));
  EXPECT_TRUE(alarm_fired);
}

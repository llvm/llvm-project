//===-- Unittests for setitimer -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/sys_time_macros.h"
#include "hdr/types/struct_itimerval.h"
#include "hdr/types/struct_sigaction.h"
#include "src/signal/sigaction.h"
#include "src/signal/sigemptyset.h"
#include "src/sys/time/setitimer.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcSysTimeSetitimerTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

static bool timer_fired(false);

extern "C" void handle_sigalrm(int) { timer_fired = true; }

TEST_F(LlvmLibcSysTimeSetitimerTest, SmokeTest) {
  libc_errno = 0;
  struct sigaction sa;
  sa.sa_handler = handle_sigalrm;
  LIBC_NAMESPACE::sigemptyset(&sa.sa_mask);
  sa.sa_flags = 0;
  LIBC_NAMESPACE::sigaction(SIGALRM, &sa, nullptr);

  struct itimerval timer;
  timer.it_value.tv_sec = 0;
  timer.it_value.tv_usec = 200000;
  timer.it_interval.tv_sec = 0;
  timer.it_interval.tv_usec = 0; // One-shot timer

  ASSERT_THAT(LIBC_NAMESPACE::setitimer(ITIMER_REAL, &timer, nullptr),
              returns(EQ(0)).with_errno(EQ(0)));

  while (true) {
    if (timer_fired)
      break;
  }

  ASSERT_TRUE(timer_fired);
}

TEST_F(LlvmLibcSysTimeSetitimerTest, InvalidRetTest) {
  struct itimerval timer = {};

  // out of range timer type (which)
  ASSERT_THAT(LIBC_NAMESPACE::setitimer(99, &timer, nullptr),
              returns(NE(0)).with_errno(NE(0)));

  // invalid microseconds (>= 1000000)
  timer.it_value.tv_sec = 0;
  timer.it_value.tv_usec = 1000000;
  ASSERT_THAT(LIBC_NAMESPACE::setitimer(ITIMER_REAL, &timer, nullptr),
              returns(EQ(-1)).with_errno(EQ(EINVAL)));

  // negative microseconds
  timer.it_value.tv_sec = 0;
  timer.it_value.tv_usec = -1;
  ASSERT_THAT(LIBC_NAMESPACE::setitimer(ITIMER_REAL, &timer, nullptr),
              returns(EQ(-1)).with_errno(EQ(EINVAL)));

  // Test 64-bit microsecond values that would truncate to a valid 32-bit value
  // (< 1000000). 0x100000047, If truncated to 32-bit integer, becomes 0x47.
  timer.it_value.tv_sec = 0;
  timer.it_value.tv_usec = 0x1'0000'0047;
  ASSERT_THAT(LIBC_NAMESPACE::setitimer(ITIMER_REAL, &timer, nullptr),
              returns(EQ(-1)).with_errno(EQ(EINVAL)));

  timer.it_value.tv_sec = 0;
  timer.it_value.tv_usec = 0xffff'ffff'0000'0047;
  ASSERT_THAT(LIBC_NAMESPACE::setitimer(ITIMER_REAL, &timer, nullptr),
              returns(EQ(-1)).with_errno(EQ(EINVAL)));
}

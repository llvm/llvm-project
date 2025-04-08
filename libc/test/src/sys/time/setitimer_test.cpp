//===-- Unittests for setitimer -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/signal/signal.h"
#include "hdr/types/struct_itimerval.h"
#include "src/sys/time/setitimer.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include <atomic>
#include <signal.h>

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;

static std::atomic<bool> timer_fired(false);

extern "C" void handle_sigalrm(int) { timer_fired.store(true); }

TEST(LlvmLibcSysTimeSetitimerTest, SmokeTest) {
  errno = 0;
  struct sigaction sa;
  sa.sa_handler = handle_sigalrm;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = 0;
  sigaction(SIGALRM, &sa, nullptr);

  struct itimerval timer;
  timer.it_value.tv_sec = 0;
  timer.it_value.tv_usec = 200000;
  timer.it_interval.tv_sec = 0;
  timer.it_interval.tv_usec = 0; // One-shot timer

  ASSERT_THAT(LIBC_NAMESPACE::setitimer(ITIMER_REAL, &timer, nullptr),
              returns(EQ(0)).with_errno(EQ(0)));

  while (true) {
    if (timer_fired.load())
      break;
  }

  ASSERT_TRUE(timer_fired.load());
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for the timeval macros in sys-time-macros.h.
///
//===----------------------------------------------------------------------===//

#include "include/llvm-libc-macros/sys-time-macros.h"
#include "include/llvm-libc-types/struct_timeval.h"
#include "include/llvm-libc-types/suseconds_t.h"
#include "include/llvm-libc-types/time_t.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcTimerMacrosTest, TimerAdd) {
  struct timeval a = {1, 600000};
  struct timeval b = {2, 700000};
  struct timeval result = {};
  timeradd(&a, &b, &result);
  EXPECT_EQ(result.tv_sec, time_t(4));
  EXPECT_EQ(result.tv_usec, suseconds_t(300000));

  // Without a carry.
  b = {2, 100000};
  timeradd(&a, &b, &result);
  EXPECT_EQ(result.tv_sec, time_t(3));
  EXPECT_EQ(result.tv_usec, suseconds_t(700000));
}

TEST(LlvmLibcTimerMacrosTest, TimerSub) {
  struct timeval a = {5, 100000};
  struct timeval b = {2, 300000};
  struct timeval result = {};
  timersub(&a, &b, &result);
  EXPECT_EQ(result.tv_sec, time_t(2));
  EXPECT_EQ(result.tv_usec, suseconds_t(800000));

  // Without a borrow.
  b = {2, 50000};
  timersub(&a, &b, &result);
  EXPECT_EQ(result.tv_sec, time_t(3));
  EXPECT_EQ(result.tv_usec, suseconds_t(50000));
}

TEST(LlvmLibcTimerMacrosTest, TimerClearAndIsSet) {
  struct timeval t = {3, 4};
  EXPECT_TRUE(timerisset(&t));
  timerclear(&t);
  EXPECT_EQ(t.tv_sec, time_t(0));
  EXPECT_EQ(t.tv_usec, suseconds_t(0));
  EXPECT_FALSE(timerisset(&t));

  t = {0, 1};
  EXPECT_TRUE(timerisset(&t));
}

// Each macro has to be usable as a single statement, or the else below
// attaches to the wrong if or does not compile.
TEST(LlvmLibcTimerMacrosTest, UsableAsOneStatement) {
  struct timeval a = {1, 0};
  struct timeval b = {1, 0};
  struct timeval result = {};
  int taken = 0;

  if (timerisset(&a))
    timeradd(&a, &b, &result);
  else
    taken = 1;
  EXPECT_EQ(result.tv_sec, time_t(2));
  EXPECT_EQ(taken, 0);

  if (!timerisset(&a))
    taken = 1;
  else
    timersub(&a, &b, &result);
  EXPECT_EQ(result.tv_sec, time_t(0));
  EXPECT_EQ(taken, 0);

  if (taken)
    taken = 1;
  else
    timerclear(&result);
  EXPECT_FALSE(timerisset(&result));
}

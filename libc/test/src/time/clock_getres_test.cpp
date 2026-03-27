//===-- Unittests for clock_getres- ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/time_macros.h"
#include "src/time/clock_getres.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;

TEST(LlvmLibcClockGetRes, Invalid) {
  timespec tp;
  EXPECT_THAT(LIBC_NAMESPACE::clock_getres(-1, &tp), Fails(EINVAL));
}

TEST(LlvmLibcClockGetRes, NullSpec) {
  EXPECT_THAT(LIBC_NAMESPACE::clock_getres(CLOCK_REALTIME, nullptr),
              Succeeds());
}

TEST(LlvmLibcClockGetRes, Realtime) {
  timespec tp;
  EXPECT_THAT(LIBC_NAMESPACE::clock_getres(CLOCK_REALTIME, &tp), Succeeds());
  EXPECT_GE(tp.tv_sec, static_cast<decltype(tp.tv_sec)>(0));
  EXPECT_GE(tp.tv_nsec, static_cast<decltype(tp.tv_nsec)>(0));
}

TEST(LlvmLibcClockGetRes, Monotonic) {
  timespec tp;
  ASSERT_THAT(LIBC_NAMESPACE::clock_getres(CLOCK_MONOTONIC, &tp), Succeeds());
  EXPECT_GE(tp.tv_sec, static_cast<decltype(tp.tv_sec)>(0));
  EXPECT_GE(tp.tv_nsec, static_cast<decltype(tp.tv_nsec)>(0));
}

TEST(LlvmLibcClockGetRes, ProcessCpuTime) {
  timespec tp;
  ASSERT_THAT(LIBC_NAMESPACE::clock_getres(CLOCK_PROCESS_CPUTIME_ID, &tp),
              Succeeds());
  EXPECT_GE(tp.tv_sec, static_cast<decltype(tp.tv_sec)>(0));
  EXPECT_GE(tp.tv_nsec, static_cast<decltype(tp.tv_nsec)>(0));
}

TEST(LlvmLibcClockGetRes, ThreadCpuTime) {
  timespec tp;
  ASSERT_THAT(LIBC_NAMESPACE::clock_getres(CLOCK_THREAD_CPUTIME_ID, &tp),
              Succeeds());
  EXPECT_GE(tp.tv_sec, static_cast<decltype(tp.tv_sec)>(0));
  EXPECT_GE(tp.tv_nsec, static_cast<decltype(tp.tv_nsec)>(0));
}

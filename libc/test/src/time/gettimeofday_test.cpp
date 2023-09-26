//===-- Unittests for gettimeofday ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <time.h>

#include "src/time/gettimeofday.h"
#include "src/time/nanosleep.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

namespace cpp = LIBC_NAMESPACE::cpp;

TEST(LlvmLibcGettimeofday, SmokeTest) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
  void *tz = nullptr;
  struct timeval tv;

  int sleep_times[2] = {200, 1000};
  for (int i = 0; i < 2; i++) {
    int ret = LIBC_NAMESPACE::gettimeofday(&tv, tz);
    ASSERT_EQ(ret, 0);

    int sleep_time = sleep_times[i];
    // Sleep for {sleep_time} microsceconds.
    struct timespec tim = {0, sleep_time * 1000};
    struct timespec tim2 = {0, 0};
    ret = LIBC_NAMESPACE::nanosleep(&tim, &tim2);

    // Call gettimeofday again and verify that it is more {sleep_time}
    // microscecods.
    struct timeval tv1;
    ret = LIBC_NAMESPACE::gettimeofday(&tv1, tz);
    ASSERT_EQ(ret, 0);
    ASSERT_GE(tv1.tv_usec - tv.tv_usec, sleep_time);
  }
}

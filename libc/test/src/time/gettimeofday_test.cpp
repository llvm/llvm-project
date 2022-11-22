//===-- Unittests for gettimeofday ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <errno.h>
#include <time.h>

#include "src/time/gettimeofday.h"
#include "src/time/nanosleep.h"
#include "test/ErrnoSetterMatcher.h"
#include "utils/UnitTest/Test.h"

namespace cpp = __llvm_libc::cpp;

TEST(LlvmLibcGettimeofday, SmokeTest) {
  using __llvm_libc::testing::ErrnoSetterMatcher::Succeeds;
  void *tz = nullptr;
  struct timeval tv;
  int ret = __llvm_libc::gettimeofday(&tv, tz);
  ASSERT_EQ(ret, 0);

  // Sleep for 200 microsceconds.
  struct timespec tim = {0, 200 * 1000};
  struct timespec tim2 = {0, 0};
  ret = __llvm_libc::nanosleep(&tim, &tim2);

  // Call gettimeofday again and verify that it is more 100 microscecods and
  // less than 300 microseconds,
  struct timeval tv1;
  ret = __llvm_libc::gettimeofday(&tv1, tz);
  ASSERT_EQ(ret, 0);
  ASSERT_GT(tv1.tv_usec - tv.tv_usec, 100);
  ASSERT_LT(tv1.tv_usec - tv.tv_usec, 300);
}

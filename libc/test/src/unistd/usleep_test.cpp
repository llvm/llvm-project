//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for usleep.
///
//===----------------------------------------------------------------------===//

#include "hdr/time_macros.h"
#include "hdr/types/struct_timespec.h"
#include "src/__support/time/clock_gettime.h"
#include "src/unistd/usleep.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

using LlvmLibcUsleepTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcUsleepTest, SmokeTest) {
  struct timespec start, end;
  ASSERT_TRUE(LIBC_NAMESPACE::internal::clock_gettime(CLOCK_MONOTONIC, &start)
                  .has_value());

  // Sleep for 10000 microseconds (10 ms).
  ASSERT_EQ(LIBC_NAMESPACE::usleep(10000), 0);

  ASSERT_TRUE(LIBC_NAMESPACE::internal::clock_gettime(CLOCK_MONOTONIC, &end)
                  .has_value());

  // Calculate elapsed time in nanoseconds.
  int64_t diff_sec = end.tv_sec - start.tv_sec;
  int64_t diff_nsec = end.tv_nsec - start.tv_nsec;
  int64_t elapsed_ns = diff_sec * 1000000000LL + diff_nsec;

  // It should have slept for at least 10 ms (10,000,000 ns).
  ASSERT_GE(elapsed_ns, int64_t{10000000});
}

TEST_F(LlvmLibcUsleepTest, Zero) { ASSERT_EQ(LIBC_NAMESPACE::usleep(0), 0); }

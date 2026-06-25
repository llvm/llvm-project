//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for sleep.
///
//===----------------------------------------------------------------------===//

#include "src/unistd/sleep.h"
#include "hdr/time_macros.h"
#include "hdr/types/struct_timespec.h"
#include "src/__support/time/clock_gettime.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcSleepTest, SmokeTest) {
  timespec start, end;
  auto start_res =
      LIBC_NAMESPACE::internal::clock_gettime(CLOCK_MONOTONIC, &start);
  ASSERT_TRUE(start_res.has_value());

  // Sleep for 1 second.
  ASSERT_EQ(LIBC_NAMESPACE::sleep(1), 0u);

  auto end_res = LIBC_NAMESPACE::internal::clock_gettime(CLOCK_MONOTONIC, &end);
  ASSERT_TRUE(end_res.has_value());

  // Calculate elapsed time in seconds.
  uint64_t elapsed_sec = static_cast<uint64_t>(end.tv_sec - start.tv_sec);
  if (end.tv_nsec < start.tv_nsec)
    --elapsed_sec;

  // It should have slept for at least 1 second.
  ASSERT_GE(elapsed_sec, uint64_t{1});
}

TEST(LlvmLibcSleepTest, Zero) { ASSERT_EQ(LIBC_NAMESPACE::sleep(0), 0u); }

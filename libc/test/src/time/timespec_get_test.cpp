//===-- Unittests for timespec_get ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/time_macros.h"
#include "hdr/types/struct_timespec.h"
#include "src/__support/macros/properties/architectures.h"
#include "src/time/timespec_get.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcTimespecGet, Utc) {
  timespec ts;
  int result;
  result = LIBC_NAMESPACE::timespec_get(&ts, TIME_UTC);
#ifdef LIBC_TARGET_ARCH_IS_GPU
  ASSERT_EQ(result, 0);
#else
  ASSERT_EQ(result, TIME_UTC);
  ASSERT_GT(ts.tv_sec, time_t(0));
#endif
}

// Baremetal implementation currently only supports TIME_UTC
#ifndef LIBC_TARGET_OS_IS_BAREMETAL
TEST(LlvmLibcTimespecGet, Monotonic) {
  timespec ts1, ts2;
  int result;
  result = LIBC_NAMESPACE::timespec_get(&ts1, TIME_MONOTONIC);
  ASSERT_EQ(result, TIME_MONOTONIC);
  ASSERT_GT(ts1.tv_sec, time_t(0));
  result = LIBC_NAMESPACE::timespec_get(&ts2, TIME_MONOTONIC);
  ASSERT_EQ(result, TIME_MONOTONIC);
  ASSERT_GE(ts2.tv_sec, ts1.tv_sec); // The monotonic time should increase.
  if (ts2.tv_sec == ts1.tv_sec) {
    ASSERT_GE(ts2.tv_nsec, ts1.tv_nsec);
  }
}
#endif

TEST(LlvmLibcTimespecGet, Unknown) {
  timespec ts;
  int result;
  result = LIBC_NAMESPACE::timespec_get(&ts, 0);
  ASSERT_EQ(result, 0);
}

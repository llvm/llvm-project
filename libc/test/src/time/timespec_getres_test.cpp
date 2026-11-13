//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for timespec_getres.
///
//===----------------------------------------------------------------------===//

#include "hdr/time_macros.h"
#include "hdr/types/struct_timespec.h"
#include "src/time/timespec_getres.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcTimespecGetRes, Utc) {
  timespec ts;
  int result = LIBC_NAMESPACE::timespec_getres(&ts, TIME_UTC);
  ASSERT_EQ(result, TIME_UTC);
  ASSERT_GE(ts.tv_sec, time_t(0));
  ASSERT_GE(ts.tv_nsec, 0L);
}

TEST(LlvmLibcTimespecGetRes, Monotonic) {
  timespec ts;
  int result = LIBC_NAMESPACE::timespec_getres(&ts, TIME_MONOTONIC);
  ASSERT_EQ(result, TIME_MONOTONIC);
  ASSERT_GE(ts.tv_sec, time_t(0));
  ASSERT_GE(ts.tv_nsec, 0L);
}

TEST(LlvmLibcTimespecGetRes, NullSpec) {
  int result = LIBC_NAMESPACE::timespec_getres(nullptr, TIME_UTC);
  ASSERT_EQ(result, TIME_UTC);
}

TEST(LlvmLibcTimespecGetRes, InvalidBase) {
  timespec ts;
  int result = LIBC_NAMESPACE::timespec_getres(&ts, -1);
  ASSERT_EQ(result, 0);

  result = LIBC_NAMESPACE::timespec_getres(&ts, 0);
  ASSERT_EQ(result, 0);
}


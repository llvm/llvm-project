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

#include "hdr/errno_macros.h"
#include "hdr/time_macros.h"
#include "hdr/types/struct_timespec.h"
#include "src/time/timespec_getres.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

using LlvmLibcTimespecGetRes = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcTimespecGetRes, Utc) {
  timespec ts;
  int result = LIBC_NAMESPACE::timespec_getres(&ts, TIME_UTC);
  ASSERT_EQ(result, TIME_UTC);
  ASSERT_GE(ts.tv_sec, time_t(0));
  ASSERT_GE(ts.tv_nsec, 0L);
  ASSERT_ERRNO_SUCCESS();
}

TEST_F(LlvmLibcTimespecGetRes, Monotonic) {
  timespec ts;
  int result = LIBC_NAMESPACE::timespec_getres(&ts, TIME_MONOTONIC);
  ASSERT_EQ(result, TIME_MONOTONIC);
  ASSERT_GE(ts.tv_sec, time_t(0));
  ASSERT_GE(ts.tv_nsec, 0L);
  ASSERT_ERRNO_SUCCESS();
}

#ifdef TIME_ACTIVE
TEST_F(LlvmLibcTimespecGetRes, Active) {
  timespec ts;
  int result = LIBC_NAMESPACE::timespec_getres(&ts, TIME_ACTIVE);
  ASSERT_EQ(result, TIME_ACTIVE);
  ASSERT_GE(ts.tv_sec, time_t(0));
  ASSERT_GE(ts.tv_nsec, 0L);
  ASSERT_ERRNO_SUCCESS();
}
#endif // TIME_ACTIVE

#ifdef TIME_THREAD_ACTIVE
TEST_F(LlvmLibcTimespecGetRes, ThreadActive) {
  timespec ts;
  int result = LIBC_NAMESPACE::timespec_getres(&ts, TIME_THREAD_ACTIVE);
  ASSERT_EQ(result, TIME_THREAD_ACTIVE);
  ASSERT_GE(ts.tv_sec, time_t(0));
  ASSERT_GE(ts.tv_nsec, 0L);
  ASSERT_ERRNO_SUCCESS();
}
#endif // TIME_THREAD_ACTIVE

TEST_F(LlvmLibcTimespecGetRes, NullSpec) {
  int result = LIBC_NAMESPACE::timespec_getres(nullptr, TIME_UTC);
  ASSERT_EQ(result, TIME_UTC);
  ASSERT_ERRNO_SUCCESS();
}

TEST_F(LlvmLibcTimespecGetRes, InvalidBase) {
  timespec ts;
  int result = LIBC_NAMESPACE::timespec_getres(&ts, -1);
  ASSERT_EQ(result, 0);
  ASSERT_ERRNO_SUCCESS();

  result = LIBC_NAMESPACE::timespec_getres(&ts, 0);
  ASSERT_EQ(result, 0);
  ASSERT_ERRNO_SUCCESS();
}

TEST_F(LlvmLibcTimespecGetRes, BadPointerSetsErrno) {
  int result = LIBC_NAMESPACE::timespec_getres(reinterpret_cast<timespec *>(-1),
                                               TIME_UTC);
  ASSERT_EQ(result, 0);
  ASSERT_ERRNO_EQ(EFAULT);
}

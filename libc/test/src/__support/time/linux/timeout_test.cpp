//===-- unit tests for linux's timeout utilities --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/expected.h"
#include "src/__support/time/linux/abs_timeout.h"
#include "src/__support/time/linux/monotonicity.h"
#include "test/UnitTest/Test.h"

template <class T, class E>
using expected = LIBC_NAMESPACE::cpp::expected<T, E>;
using AbsTimeout = LIBC_NAMESPACE::internal::AbsTimeout;

TEST(LlvmLibcSupportLinuxTimeoutTest, NegativeSecond) {
  timespec ts = {-1, 0};
  expected<AbsTimeout, AbsTimeout::Error> result =
      AbsTimeout::from_timespec(ts, false);
  ASSERT_FALSE(result.has_value());
  ASSERT_EQ(result.error(), AbsTimeout::Error::BeforeEpoch);
}
TEST(LlvmLibcSupportLinuxTimeoutTest, OverflowNano) {
  using namespace LIBC_NAMESPACE::time_units;
  timespec ts = {0, 2_s_ns};
  expected<AbsTimeout, AbsTimeout::Error> result =
      AbsTimeout::from_timespec(ts, false);
  ASSERT_FALSE(result.has_value());
  ASSERT_EQ(result.error(), AbsTimeout::Error::Invalid);
}
TEST(LlvmLibcSupportLinuxTimeoutTest, UnderflowNano) {
  timespec ts = {0, -1};
  expected<AbsTimeout, AbsTimeout::Error> result =
      AbsTimeout::from_timespec(ts, false);
  ASSERT_FALSE(result.has_value());
  ASSERT_EQ(result.error(), AbsTimeout::Error::Invalid);
}
TEST(LlvmLibcSupportLinuxTimeoutTest, NoChangeIfClockIsMonotonic) {
  timespec ts = {10000, 0};
  expected<AbsTimeout, AbsTimeout::Error> result =
      AbsTimeout::from_timespec(ts, false);
  ASSERT_TRUE(result.has_value());
  ensure_monotonicity(*result);
  ASSERT_FALSE(result->is_realtime());
  ASSERT_EQ(result->get_timespec().tv_sec, static_cast<time_t>(10000));
  ASSERT_EQ(result->get_timespec().tv_nsec, static_cast<time_t>(0));
}
TEST(LlvmLibcSupportLinuxTimeoutTest, ValidAfterConversion) {
  timespec ts;
  LIBC_NAMESPACE::internal::clock_gettime(CLOCK_REALTIME, &ts);
  expected<AbsTimeout, AbsTimeout::Error> result =
      AbsTimeout::from_timespec(ts, true);
  ASSERT_TRUE(result.has_value());
  ensure_monotonicity(*result);
  ASSERT_FALSE(result->is_realtime());
  ASSERT_TRUE(
      AbsTimeout::from_timespec(result->get_timespec(), false).has_value());
}

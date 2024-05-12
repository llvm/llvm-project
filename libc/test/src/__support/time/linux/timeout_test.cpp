//===-- unit tests for linux's timeout utilities --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/expected.h"
#include "src/__support/time/linux/timeout.h"

#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE {
namespace internal {
TEST(LlvmLibcSupportLinuxTimeoutTest, NegativeSecond) {
  timespec ts = {-1, 0};
  cpp::expected<AbsTimeout, AbsTimeout::Error> result =
      AbsTimeout::from_timespec(ts, false);
  ASSERT_FALSE(result.has_value());
  ASSERT_EQ(result.error(), AbsTimeout::Error::BeforeEpoch);
}
TEST(LlvmLibcSupportLinuxTimeoutTest, OverflowNano) {
  using namespace time_units;
  timespec ts = {0, 2_s_ns};
  cpp::expected<AbsTimeout, AbsTimeout::Error> result =
      AbsTimeout::from_timespec(ts, false);
  ASSERT_FALSE(result.has_value());
  ASSERT_EQ(result.error(), AbsTimeout::Error::Invalid);
}
TEST(LlvmLibcSupportLinuxTimeoutTest, UnderflowNano) {
  timespec ts = {0, -1};
  cpp::expected<AbsTimeout, AbsTimeout::Error> result =
      AbsTimeout::from_timespec(ts, false);
  ASSERT_FALSE(result.has_value());
  ASSERT_EQ(result.error(), AbsTimeout::Error::Invalid);
}
TEST(LlvmLibcSupportLinuxTimeoutTest, NoChangeIfClockIsMonotonic) {
  timespec ts = {10000, 0};
  cpp::expected<AbsTimeout, AbsTimeout::Error> result =
      AbsTimeout::from_timespec(ts, false);
  ASSERT_TRUE(result.has_value());
  result->ensure_monotonic();
  ASSERT_FALSE(result->is_realtime());
  ASSERT_EQ(result->get_timespec().tv_sec, static_cast<time_t>(10000));
  ASSERT_EQ(result->get_timespec().tv_nsec, static_cast<time_t>(0));
}
TEST(LlvmLibcSupportLinuxTimeoutTest, ValidAfterConversion) {
  timespec ts;
  internal::clock_gettime(CLOCK_REALTIME, &ts);
  cpp::expected<AbsTimeout, AbsTimeout::Error> result =
      AbsTimeout::from_timespec(ts, true);
  ASSERT_TRUE(result.has_value());
  result->ensure_monotonic();
  ASSERT_FALSE(result->is_realtime());
  ASSERT_TRUE(
      AbsTimeout::from_timespec(result->get_timespec(), false).has_value());
}
} // namespace internal
} // namespace LIBC_NAMESPACE

//===-- flang/unittests/Runtime/Time.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "flang/Runtime/time-intrinsic.h"
#include <algorithm>
#include <cctype>
#include <charconv>
#include <string>

using namespace Fortran::runtime;

TEST(TimeIntrinsics, CpuTime) {
  // We can't really test that we get the "right" result for CPU_TIME, but we
  // can have a smoke test to see that we get something reasonable on the
  // platforms where we expect to support it.
  double start{RTNAME(CpuTime)()};
  ASSERT_GE(start, 0.0);

  // Loop until we get a different value from CpuTime. If we don't get one
  // before we time out, then we should probably look into an implementation
  // for CpuTime with a better timer resolution.
  for (double end = start; end == start; end = RTNAME(CpuTime)()) {
    ASSERT_GE(end, 0.0);
    ASSERT_GE(end, start);
  }
}

using count_t = std::int64_t;

TEST(TimeIntrinsics, SystemClock) {
  // We can't really test that we get the "right" result for SYSTEM_CLOCK, but
  // we can have a smoke test to see that we get something reasonable on the
  // platforms where we expect to support it.

  // The value of the count rate and max will vary by platform, but they should
  // always be strictly positive if we have a working implementation of
  // SYSTEM_CLOCK.
  EXPECT_GT(RTNAME(SystemClockCountRate)(), 0);

  count_t max1{RTNAME(SystemClockCountMax)(1)};
  EXPECT_GT(max1, 0);
  EXPECT_LE(max1, static_cast<count_t>(0x7f));
  count_t start1{RTNAME(SystemClockCount)(1)};
  EXPECT_GE(start1, 0);
  EXPECT_LE(start1, max1);

  count_t max2{RTNAME(SystemClockCountMax)(2)};
  EXPECT_GT(max2, 0);
  EXPECT_LE(max2, static_cast<count_t>(0x7fff));
  count_t start2{RTNAME(SystemClockCount)(2)};
  EXPECT_GE(start2, 0);
  EXPECT_LE(start2, max2);

  count_t max4{RTNAME(SystemClockCountMax)(4)};
  EXPECT_GT(max4, 0);
  EXPECT_LE(max4, static_cast<count_t>(0x7fffffff));
  count_t start4{RTNAME(SystemClockCount)(4)};
  EXPECT_GE(start4, 0);
  EXPECT_LE(start4, max4);

  count_t max8{RTNAME(SystemClockCountMax)(8)};
  EXPECT_GT(max8, 0);
  count_t start8{RTNAME(SystemClockCount)(8)};
  EXPECT_GE(start8, 0);
  EXPECT_LT(start8, max8);

  count_t max16{RTNAME(SystemClockCountMax)(16)};
  EXPECT_GT(max16, 0);
  count_t start16{RTNAME(SystemClockCount)(16)};
  EXPECT_GE(start16, 0);
  EXPECT_LT(start16, max16);

  // Loop until we get a different value from SystemClockCount. If we don't get
  // one before we time out, then we should probably look into an implementation
  // for SystemClokcCount with a better timer resolution on this platform.
  for (count_t end{start8}; end == start8; end = RTNAME(SystemClockCount)(8)) {
    EXPECT_GE(end, 0);
    EXPECT_LE(end, max8);
    EXPECT_GE(end, start8);
  }
}

TEST(TimeIntrinsics, DateAndTime) {
  constexpr std::size_t bufferSize{16};
  std::string date(bufferSize, 'Z'), time(bufferSize, 'Z'),
      zone(bufferSize, 'Z');
  RTNAME(DateAndTime)
  (date.data(), date.size(), time.data(), time.size(), zone.data(), zone.size(),
      /*source=*/nullptr, /*line=*/0, /*values=*/nullptr);
  auto isBlank = [](const std::string &s) -> bool {
    return std::all_of(
        s.begin(), s.end(), [](char c) { return std::isblank(c); });
  };
  // Validate date is blank or YYYYMMDD.
  if (isBlank(date)) {
    EXPECT_TRUE(true);
  } else {
    count_t number{-1};
    auto [_, ec]{
        std::from_chars(date.data(), date.data() + date.size(), number)};
    ASSERT_TRUE(ec != std::errc::invalid_argument &&
        ec != std::errc::result_out_of_range);
    EXPECT_GE(number, 0);
    auto year = number / 10000;
    auto month = (number - year * 10000) / 100;
    auto day = number % 100;
    // Do not assume anything about the year, the test could be
    // run on system with fake/outdated dates.
    EXPECT_LE(month, 12);
    EXPECT_GT(month, 0);
    EXPECT_LE(day, 31);
    EXPECT_GT(day, 0);
  }

  // Validate time is hhmmss.sss or blank.
  if (isBlank(time)) {
    EXPECT_TRUE(true);
  } else {
    count_t number{-1};
    auto [next, ec]{
        std::from_chars(time.data(), time.data() + date.size(), number)};
    ASSERT_TRUE(ec != std::errc::invalid_argument &&
        ec != std::errc::result_out_of_range);
    ASSERT_GE(number, 0);
    auto hours = number / 10000;
    auto minutes = (number - hours * 10000) / 100;
    auto seconds = number % 100;
    EXPECT_LE(hours, 23);
    EXPECT_LE(minutes, 59);
    // Accept 60 for leap seconds.
    EXPECT_LE(seconds, 60);
    ASSERT_TRUE(next != time.data() + time.size());
    EXPECT_EQ(*next, '.');

    count_t milliseconds{-1};
    ASSERT_TRUE(next + 1 != time.data() + time.size());
    auto [_, ec2]{
        std::from_chars(next + 1, time.data() + date.size(), milliseconds)};
    ASSERT_TRUE(ec2 != std::errc::invalid_argument &&
        ec2 != std::errc::result_out_of_range);
    EXPECT_GE(milliseconds, 0);
    EXPECT_LE(milliseconds, 999);
  }

  // Validate zone is +hhmm or -hhmm or blank.
  if (isBlank(zone)) {
    EXPECT_TRUE(true);
  } else {
    ASSERT_TRUE(zone.size() > 1);
    EXPECT_TRUE(zone[0] == '+' || zone[0] == '-');
    count_t number{-1};
    auto [next, ec]{
        std::from_chars(zone.data() + 1, zone.data() + zone.size(), number)};
    ASSERT_TRUE(ec != std::errc::invalid_argument &&
        ec != std::errc::result_out_of_range);
    ASSERT_GE(number, 0);
    auto hours = number / 100;
    auto minutes = number % 100;
    EXPECT_LE(hours, 23);
    EXPECT_LE(minutes, 59);
  }
}


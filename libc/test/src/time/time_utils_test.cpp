//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for the time_utils helpers get_weekday, get_time_of_day and
/// update_from_seconds.
///
//===----------------------------------------------------------------------===//

#include "hdr/limits_macros.h"
#include "hdr/stdint_proxy.h"
#include "hdr/types/struct_tm.h"
#include "hdr/types/time_t.h"
#include "src/time/time_constants.h"
#include "src/time/time_utils.h"
#include "test/UnitTest/Test.h"

namespace time_constants = LIBC_NAMESPACE::time_constants;
namespace time_utils = LIBC_NAMESPACE::time_utils;

// Straightforward reference: (unix_days + 4) mod 7 with a non-negative result.
static int reference_weekday(int64_t unix_days) {
  int64_t wday = (unix_days % 7 + 7 + 4) % 7;
  return static_cast<int>(wday);
}

// Smallest and largest day counts accepted by update_from_seconds.
constexpr int64_t MIN_SUPPORTED_DAYS =
    INT_MIN *
    static_cast<int64_t>(time_constants::NUMBER_OF_SECONDS_IN_LEAP_YEAR) /
    time_constants::SECONDS_PER_DAY;
constexpr int64_t MAX_SUPPORTED_DAYS =
    INT_MAX *
    static_cast<int64_t>(time_constants::NUMBER_OF_SECONDS_IN_LEAP_YEAR) /
    time_constants::SECONDS_PER_DAY;

// Documented valid input range of get_weekday.
constexpr int64_t MIN_WEEKDAY_DAYS = -230471710930685136LL;
constexpr int64_t MAX_WEEKDAY_DAYS = 230696890912053658LL;

// Documented valid input range of get_time_of_day.
constexpr uint32_t MAX_TIME_OF_DAY_SECONDS = 2255818;

TEST(LlvmLibcTimeUtilsGetWeekday, KnownDates) {
  // 1970-01-01 Thursday
  EXPECT_EQ(time_utils::get_weekday(0), 4);
  // 1969-12-31 Wednesday
  EXPECT_EQ(time_utils::get_weekday(-1), 3);
  // 1970-01-02 Friday
  EXPECT_EQ(time_utils::get_weekday(1), 5);
  // 1970-01-04 Sunday
  EXPECT_EQ(time_utils::get_weekday(3), 0);
  // 1900-01-01 Monday
  EXPECT_EQ(time_utils::get_weekday(-25567), 1);
  // 2000-01-01 Saturday
  EXPECT_EQ(time_utils::get_weekday(10957), 6);
  // 2038-01-19 Tuesday
  EXPECT_EQ(time_utils::get_weekday(24855), 2);
}

TEST(LlvmLibcTimeUtilsGetWeekday, IsConstexpr) {
  static_assert(time_utils::get_weekday(0) == 4);
  static_assert(time_utils::get_weekday(-1) == 3);
  static_assert(time_utils::get_weekday(10957) == 6);
}

TEST(LlvmLibcTimeUtilsGetWeekday, MatchesReferenceAroundEpoch) {
  // About +/-27,000 years around the Unix epoch.
  for (int64_t days = -10'000'000; days <= 10'000'000; ++days)
    ASSERT_EQ(time_utils::get_weekday(days), reference_weekday(days));
}

TEST(LlvmLibcTimeUtilsGetWeekday, MatchesReferenceAtSupportedLimits) {
  // Every day near both ends of the range accepted by update_from_seconds.
  for (int64_t i = 0; i < 1'000'000; ++i) {
    ASSERT_EQ(time_utils::get_weekday(MIN_SUPPORTED_DAYS + i),
              reference_weekday(MIN_SUPPORTED_DAYS + i));
    ASSERT_EQ(time_utils::get_weekday(MAX_SUPPORTED_DAYS - i),
              reference_weekday(MAX_SUPPORTED_DAYS - i));
  }
}

TEST(LlvmLibcTimeUtilsGetWeekday, MatchesReferenceAtDocumentedLimits) {
  for (int64_t i = 0; i < 1'000'000; ++i) {
    ASSERT_EQ(time_utils::get_weekday(MIN_WEEKDAY_DAYS + i),
              reference_weekday(MIN_WEEKDAY_DAYS + i));
    ASSERT_EQ(time_utils::get_weekday(MAX_WEEKDAY_DAYS - i),
              reference_weekday(MAX_WEEKDAY_DAYS - i));
  }
}

TEST(LlvmLibcTimeUtilsGetWeekday, MatchesReferenceAcrossRange) {
  // Sparse sweep across the whole documented range using a step that is
  // coprime with 7 so all residues are exercised.
  constexpr int64_t STEP = 1'000'000'007LL * 1009;
  for (int64_t days = MIN_WEEKDAY_DAYS; days <= MAX_WEEKDAY_DAYS - STEP;
       days += STEP)
    ASSERT_EQ(time_utils::get_weekday(days), reference_weekday(days));
}

TEST(LlvmLibcTimeUtilsGetTimeOfDay, KnownTimes) {
  time_utils::TimeOfDay t = time_utils::get_time_of_day(0);
  EXPECT_EQ(t.hour, 0);
  EXPECT_EQ(t.minute, 0);
  EXPECT_EQ(t.second, 0);

  t = time_utils::get_time_of_day(59);
  EXPECT_EQ(t.hour, 0);
  EXPECT_EQ(t.minute, 0);
  EXPECT_EQ(t.second, 59);

  t = time_utils::get_time_of_day(60);
  EXPECT_EQ(t.hour, 0);
  EXPECT_EQ(t.minute, 1);
  EXPECT_EQ(t.second, 0);

  t = time_utils::get_time_of_day(3599);
  EXPECT_EQ(t.hour, 0);
  EXPECT_EQ(t.minute, 59);
  EXPECT_EQ(t.second, 59);

  t = time_utils::get_time_of_day(3600);
  EXPECT_EQ(t.hour, 1);
  EXPECT_EQ(t.minute, 0);
  EXPECT_EQ(t.second, 0);

  // 12:34:56
  t = time_utils::get_time_of_day(12 * 3600 + 34 * 60 + 56);
  EXPECT_EQ(t.hour, 12);
  EXPECT_EQ(t.minute, 34);
  EXPECT_EQ(t.second, 56);

  // Last second of the day: 23:59:59
  t = time_utils::get_time_of_day(86399);
  EXPECT_EQ(t.hour, 23);
  EXPECT_EQ(t.minute, 59);
  EXPECT_EQ(t.second, 59);
}

TEST(LlvmLibcTimeUtilsGetTimeOfDay, IsConstexpr) {
  constexpr time_utils::TimeOfDay TIME_OF_DAY =
      time_utils::get_time_of_day(86399);
  static_assert(TIME_OF_DAY.hour == 23 && TIME_OF_DAY.minute == 59 &&
                TIME_OF_DAY.second == 59);
}

TEST(LlvmLibcTimeUtilsGetTimeOfDay, ExhaustiveDocumentedRange) {
  for (uint32_t s = 0; s <= MAX_TIME_OF_DAY_SECONDS; ++s) {
    time_utils::TimeOfDay t = time_utils::get_time_of_day(s);
    ASSERT_EQ(t.hour, static_cast<int>(s / 3600));
    ASSERT_EQ(t.minute, static_cast<int>(s / 60 % 60));
    ASSERT_EQ(t.second, static_cast<int>(s % 60));
  }
}

// Checks the weekday and time of day that update_from_seconds produces for
// seconds against straightforward arithmetic.
static void expect_weekday_and_time_of_day(time_t seconds) {
  tm result{};
  ASSERT_TRUE(time_utils::update_from_seconds(seconds, &result).has_value());
  time_t days = seconds / time_constants::SECONDS_PER_DAY;
  time_t rem = seconds % time_constants::SECONDS_PER_DAY;
  if (rem < 0) {
    rem += time_constants::SECONDS_PER_DAY;
    --days;
  }
  EXPECT_EQ(result.tm_wday, reference_weekday(days));
  EXPECT_EQ(result.tm_hour, static_cast<int>(rem / 3600));
  EXPECT_EQ(result.tm_min, static_cast<int>(rem / 60 % 60));
  EXPECT_EQ(result.tm_sec, static_cast<int>(rem % 60));
}

TEST(LlvmLibcTimeUtilsUpdateFromSeconds, WeekdayAndTimeOfDay) {
  // Step by 3599 seconds (one second short of an hour) over several years on
  // both sides of the epoch, so each successive hour is sampled at a different
  // second offset, comparing with straightforward arithmetic.
  constexpr time_t SPAN = 10LL * 366 * time_constants::SECONDS_PER_DAY;
  for (time_t seconds = -SPAN; seconds <= SPAN; seconds += 3599)
    expect_weekday_and_time_of_day(seconds);
}

TEST(LlvmLibcTimeUtilsUpdateFromSeconds, FarFutureAndPast) {
  // About +/-2.1 billion years, close to the largest years that fit in an int.
  constexpr time_t FAR_SECONDS =
      INT_MAX * static_cast<int64_t>(time_constants::DAYS_PER_NON_LEAP_YEAR) *
      time_constants::SECONDS_PER_DAY;
  for (time_t offset = 0; offset < 2 * time_constants::SECONDS_PER_DAY;
       offset += 1234) {
    expect_weekday_and_time_of_day(FAR_SECONDS - offset);
    expect_weekday_and_time_of_day(-FAR_SECONDS + offset);
  }
}

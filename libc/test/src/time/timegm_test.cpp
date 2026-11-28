//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for timegm.
///
//===----------------------------------------------------------------------===//

#include "hdr/types/struct_tm.h"
#include "hdr/types/time_t.h"
#include "src/__support/CPP/limits.h" // INT_MAX
#include "src/time/time_constants.h"
#include "src/time/time_utils.h"
#include "src/time/timegm.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"
#include "test/src/time/TmMatcher.h"

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
using LIBC_NAMESPACE::time_constants::Month;
using LlvmLibcTimegmTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

static constexpr int tm_year(int year) {
  return year - LIBC_NAMESPACE::time_constants::TIME_YEAR_BASE;
}

TEST_F(LlvmLibcTimegmTest, Epoch) {
  const tm expected{.tm_sec = 0,
                    .tm_min = 0,
                    .tm_hour = 0,
                    .tm_mday = 1,
                    .tm_mon = Month::JANUARY,
                    .tm_year = tm_year(1970),
                    .tm_wday = 4,
                    .tm_yday = 0,
                    .tm_isdst = 0};
  tm tm_data = expected;
  tm_data.tm_wday = 0;
  tm_data.tm_yday = 0;
  EXPECT_THAT(LIBC_NAMESPACE::timegm(&tm_data),
              Succeeds(static_cast<time_t>(0)));
  EXPECT_TM_EQ(expected, tm_data);
}

TEST_F(LlvmLibcTimegmTest, BeforeEpoch) {
  const tm expected{.tm_sec = 40,
                    .tm_min = 17,
                    .tm_hour = 20,
                    .tm_mday = 20,
                    .tm_mon = Month::JULY,
                    .tm_year = tm_year(1969),
                    .tm_wday = 0,
                    .tm_yday = 200,
                    .tm_isdst = 0};
  tm tm_data = expected;
  tm_data.tm_wday = 3;
  tm_data.tm_yday = 0;
  EXPECT_THAT(LIBC_NAMESPACE::timegm(&tm_data),
              Succeeds(static_cast<time_t>(-14182940)));
  EXPECT_TM_EQ(expected, tm_data);
}

TEST_F(LlvmLibcTimegmTest, LeapYear2000) {
  const tm expected{.tm_sec = 0,
                    .tm_min = 0,
                    .tm_hour = 0,
                    .tm_mday = 29,
                    .tm_mon = Month::FEBRUARY,
                    .tm_year = tm_year(2000),
                    .tm_wday = 2,
                    .tm_yday = 59,
                    .tm_isdst = 0};
  tm tm_data = expected;
  tm_data.tm_wday = 0;
  tm_data.tm_yday = 0;
  EXPECT_THAT(LIBC_NAMESPACE::timegm(&tm_data),
              Succeeds(static_cast<time_t>(951782400)));
  EXPECT_TM_EQ(expected, tm_data);
}

TEST_F(LlvmLibcTimegmTest, NonLeapYear2100) {
  // 2100 is not a leap year, so February 29th normalizes to March 1st.
  tm tm_data{.tm_sec = 0,
             .tm_min = 0,
             .tm_hour = 0,
             .tm_mday = 29,
             .tm_mon = Month::FEBRUARY,
             .tm_year = tm_year(2100),
             .tm_wday = 0,
             .tm_yday = 0,
             .tm_isdst = 0};
  EXPECT_THAT(LIBC_NAMESPACE::timegm(&tm_data),
              Succeeds(static_cast<time_t>(4107542400)));
  EXPECT_TM_EQ((tm{.tm_sec = 0,
                   .tm_min = 0,
                   .tm_hour = 0,
                   .tm_mday = 1,
                   .tm_mon = Month::MARCH,
                   .tm_year = tm_year(2100),
                   .tm_wday = 1,
                   .tm_yday = 59,
                   .tm_isdst = 0}),
               tm_data);
}

TEST_F(LlvmLibcTimegmTest, NormalizeMonth) {
  // Month 12 of 2023 is January 2024.
  tm tm_data{.tm_sec = 0,
             .tm_min = 0,
             .tm_hour = 0,
             .tm_mday = 1,
             .tm_mon = 12,
             .tm_year = tm_year(2023),
             .tm_wday = 0,
             .tm_yday = 0,
             .tm_isdst = 0};
  EXPECT_THAT(LIBC_NAMESPACE::timegm(&tm_data),
              Succeeds(static_cast<time_t>(1704067200)));
  EXPECT_TM_EQ((tm{.tm_sec = 0,
                   .tm_min = 0,
                   .tm_hour = 0,
                   .tm_mday = 1,
                   .tm_mon = Month::JANUARY,
                   .tm_year = tm_year(2024),
                   .tm_wday = 1,
                   .tm_yday = 0,
                   .tm_isdst = 0}),
               tm_data);
}

TEST_F(LlvmLibcTimegmTest, NormalizeDayZero) {
  // Day 0 of March 2024 is the last day of February 2024.
  tm tm_data{.tm_sec = 0,
             .tm_min = 0,
             .tm_hour = 0,
             .tm_mday = 0,
             .tm_mon = Month::MARCH,
             .tm_year = tm_year(2024),
             .tm_wday = 0,
             .tm_yday = 0,
             .tm_isdst = 0};
  EXPECT_THAT(LIBC_NAMESPACE::timegm(&tm_data),
              Succeeds(static_cast<time_t>(1709164800)));
  EXPECT_TM_EQ((tm{.tm_sec = 0,
                   .tm_min = 0,
                   .tm_hour = 0,
                   .tm_mday = 29,
                   .tm_mon = Month::FEBRUARY,
                   .tm_year = tm_year(2024),
                   .tm_wday = 4,
                   .tm_yday = 59,
                   .tm_isdst = 0}),
               tm_data);
}

TEST_F(LlvmLibcTimegmTest, NormalizeLargeMinutes) {
  // INT_MAX minutes after the epoch must not overflow int arithmetic.
  tm tm_data{.tm_sec = 0,
             .tm_min = INT_MAX,
             .tm_hour = 0,
             .tm_mday = 1,
             .tm_mon = Month::JANUARY,
             .tm_year = tm_year(1970),
             .tm_wday = 0,
             .tm_yday = 0,
             .tm_isdst = 0};
  EXPECT_THAT(LIBC_NAMESPACE::timegm(&tm_data),
              Succeeds(static_cast<time_t>(128849018820)));
  EXPECT_TM_EQ((tm{.tm_sec = 0,
                   .tm_min = 7,
                   .tm_hour = 2,
                   .tm_mday = 23,
                   .tm_mon = Month::JANUARY,
                   .tm_year = tm_year(6053),
                   .tm_wday = 4,
                   .tm_yday = 22,
                   .tm_isdst = 0}),
               tm_data);
}

TEST_F(LlvmLibcTimegmTest, Year2038) {
  constexpr time_t INT32_MAX_SECONDS = 2147483647;
  tm expected{.tm_sec = 7,
              .tm_min = 14,
              .tm_hour = 3,
              .tm_mday = 19,
              .tm_mon = Month::JANUARY,
              .tm_year = tm_year(2038),
              .tm_wday = 2,
              .tm_yday = 18,
              .tm_isdst = 0};
  tm tm_data = expected;
  tm_data.tm_wday = 0;
  tm_data.tm_yday = 0;
  EXPECT_THAT(LIBC_NAMESPACE::timegm(&tm_data), Succeeds(INT32_MAX_SECONDS));
  EXPECT_TM_EQ(expected, tm_data);

  // One second later does not fit in 32 bits, but time_t is 64-bit.
  expected.tm_sec = 8;
  tm_data = expected;
  tm_data.tm_wday = 0;
  tm_data.tm_yday = 0;
  EXPECT_THAT(LIBC_NAMESPACE::timegm(&tm_data),
              Succeeds(INT32_MAX_SECONDS + 1));
  EXPECT_TM_EQ(expected, tm_data);
}

TEST_F(LlvmLibcTimegmTest, IgnoresAndClearsIsdst) {
  const tm base{.tm_sec = 0,
                .tm_min = 0,
                .tm_hour = 0,
                .tm_mday = 1,
                .tm_mon = Month::JANUARY,
                .tm_year = tm_year(1970),
                .tm_wday = 0,
                .tm_yday = 0,
                .tm_isdst = 0};

  tm tm_data = base;
  tm_data.tm_isdst = -1;
  EXPECT_THAT(LIBC_NAMESPACE::timegm(&tm_data),
              Succeeds(static_cast<time_t>(0)));
  EXPECT_EQ(tm_data.tm_isdst, 0);

  tm_data = base;
  tm_data.tm_isdst = 1;
  EXPECT_THAT(LIBC_NAMESPACE::timegm(&tm_data),
              Succeeds(static_cast<time_t>(0)));
  EXPECT_EQ(tm_data.tm_isdst, 0);
}

#ifdef __linux__
TEST_F(LlvmLibcTimegmTest, SetsUtcOffset) {
  tm tm_data{.tm_sec = 0,
             .tm_min = 0,
             .tm_hour = 12,
             .tm_mday = 1,
             .tm_mon = Month::JUNE,
             .tm_year = tm_year(2024),
             .tm_wday = 0,
             .tm_yday = 0,
             .tm_isdst = 0};
  tm_data.tm_gmtoff = 3600;
  tm_data.tm_zone = nullptr;
  EXPECT_THAT(LIBC_NAMESPACE::timegm(&tm_data),
              Succeeds(static_cast<time_t>(1717243200)));
  EXPECT_EQ(tm_data.tm_gmtoff, 0L);
  EXPECT_STREQ(tm_data.tm_zone, "GMT");
}
#endif

TEST_F(LlvmLibcTimegmTest, Overflow) {
  // Month 12 of the largest tm_year needs a tm_year past INT_MAX.
  tm tm_data{.tm_sec = 0,
             .tm_min = 0,
             .tm_hour = 0,
             .tm_mday = 1,
             .tm_mon = 12,
             .tm_year = INT_MAX,
             .tm_wday = 0,
             .tm_yday = 0,
             .tm_isdst = 0};
  EXPECT_THAT(LIBC_NAMESPACE::timegm(&tm_data),
              Fails(LIBC_NAMESPACE::time_utils::TIME_OVERFLOW,
                    static_cast<time_t>(-1)));
}

TEST_F(LlvmLibcTimegmTest, NullPtr) {
  EXPECT_DEATH([] { LIBC_NAMESPACE::timegm(nullptr); }, WITH_SIGNAL(-1));
}

TEST_F(LlvmLibcTimegmTest, NormalizeMultipleFields) {
  tm tm_data{.tm_sec = 61,
             .tm_min = -1,
             .tm_hour = 25,
             .tm_mday = 0,
             .tm_mon = -1,
             .tm_year = tm_year(2024),
             .tm_wday = 0,
             .tm_yday = 0,
             .tm_isdst = 0};
  EXPECT_THAT(LIBC_NAMESPACE::timegm(&tm_data),
              Succeeds(static_cast<time_t>(1701392401)));
  EXPECT_TM_EQ((tm{.tm_sec = 1,
                   .tm_min = 0,
                   .tm_hour = 1,
                   .tm_mday = 1,
                   .tm_mon = Month::DECEMBER,
                   .tm_year = tm_year(2023),
                   .tm_wday = 5,
                   .tm_yday = 334,
                   .tm_isdst = 0}),
               tm_data);
}

TEST_F(LlvmLibcTimegmTest, AdditionalDates) {
  const tm expected[] = {
      tm{.tm_sec = 59,
         .tm_min = 59,
         .tm_hour = 23,
         .tm_mday = 31,
         .tm_mon = Month::DECEMBER,
         .tm_year = tm_year(1969),
         .tm_wday = 3,
         .tm_yday = 364,
         .tm_isdst = 0},
      tm{.tm_sec = 15,
         .tm_min = 30,
         .tm_hour = 12,
         .tm_mday = 29,
         .tm_mon = Month::FEBRUARY,
         .tm_year = tm_year(2000),
         .tm_wday = 2,
         .tm_yday = 59,
         .tm_isdst = 0},
      tm{.tm_sec = 0,
         .tm_min = 0,
         .tm_hour = 0,
         .tm_mday = 1,
         .tm_mon = Month::MARCH,
         .tm_year = tm_year(1900),
         .tm_wday = 4,
         .tm_yday = 59,
         .tm_isdst = 0},
  };

  tm tm_data = expected[0];
  tm_data.tm_wday = 0;
  tm_data.tm_yday = 0;
  EXPECT_THAT(LIBC_NAMESPACE::timegm(&tm_data),
              Succeeds(static_cast<time_t>(-1)));
  EXPECT_TM_EQ(expected[0], tm_data);

  tm_data = expected[1];
  tm_data.tm_wday = 0;
  tm_data.tm_yday = 0;
  EXPECT_THAT(LIBC_NAMESPACE::timegm(&tm_data),
              Succeeds(static_cast<time_t>(951827415)));
  EXPECT_TM_EQ(expected[1], tm_data);

  tm_data = expected[2];
  tm_data.tm_wday = 0;
  tm_data.tm_yday = 0;
  EXPECT_THAT(LIBC_NAMESPACE::timegm(&tm_data),
              Succeeds(static_cast<time_t>(-2203891200)));
  EXPECT_TM_EQ(expected[2], tm_data);
}

//===-- Unittests for strftime --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/types/struct_tm.h"
#include "src/__support/CPP/array.h"
#include "src/__support/integer_to_string.h"
#include "src/time/strftime.h"
#include "src/time/time_constants.h"
#include "test/UnitTest/Test.h"

// Copied from sprintf_test.cpp.
// TODO: put this somewhere more reusable, it's handy.
// Subtract 1 from sizeof(expected_str) to account for the null byte.
#define EXPECT_STREQ_LEN(actual_written, actual_str, expected_str)             \
  EXPECT_EQ(actual_written, sizeof(expected_str) - 1);                         \
  EXPECT_STREQ(actual_str, expected_str);

constexpr int get_adjusted_year(int year) {
  // tm_year counts years since 1900, so subtract 1900 to get the tm_year for a
  // given raw year.
  return year - LIBC_NAMESPACE::time_constants::TIME_YEAR_BASE;
}

// TODO: Move this somewhere it can be reused. It seems like a useful tool to
// have.
// A helper class to generate simple padded numbers. It places the result in its
// internal buffer, which is cleared on every call.
class SimplePaddedNum {
  static constexpr size_t BUFF_LEN = 16;
  char buff[BUFF_LEN];
  size_t cur_len; // length of string currently in buff

  void clear_buff() {
    // TODO: builtin_memset?
    for (size_t i = 0; i < BUFF_LEN; ++i)
      buff[i] = '\0';
  }

public:
  SimplePaddedNum() = default;

  // PRECONDITIONS: 0 < num < 2**31, min_width < 16
  // Returns: Pointer to the start of the padded number as a string, stored in
  // the internal buffer.
  char *get_padded_num(int num, size_t min_width, char padding_char = '0') {
    clear_buff();

    // we're not handling the negative sign here, so padding on negative numbers
    // will be incorrect. For this use case I consider that to be a reasonable
    // tradeoff for simplicity. This is more meant for the cases where we can
    // loop through all the possibilities, and for time those are all positive.
    LIBC_NAMESPACE::IntegerToString<int> raw(num);
    auto str = raw.view();
    int leading_zeroes = min_width - raw.size();

    size_t i = 0;
    for (; static_cast<int>(i) < leading_zeroes; ++i)
      buff[i] = padding_char;
    for (size_t str_cur = 0; str_cur < str.size(); ++i, ++str_cur)
      buff[i] = str[str_cur];
    cur_len = i;
    return buff;
  }

  size_t get_str_len() { return cur_len; }
};

TEST(LlvmLibcStrftimeTest, ConstantConversions) {
  // this tests %n, %t, and %%, which read nothing.
  struct tm time;
  char buffer[100];
  size_t written = 0;

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%n", &time);
  EXPECT_STREQ_LEN(written, buffer, "\n");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%t", &time);
  EXPECT_STREQ_LEN(written, buffer, "\t");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%%", &time);
  EXPECT_STREQ_LEN(written, buffer, "%");
}

TEST(LlvmLibcStrftimeTest, CenturyTests) {
  // this tests %C, which reads: [tm_year]
  struct tm time;
  char buffer[100];
  size_t written = 0;

  // basic tests
  time.tm_year = get_adjusted_year(2022);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%C", &time);
  EXPECT_STREQ_LEN(written, buffer, "20");

  time.tm_year = get_adjusted_year(11900);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%C", &time);
  EXPECT_STREQ_LEN(written, buffer, "119");

  time.tm_year = get_adjusted_year(1900);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%C", &time);
  EXPECT_STREQ_LEN(written, buffer, "19");

  time.tm_year = get_adjusted_year(900);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%C", &time);
  EXPECT_STREQ_LEN(written, buffer, "09");

  time.tm_year = get_adjusted_year(0);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%C", &time);
  EXPECT_STREQ_LEN(written, buffer, "00");

  // This case does not match what glibc does.
  // Both the C standard and Posix say %C is "Replaced by the year divided by
  // 100 and truncated to an integer, as a decimal number."
  // What glibc does is it returns the century for the provided year.
  // The difference is that glibc returns "-1" as the century for year -1, and
  // "-2" for year -101.
  // This case demonstrates that LLVM-libc instead just divides by 100, and
  // returns the result. "00" for year -1, and "-1" for year -101.
  // Personally, neither of these really feels right. Posix has a table of
  // examples where it treats "%C%y" as identical to "%Y". Neither of these
  // behaviors would handle that properly, you'd either get "-199" or "0099"
  // (since %y always returns a number in the range [00-99]).
  time.tm_year = get_adjusted_year(-1);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%C", &time);
  EXPECT_STREQ_LEN(written, buffer, "00");

  time.tm_year = get_adjusted_year(-101);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%C", &time);
  EXPECT_STREQ_LEN(written, buffer, "-1");

  time.tm_year = get_adjusted_year(-9001);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%C", &time);
  EXPECT_STREQ_LEN(written, buffer, "-90");

  time.tm_year = get_adjusted_year(-10001);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%C", &time);
  EXPECT_STREQ_LEN(written, buffer, "-100");

  // width tests (with the 0 flag, since the default padding is undefined).
  time.tm_year = get_adjusted_year(2023);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01C", &time);
  EXPECT_STREQ_LEN(written, buffer, "20");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%02C", &time);
  EXPECT_STREQ_LEN(written, buffer, "20");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%05C", &time);
  EXPECT_STREQ_LEN(written, buffer, "00020");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%010C", &time);
  EXPECT_STREQ_LEN(written, buffer, "0000000020");

  time.tm_year = get_adjusted_year(900);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01C", &time);
  EXPECT_STREQ_LEN(written, buffer, "9");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%02C", &time);
  EXPECT_STREQ_LEN(written, buffer, "09");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%05C", &time);
  EXPECT_STREQ_LEN(written, buffer, "00009");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%010C", &time);
  EXPECT_STREQ_LEN(written, buffer, "0000000009");

  time.tm_year = get_adjusted_year(12345);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01C", &time);
  EXPECT_STREQ_LEN(written, buffer, "123");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%02C", &time);
  EXPECT_STREQ_LEN(written, buffer, "123");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%05C", &time);
  EXPECT_STREQ_LEN(written, buffer, "00123");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%010C", &time);
  EXPECT_STREQ_LEN(written, buffer, "0000000123");

  time.tm_year = get_adjusted_year(-123);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01C", &time);
  EXPECT_STREQ_LEN(written, buffer, "-1");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%02C", &time);
  EXPECT_STREQ_LEN(written, buffer, "-1");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%05C", &time);
  EXPECT_STREQ_LEN(written, buffer, "-0001");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%010C", &time);
  EXPECT_STREQ_LEN(written, buffer, "-000000001");

  // '+' flag tests
  time.tm_year = get_adjusted_year(2023);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+1C", &time);
  EXPECT_STREQ_LEN(written, buffer, "20");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+2C", &time);
  EXPECT_STREQ_LEN(written, buffer, "20");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+5C", &time);
  EXPECT_STREQ_LEN(written, buffer, "+0020");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+10C", &time);
  EXPECT_STREQ_LEN(written, buffer, "+000000020");

  time.tm_year = get_adjusted_year(900);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+1C", &time);
  EXPECT_STREQ_LEN(written, buffer, "9");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+2C", &time);
  EXPECT_STREQ_LEN(written, buffer, "09");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+5C", &time);
  EXPECT_STREQ_LEN(written, buffer, "+0009");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+10C", &time);
  EXPECT_STREQ_LEN(written, buffer, "+000000009");

  time.tm_year = get_adjusted_year(12345);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+1C", &time);
  EXPECT_STREQ_LEN(written, buffer, "+123");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+2C", &time);
  EXPECT_STREQ_LEN(written, buffer, "+123");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+5C", &time);
  EXPECT_STREQ_LEN(written, buffer, "+0123");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+10C", &time);
  EXPECT_STREQ_LEN(written, buffer, "+000000123");

  time.tm_year = get_adjusted_year(-123);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+1C", &time);
  EXPECT_STREQ_LEN(written, buffer, "-1");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+2C", &time);
  EXPECT_STREQ_LEN(written, buffer, "-1");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+5C", &time);
  EXPECT_STREQ_LEN(written, buffer, "-0001");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+10C", &time);
  EXPECT_STREQ_LEN(written, buffer, "-000000001");

  // Posix specified tests:
  time.tm_year = get_adjusted_year(17);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%C", &time);
  EXPECT_STREQ_LEN(written, buffer, "00");

  time.tm_year = get_adjusted_year(270);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%C", &time);
  EXPECT_STREQ_LEN(written, buffer, "02");

  time.tm_year = get_adjusted_year(270);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+3C", &time);
  EXPECT_STREQ_LEN(written, buffer, "+02");

  time.tm_year = get_adjusted_year(12345);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+3C", &time);
  EXPECT_STREQ_LEN(written, buffer, "+123");

  time.tm_year = get_adjusted_year(12345);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%04C", &time);
  EXPECT_STREQ_LEN(written, buffer, "0123");

  time.tm_year = get_adjusted_year(12345);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+4C", &time);
  EXPECT_STREQ_LEN(written, buffer, "+123");

  time.tm_year = get_adjusted_year(123456);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%06C", &time);
  EXPECT_STREQ_LEN(written, buffer, "001234");

  time.tm_year = get_adjusted_year(123456);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+6C", &time);
  EXPECT_STREQ_LEN(written, buffer, "+01234");
}

TEST(LlvmLibcStrftimeTest, TwoDigitDayOfMonth) {
  // this tests %d, which reads: [tm_mday]
  struct tm time;
  char buffer[100];
  size_t written = 0;
  SimplePaddedNum spn;

  // Tests on all the well defined values
  for (size_t i = 1; i < 32; ++i) {
    time.tm_mday = i;
    written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%d", &time);
    char *result = spn.get_padded_num(i, 2);

    ASSERT_STREQ(buffer, result);
    ASSERT_EQ(written, size_t(2));
  }

  // padding is technically undefined for this conversion, but we support it, so
  // we need to test it.
  time.tm_mday = 5;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01d", &time);
  EXPECT_STREQ_LEN(written, buffer, "5");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%02d", &time);
  EXPECT_STREQ_LEN(written, buffer, "05");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%05d", &time);
  EXPECT_STREQ_LEN(written, buffer, "00005");

  time.tm_mday = 31;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01d", &time);
  EXPECT_STREQ_LEN(written, buffer, "31");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%02d", &time);
  EXPECT_STREQ_LEN(written, buffer, "31");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%05d", &time);
  EXPECT_STREQ_LEN(written, buffer, "00031");
}

TEST(LlvmLibcStrftimeTest, MinDigitDayOfMonth) {
  // this tests %e, which reads: [tm_mday]
  struct tm time;
  char buffer[100];
  size_t written = 0;
  SimplePaddedNum spn;

  // Tests on all the well defined values
  for (size_t i = 1; i < 32; ++i) {
    time.tm_mday = i;
    written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%e", &time);
    char *result = spn.get_padded_num(i, 2, ' ');

    ASSERT_STREQ(buffer, result);
    ASSERT_EQ(written, spn.get_str_len());
  }

  // padding is technically undefined for this conversion, but we support it, so
  // we need to test it.
  time.tm_mday = 5;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01e", &time);
  EXPECT_STREQ_LEN(written, buffer, "5");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%02e", &time);
  EXPECT_STREQ_LEN(written, buffer, "05");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%05e", &time);
  EXPECT_STREQ_LEN(written, buffer, "00005");

  time.tm_mday = 31;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01e", &time);
  EXPECT_STREQ_LEN(written, buffer, "31");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%02e", &time);
  EXPECT_STREQ_LEN(written, buffer, "31");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%05e", &time);
  EXPECT_STREQ_LEN(written, buffer, "00031");
}

TEST(LlvmLibcStrftimeTest, ISOYearOfCentury) {
  // this tests %g, which reads: [tm_year, tm_wday, tm_yday]

  // A brief primer on ISO dates:
  // 1) ISO weeks start on Monday and end on Sunday
  // 2) ISO years start on the Monday of the 1st ISO week of the year
  // 3) The 1st ISO week of the ISO year has the 4th day of the Gregorian year.

  struct tm time;
  char buffer[100];
  size_t written = 0;
  SimplePaddedNum spn;

  // a sunday in the middle of the year. No need to worry about rounding
  time.tm_wday = 0;
  time.tm_yday = 100;

  // Test the easy cases
  for (size_t i = 0; i < 102; ++i) {
    time.tm_year = i;
    written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%g", &time);
    char *result = spn.get_padded_num(i % 100, 2);

    ASSERT_STREQ(buffer, result);
    ASSERT_EQ(written, spn.get_str_len());
  }

  // Test the harder to round cases

  // not a leap year. Not relevant for the start-of-year tests, but it does
  // matter for the end-of-year tests.
  time.tm_year = 99;

  /*
This table has an X for each day that should be in the previous year,
everywhere else should be in the current year.

       yday
      0123456
  i 1         Monday
  s 2         Tuesday
  o 3         Wednesday
  w 4         Thursday
  d 5 X       Friday
  a 6 XX      Saturday
  y 7 XXX     Sunday
*/

  // check the first days of the year
  for (size_t yday = 0; yday < 5; ++yday) {
    for (size_t iso_wday = LIBC_NAMESPACE::time_constants::MONDAY; iso_wday < 8;
         ++iso_wday) {
      // start with monday, to match the ISO week.
      time.tm_wday = iso_wday % LIBC_NAMESPACE::time_constants::DAYS_PER_WEEK;
      time.tm_yday = yday;

      written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%g", &time);

      if (iso_wday <= LIBC_NAMESPACE::time_constants::THURSDAY || yday >= 3) {
        // monday - thursday are never in the previous year, nor are the 4th and
        // after.
        EXPECT_STREQ_LEN(written, buffer, "99");
      } else {
        // iso_wday is 5, 6, or 7 and yday is 0, 1, or 2.
        // days_since_thursday is therefor 1, 2, or 3.
        const size_t days_since_thursday =
            iso_wday - LIBC_NAMESPACE::time_constants::THURSDAY;

        if (days_since_thursday > yday) {
          EXPECT_STREQ_LEN(written, buffer, "98");
        } else {
          EXPECT_STREQ_LEN(written, buffer, "99");
        }
      }
    }
  }

  /*
  Similar to above, but the Xs represent being in the NEXT year. Also the
  top counts down until the end of the year.

    year end - yday
        6543210
    i 1     XXX Monday
    s 2      XX Tuesday
    o 3       X Wednesday
    w 4         Thursday
    d 5         Friday
    a 6         Saturday
    y 7         Sunday


  If we place the charts next to each other, you can more easily see the
  pattern:

year end - yday yday
        6543210 0123456
    i 1     XXX         Monday
    s 2      XX         Tuesday
    o 3       X         Wednesday
    w 4                 Thursday
    d 5         X       Friday
    a 6         XX      Saturday
    y 7         XXX     Sunday

    From this we can see that thursday is always in the same ISO and regular
    year, because the ISO year starts on the week with the 4th. Since Thursday
    is at least 3 days from either edge of the ISO week, the first thursday of
    the year is always in the first ISO week of the year.
  */

  // set up all the extra stuff to cover leap years.
  struct tm time_leap_year;
  char buffer_leap_year[100];
  size_t written_leap_year = 0;
  time_leap_year = time;
  time_leap_year.tm_year = 100; // 2000 is a leap year.

  // check the last days of the year. Checking 5 to make sure all the leap year
  // cases are covered as well.
  for (size_t days_left = 0; days_left < 5; ++days_left) {
    for (size_t iso_wday = LIBC_NAMESPACE::time_constants::MONDAY; iso_wday < 8;
         ++iso_wday) {
      // start with monday, to match the ISO week.
      time.tm_wday = iso_wday % LIBC_NAMESPACE::time_constants::DAYS_PER_WEEK;
      // subtract 1 from the max yday to handle yday being 0-indexed.
      time.tm_yday = LIBC_NAMESPACE::time_constants::DAYS_PER_NON_LEAP_YEAR -
                     1 - days_left;

      time_leap_year.tm_wday =
          iso_wday % LIBC_NAMESPACE::time_constants::DAYS_PER_WEEK;
      time_leap_year.tm_yday =
          LIBC_NAMESPACE::time_constants::LAST_DAY_OF_LEAP_YEAR - days_left;

      written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%g", &time);
      written_leap_year = LIBC_NAMESPACE::strftime(
          buffer_leap_year, sizeof(buffer_leap_year), "%g", &time_leap_year);

      if (iso_wday >= LIBC_NAMESPACE::time_constants::THURSDAY ||
          days_left >= 3) {
        // thursday - sunday are never in the next year, nor are days more than
        // 3 days before the end.
        EXPECT_STREQ_LEN(written, buffer, "99");
        EXPECT_STREQ_LEN(written_leap_year, buffer_leap_year, "00");
      } else {
        // iso_wday is 1, 2 or 3 and days_left is 0, 1, or 2
        if (iso_wday + days_left <= 3) {
          EXPECT_STREQ_LEN(written, buffer, "00");
          EXPECT_STREQ_LEN(written_leap_year, buffer_leap_year, "01");
        } else {
          EXPECT_STREQ_LEN(written, buffer, "99");
          EXPECT_STREQ_LEN(written_leap_year, buffer_leap_year, "00");
        }
      }
    }
  }

  // padding is technically undefined for this conversion, but we support it, so
  // we need to test it.
  time.tm_year = 5;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01g", &time);
  EXPECT_STREQ_LEN(written, buffer, "5");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%02g", &time);
  EXPECT_STREQ_LEN(written, buffer, "05");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%05g", &time);
  EXPECT_STREQ_LEN(written, buffer, "00005");

  time.tm_year = 31;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01g", &time);
  EXPECT_STREQ_LEN(written, buffer, "31");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%02g", &time);
  EXPECT_STREQ_LEN(written, buffer, "31");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%05g", &time);
  EXPECT_STREQ_LEN(written, buffer, "00031");
}

TEST(LlvmLibcStrftimeTest, ISOYear) {
  // this tests %G, which reads: [tm_year, tm_wday, tm_yday]

  // This stuff is all the same as above, but for brevity I'm not going to
  // duplicate all the comments explaining exactly how ISO years work. The
  // general comments are still here though.

  struct tm time;
  char buffer[100];
  size_t written = 0;
  SimplePaddedNum spn;

  // a sunday in the middle of the year. No need to worry about rounding
  time.tm_wday = 0;
  time.tm_yday = 100;

  // Test the easy cases
  for (int i = 1; i < 10000; ++i) {
    time.tm_year = get_adjusted_year(i);
    written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%G", &time);
    char *result = spn.get_padded_num(i, 4);

    ASSERT_STREQ(buffer, result);
    ASSERT_EQ(written, spn.get_str_len());
  }

  // also check it handles years with extra digits properly
  time.tm_year = get_adjusted_year(12345);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%G", &time);
  EXPECT_STREQ_LEN(written, buffer, "12345");

  // Test the harder to round cases

  // not a leap year. Not relevant for the start-of-year tests, but it does
  // matter for the end-of-year tests.
  time.tm_year = get_adjusted_year(1999);

  // check the first days of the year
  for (size_t yday = 0; yday < 5; ++yday) {
    for (size_t iso_wday = 1; iso_wday < 8; ++iso_wday) {
      // start with monday, to match the ISO week.
      time.tm_wday = iso_wday % LIBC_NAMESPACE::time_constants::DAYS_PER_WEEK;
      time.tm_yday = yday;

      written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%G", &time);

      if (iso_wday <= LIBC_NAMESPACE::time_constants::THURSDAY || yday >= 4) {
        // monday - thursday are never in the previous year, nor are the 4th and
        // after.
        EXPECT_STREQ_LEN(written, buffer, "1999");
      } else {
        // iso_wday is 5, 6, or 7 and yday is 0, 1, or 2.
        // days_since_thursday is therefor 1, 2, or 3.
        const size_t days_since_thursday =
            iso_wday - LIBC_NAMESPACE::time_constants::THURSDAY;

        if (days_since_thursday > yday) {
          EXPECT_STREQ_LEN(written, buffer, "1998");
        } else {
          EXPECT_STREQ_LEN(written, buffer, "1999");
        }
      }
    }
  }

  // set up all the extra stuff to cover leap years.
  struct tm time_leap_year;
  char buffer_leap_year[100];
  size_t written_leap_year = 0;
  time_leap_year = time;
  time_leap_year.tm_year = 100; // 2000 is a leap year.

  // check the last days of the year. Checking 5 to make sure all the leap year
  // cases are covered as well.
  for (size_t days_left = 0; days_left < 5; ++days_left) {
    for (size_t iso_wday = 1; iso_wday < 8; ++iso_wday) {
      // start with monday, to match the ISO week.
      time.tm_wday = iso_wday % LIBC_NAMESPACE::time_constants::DAYS_PER_WEEK;
      // subtract 1 from the max yday to handle yday being 0-indexed.
      time.tm_yday =
          LIBC_NAMESPACE::time_constants::LAST_DAY_OF_NON_LEAP_YEAR - days_left;

      time_leap_year.tm_wday =
          iso_wday % LIBC_NAMESPACE::time_constants::DAYS_PER_WEEK;
      time_leap_year.tm_yday =
          LIBC_NAMESPACE::time_constants::LAST_DAY_OF_LEAP_YEAR - days_left;

      written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%G", &time);
      written_leap_year = LIBC_NAMESPACE::strftime(
          buffer_leap_year, sizeof(buffer_leap_year), "%G", &time_leap_year);

      if (iso_wday >= 4 || days_left >= 3) {
        // thursday - sunday are never in the next year, nor are days more than
        // 3 days before the end.
        EXPECT_STREQ_LEN(written, buffer, "1999");
        EXPECT_STREQ_LEN(written_leap_year, buffer_leap_year, "2000");
      } else {
        // iso_wday is 1, 2 or 3 and days_left is 0, 1, or 2
        if (iso_wday + days_left <= 3) {
          EXPECT_STREQ_LEN(written, buffer, "2000");
          EXPECT_STREQ_LEN(written_leap_year, buffer_leap_year, "2001");
        } else {
          EXPECT_STREQ_LEN(written, buffer, "1999");
          EXPECT_STREQ_LEN(written_leap_year, buffer_leap_year, "2000");
        }
      }
    }
  }

  // padding is technically undefined for this conversion, but we support it, so
  // we need to test it.
  time.tm_year = get_adjusted_year(5);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01G", &time);
  EXPECT_STREQ_LEN(written, buffer, "5");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%02G", &time);
  EXPECT_STREQ_LEN(written, buffer, "05");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%05G", &time);
  EXPECT_STREQ_LEN(written, buffer, "00005");

  time.tm_year = get_adjusted_year(31);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01G", &time);
  EXPECT_STREQ_LEN(written, buffer, "31");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%02G", &time);
  EXPECT_STREQ_LEN(written, buffer, "31");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%05G", &time);
  EXPECT_STREQ_LEN(written, buffer, "00031");

  time.tm_year = get_adjusted_year(2001);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01G", &time);
  EXPECT_STREQ_LEN(written, buffer, "2001");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%02G", &time);
  EXPECT_STREQ_LEN(written, buffer, "2001");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%05G", &time);
  EXPECT_STREQ_LEN(written, buffer, "02001");
}

TEST(LlvmLibcStrftimeTest, TwentyFourHour) {
  // this tests %H, which reads: [tm_hour]
  struct tm time;
  char buffer[100];
  size_t written = 0;
  SimplePaddedNum spn;

  // Tests on all the well defined values
  for (size_t i = 0; i < 24; ++i) {
    time.tm_hour = i;
    written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%H", &time);
    char *result = spn.get_padded_num(i, 2);

    ASSERT_STREQ(buffer, result);
    ASSERT_EQ(written, spn.get_str_len());
  }

  // padding is technically undefined for this conversion, but we support it, so
  // we need to test it.
  time.tm_hour = 5;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01H", &time);
  EXPECT_STREQ_LEN(written, buffer, "5");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%02H", &time);
  EXPECT_STREQ_LEN(written, buffer, "05");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%05H", &time);
  EXPECT_STREQ_LEN(written, buffer, "00005");

  time.tm_hour = 23;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01H", &time);
  EXPECT_STREQ_LEN(written, buffer, "23");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%02H", &time);
  EXPECT_STREQ_LEN(written, buffer, "23");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%05H", &time);
  EXPECT_STREQ_LEN(written, buffer, "00023");
}

TEST(LlvmLibcStrftimeTest, TwelveHour) {
  // this tests %I, which reads: [tm_hour]
  struct tm time;
  char buffer[100];
  size_t written = 0;
  SimplePaddedNum spn;

  time.tm_hour = 0;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%I", &time);
  EXPECT_STREQ_LEN(written, buffer, "12");

  // Tests on all the well defined values, except 0 since it was easier to
  // special case it.
  for (size_t i = 1; i <= 12; ++i) {
    char *result = spn.get_padded_num(i, 2);

    time.tm_hour = i;
    written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%I", &time);
    ASSERT_STREQ(buffer, result);
    ASSERT_EQ(written, spn.get_str_len());

    // hour + 12 should give the same result
    time.tm_hour = i + 12;
    written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%I", &time);
    ASSERT_STREQ(buffer, result);
    ASSERT_EQ(written, spn.get_str_len());
  }

  // padding is technically undefined for this conversion, but we support it, so
  // we need to test it.
  time.tm_hour = 5;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01I", &time);
  EXPECT_STREQ_LEN(written, buffer, "5");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%02I", &time);
  EXPECT_STREQ_LEN(written, buffer, "05");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%05I", &time);
  EXPECT_STREQ_LEN(written, buffer, "00005");

  time.tm_hour = 23;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01I", &time);
  EXPECT_STREQ_LEN(written, buffer, "11");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%02I", &time);
  EXPECT_STREQ_LEN(written, buffer, "11");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%05I", &time);
  EXPECT_STREQ_LEN(written, buffer, "00011");
}

TEST(LlvmLibcStrftimeTest, DayOfYear) {
  // this tests %j, which reads: [tm_yday]
  struct tm time;
  char buffer[100];
  size_t written = 0;
  SimplePaddedNum spn;

  // Tests on all the well defined values
  for (size_t i = 0; i < LIBC_NAMESPACE::time_constants::DAYS_PER_LEAP_YEAR;
       ++i) {
    time.tm_yday = i;
    written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%j", &time);
    char *result = spn.get_padded_num(i + 1, 3);

    ASSERT_STREQ(buffer, result);
    ASSERT_EQ(written, spn.get_str_len());
  }

  // padding is technically undefined for this conversion, but we support it, so
  // we need to test it.
  time.tm_yday = 5 - 1;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01j", &time);
  EXPECT_STREQ_LEN(written, buffer, "5");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%02j", &time);
  EXPECT_STREQ_LEN(written, buffer, "05");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%05j", &time);
  EXPECT_STREQ_LEN(written, buffer, "00005");

  time.tm_yday = 123 - 1;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01j", &time);
  EXPECT_STREQ_LEN(written, buffer, "123");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%02j", &time);
  EXPECT_STREQ_LEN(written, buffer, "123");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%05j", &time);
  EXPECT_STREQ_LEN(written, buffer, "00123");
}

TEST(LlvmLibcStrftimeTest, MonthOfYear) {
  // this tests %m, which reads: [tm_mon]
  struct tm time;
  char buffer[100];
  size_t written = 0;
  SimplePaddedNum spn;

  // Tests on all the well defined values
  for (size_t i = 0; i < LIBC_NAMESPACE::time_constants::MONTHS_PER_YEAR; ++i) {
    time.tm_mon = i;
    written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%m", &time);
    // %m is 1 indexed, so add 1 to the number we're comparing to.
    char *result = spn.get_padded_num(i + 1, 2);

    ASSERT_STREQ(buffer, result);
    ASSERT_EQ(written, spn.get_str_len());
  }

  // padding is technically undefined for this conversion, but we support it, so
  // we need to test it.
  time.tm_mon = 5 - 1;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01m", &time);
  EXPECT_STREQ_LEN(written, buffer, "5");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%02m", &time);
  EXPECT_STREQ_LEN(written, buffer, "05");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%05m", &time);
  EXPECT_STREQ_LEN(written, buffer, "00005");

  time.tm_mon = 11 - 1;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01m", &time);
  EXPECT_STREQ_LEN(written, buffer, "11");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%02m", &time);
  EXPECT_STREQ_LEN(written, buffer, "11");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%05m", &time);
  EXPECT_STREQ_LEN(written, buffer, "00011");
}

TEST(LlvmLibcStrftimeTest, MinuteOfHour) {
  // this tests %M, which reads: [tm_min]
  struct tm time;
  char buffer[100];
  size_t written = 0;
  SimplePaddedNum spn;

  // Tests on all the well defined values
  for (size_t i = 0; i < LIBC_NAMESPACE::time_constants::MINUTES_PER_HOUR;
       ++i) {
    time.tm_min = i;
    written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%M", &time);
    char *result = spn.get_padded_num(i, 2);

    ASSERT_STREQ(buffer, result);
    ASSERT_EQ(written, spn.get_str_len());
  }

  // padding is technically undefined for this conversion, but we support it, so
  // we need to test it.
  time.tm_min = 5;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01M", &time);
  EXPECT_STREQ_LEN(written, buffer, "5");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%02M", &time);
  EXPECT_STREQ_LEN(written, buffer, "05");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%05M", &time);
  EXPECT_STREQ_LEN(written, buffer, "00005");

  time.tm_min = 11;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01M", &time);
  EXPECT_STREQ_LEN(written, buffer, "11");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%02M", &time);
  EXPECT_STREQ_LEN(written, buffer, "11");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%05M", &time);
  EXPECT_STREQ_LEN(written, buffer, "00011");
}

TEST(LlvmLibcStrftimeTest, SecondsSinceEpoch) {
  // this tests %s, which reads: [tm_year, tm_mon, tm_mday, tm_hour, tm_min,
  // tm_sec, tm_isdst]
  struct tm time;
  char buffer[100];
  size_t written = 0;

  time.tm_year = get_adjusted_year(1970);
  // yday is not used, the day of the year is calculated from the month and mday
  time.tm_mon = 0;
  time.tm_mday = 1; // the only 1-indexed member
  time.tm_hour = 0;
  time.tm_min = 0;
  time.tm_sec = 1;
  time.tm_isdst = 0;

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%s", &time);
  EXPECT_STREQ_LEN(written, buffer, "1");

  // The time as of writing this test
  time.tm_year = get_adjusted_year(2025);
  time.tm_mon = 1;
  time.tm_mday = 4;
  time.tm_hour = 11;
  time.tm_min = 8;
  time.tm_sec = 41;
  time.tm_isdst = 0;

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%s", &time);
  // if you run your system's strftime to compare you will likely get a slightly
  // different result because it's supposed to respect timezones.
  EXPECT_STREQ_LEN(written, buffer, "1738667321");

  // Thorough testing of the mktime mechanism is done in the mktime tests, so
  // they aren't duplicated here.

  // padding is technically undefined for this conversion, but we support it, so
  // we need to test it.
  time.tm_year = get_adjusted_year(1970);
  time.tm_mon = 0;
  time.tm_mday = 1;
  time.tm_hour = 0;
  time.tm_min = 0;
  time.tm_sec = 5;
  time.tm_isdst = 0;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01s", &time);
  EXPECT_STREQ_LEN(written, buffer, "5");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%02s", &time);
  EXPECT_STREQ_LEN(written, buffer, "05");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%05s", &time);
  EXPECT_STREQ_LEN(written, buffer, "00005");

  time.tm_min = 11;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01s", &time);
  EXPECT_STREQ_LEN(written, buffer, "665");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%02s", &time);
  EXPECT_STREQ_LEN(written, buffer, "665");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%05s", &time);
  EXPECT_STREQ_LEN(written, buffer, "00665");
}

TEST(LlvmLibcStrftimeTest, SecondOfMinute) {
  // this tests %S, which reads: [tm_sec]
  struct tm time;
  char buffer[100];
  size_t written = 0;
  SimplePaddedNum spn;

  // Tests on all the well defined values
  for (size_t i = 0; i < LIBC_NAMESPACE::time_constants::SECONDS_PER_MIN; ++i) {
    time.tm_sec = i;
    written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%S", &time);
    char *result = spn.get_padded_num(i, 2);

    ASSERT_STREQ(buffer, result);
    ASSERT_EQ(written, spn.get_str_len());
  }

  // padding is technically undefined for this conversion, but we support it, so
  // we need to test it.
  time.tm_sec = 5;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01S", &time);
  EXPECT_STREQ_LEN(written, buffer, "5");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%02S", &time);
  EXPECT_STREQ_LEN(written, buffer, "05");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%05S", &time);
  EXPECT_STREQ_LEN(written, buffer, "00005");

  time.tm_sec = 11;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01S", &time);
  EXPECT_STREQ_LEN(written, buffer, "11");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%02S", &time);
  EXPECT_STREQ_LEN(written, buffer, "11");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%05S", &time);
  EXPECT_STREQ_LEN(written, buffer, "00011");
}

TEST(LlvmLibcStrftimeTest, ISODayOfWeek) {
  // this tests %u, which reads: [tm_wday]
  struct tm time;
  char buffer[100];
  size_t written = 0;
  SimplePaddedNum spn;

  time.tm_wday = LIBC_NAMESPACE::time_constants::SUNDAY;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%u", &time);
  EXPECT_STREQ_LEN(written, buffer, "7");

  // Tests on all the well defined values except for sunday, which is 0 in
  // normal weekdays but 7 here.
  for (size_t i = LIBC_NAMESPACE::time_constants::MONDAY;
       i <= LIBC_NAMESPACE::time_constants::SATURDAY; ++i) {
    time.tm_wday = i;
    written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%u", &time);
    char *result = spn.get_padded_num(i, 1);

    ASSERT_STREQ(buffer, result);
    ASSERT_EQ(written, spn.get_str_len());
  }

  // padding is technically undefined for this conversion, but we support it, so
  // we need to test it.
  time.tm_wday = 5;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01u", &time);
  EXPECT_STREQ_LEN(written, buffer, "5");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%02u", &time);
  EXPECT_STREQ_LEN(written, buffer, "05");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%05u", &time);
  EXPECT_STREQ_LEN(written, buffer, "00005");
}

TEST(LlvmLibcStrftimeTest, WeekOfYearStartingSunday) {
  // this tests %U, which reads: [tm_year, tm_wday, tm_yday]
  struct tm time;
  char buffer[100];
  size_t written = 0;
  SimplePaddedNum spn;

  // setting the year to a leap year, but it doesn't actually matter. This
  // conversion doesn't end up checking the year at all.
  time.tm_year = get_adjusted_year(2000);

  const int WEEK_START = LIBC_NAMESPACE::time_constants::SUNDAY;

  for (size_t first_weekday = LIBC_NAMESPACE::time_constants::SUNDAY;
       first_weekday <= LIBC_NAMESPACE::time_constants::SATURDAY;
       ++first_weekday) {
    time.tm_wday = first_weekday;
    size_t cur_week = 0;

    // iterate through the year, starting on first_weekday.
    for (size_t yday = 0;
         yday < LIBC_NAMESPACE::time_constants::DAYS_PER_LEAP_YEAR; ++yday) {
      time.tm_yday = yday;
      // If the week just ended, move to the next week.
      if (time.tm_wday == WEEK_START)
        ++cur_week;

      written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%U", &time);
      char *result = spn.get_padded_num(cur_week, 2);

      ASSERT_STREQ(buffer, result);
      ASSERT_EQ(written, spn.get_str_len());

      // a day has passed, move to the next weekday, looping as necessary.
      time.tm_wday =
          (time.tm_wday + 1) % LIBC_NAMESPACE::time_constants::DAYS_PER_WEEK;
    }
  }

  // padding is technically undefined for this conversion, but we support it, so
  // we need to test it.
  time.tm_wday = LIBC_NAMESPACE::time_constants::SUNDAY;
  time.tm_yday = 22;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01U", &time);
  EXPECT_STREQ_LEN(written, buffer, "4");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%02U", &time);
  EXPECT_STREQ_LEN(written, buffer, "04");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%05U", &time);
  EXPECT_STREQ_LEN(written, buffer, "00004");

  time.tm_wday = LIBC_NAMESPACE::time_constants::SUNDAY;
  time.tm_yday = 78;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01U", &time);
  EXPECT_STREQ_LEN(written, buffer, "12");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%02U", &time);
  EXPECT_STREQ_LEN(written, buffer, "12");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%05U", &time);
  EXPECT_STREQ_LEN(written, buffer, "00012");
}

TEST(LlvmLibcStrftimeTest, ISOWeekOfYear) {
  // this tests %V, which reads: [tm_year, tm_wday, tm_yday]
  struct tm time;
  char buffer[100];
  size_t written = 0;
  SimplePaddedNum spn;

  const int starting_year = get_adjusted_year(1999);

  // we're going to check the days from 1999 to 2001 to cover all the
  // transitions to and from leap years and non-leap years (the start of 1999
  // and end of 2001 cover the non-leap years).
  const int days_to_check = // 1096
      LIBC_NAMESPACE::time_constants::DAYS_PER_NON_LEAP_YEAR +
      LIBC_NAMESPACE::time_constants::DAYS_PER_LEAP_YEAR +
      LIBC_NAMESPACE::time_constants::DAYS_PER_NON_LEAP_YEAR;

  const int WEEK_START = LIBC_NAMESPACE::time_constants::MONDAY;

  for (size_t first_weekday = LIBC_NAMESPACE::time_constants::SUNDAY;
       first_weekday <= LIBC_NAMESPACE::time_constants::SATURDAY;
       ++first_weekday) {
    time.tm_year = starting_year;
    time.tm_wday = first_weekday;
    time.tm_yday = 0;
    size_t cur_week = 1;
    if (first_weekday == LIBC_NAMESPACE::time_constants::SUNDAY ||
        first_weekday == LIBC_NAMESPACE::time_constants::SATURDAY)
      cur_week = 52;
    else if (first_weekday == LIBC_NAMESPACE::time_constants::FRIDAY)
      cur_week = 53;

    // iterate through the year, starting on first_weekday.
    for (size_t cur_day = 0; cur_day < days_to_check; ++cur_day) {
      // If the week just ended, move to the next week.

      written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%V", &time);
      char *result = spn.get_padded_num(cur_week, 2);

      ASSERT_STREQ(buffer, result);
      ASSERT_EQ(written, spn.get_str_len());

      // a day has passed, increment the counters.
      ++time.tm_yday;
      if (time.tm_yday ==
          (time.tm_year == get_adjusted_year(2000)
               ? LIBC_NAMESPACE::time_constants::DAYS_PER_LEAP_YEAR
               : LIBC_NAMESPACE::time_constants::DAYS_PER_NON_LEAP_YEAR)) {
        time.tm_yday = 0;
        ++time.tm_year;
      }

      time.tm_wday =
          (time.tm_wday + 1) % LIBC_NAMESPACE::time_constants::DAYS_PER_WEEK;
      if (time.tm_wday == WEEK_START) {
        ++cur_week;
        const int days_left_in_year =
            (time.tm_year == get_adjusted_year(2000)
                 ? LIBC_NAMESPACE::time_constants::LAST_DAY_OF_LEAP_YEAR
                 : LIBC_NAMESPACE::time_constants::LAST_DAY_OF_NON_LEAP_YEAR) -
            time.tm_yday;

        // if the week we're currently in is in the next year, or if the year
        // has turned over, reset the week.
        if (days_left_in_year < 3 || (cur_week > 51 && time.tm_yday < 10))
          cur_week = 1;
      }
    }
  }

  // padding is technically undefined for this conversion, but we support it, so
  // we need to test it.
  time.tm_wday = LIBC_NAMESPACE::time_constants::SUNDAY;
  time.tm_yday = 22;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01U", &time);
  EXPECT_STREQ_LEN(written, buffer, "4");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%02U", &time);
  EXPECT_STREQ_LEN(written, buffer, "04");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%05U", &time);
  EXPECT_STREQ_LEN(written, buffer, "00004");

  time.tm_wday = LIBC_NAMESPACE::time_constants::SUNDAY;
  time.tm_yday = 78;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01U", &time);
  EXPECT_STREQ_LEN(written, buffer, "12");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%02U", &time);
  EXPECT_STREQ_LEN(written, buffer, "12");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%05U", &time);
  EXPECT_STREQ_LEN(written, buffer, "00012");
}

TEST(LlvmLibcStrftimeTest, DayOfWeek) {
  // this tests %w, which reads: [tm_wday]
  struct tm time;
  char buffer[100];
  size_t written = 0;
  SimplePaddedNum spn;

  // Tests on all the well defined values.
  for (size_t i = LIBC_NAMESPACE::time_constants::SUNDAY;
       i <= LIBC_NAMESPACE::time_constants::SATURDAY; ++i) {
    time.tm_wday = i;
    written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%w", &time);
    char *result = spn.get_padded_num(i, 1);

    ASSERT_STREQ(buffer, result);
    ASSERT_EQ(written, spn.get_str_len());
  }

  // padding is technically undefined for this conversion, but we support it, so
  // we need to test it.
  time.tm_wday = 5;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01w", &time);
  EXPECT_STREQ_LEN(written, buffer, "5");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%02w", &time);
  EXPECT_STREQ_LEN(written, buffer, "05");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%05w", &time);
  EXPECT_STREQ_LEN(written, buffer, "00005");
}

TEST(LlvmLibcStrftimeTest, WeekOfYearStartingMonday) {
  // this tests %W, which reads: [tm_year, tm_wday, tm_yday]
  struct tm time;
  char buffer[100];
  size_t written = 0;
  SimplePaddedNum spn;

  // setting the year to a leap year, but it doesn't actually matter. This
  // conversion doesn't end up checking the year at all.
  time.tm_year = get_adjusted_year(2000);

  const int WEEK_START = LIBC_NAMESPACE::time_constants::MONDAY;

  for (size_t first_weekday = LIBC_NAMESPACE::time_constants::SUNDAY;
       first_weekday <= LIBC_NAMESPACE::time_constants::SATURDAY;
       ++first_weekday) {
    time.tm_wday = first_weekday;
    size_t cur_week = 0;

    // iterate through the year, starting on first_weekday.
    for (size_t yday = 0;
         yday < LIBC_NAMESPACE::time_constants::DAYS_PER_LEAP_YEAR; ++yday) {
      time.tm_yday = yday;
      // If the week just ended, move to the next week.
      if (time.tm_wday == WEEK_START)
        ++cur_week;

      written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%W", &time);
      char *result = spn.get_padded_num(cur_week, 2);

      ASSERT_STREQ(buffer, result);
      ASSERT_EQ(written, spn.get_str_len());

      // a day has passed, move to the next weekday, looping as necessary.
      time.tm_wday =
          (time.tm_wday + 1) % LIBC_NAMESPACE::time_constants::DAYS_PER_WEEK;
    }
  }

  // padding is technically undefined for this conversion, but we support it, so
  // we need to test it.
  time.tm_wday = LIBC_NAMESPACE::time_constants::MONDAY;
  time.tm_yday = 22;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01W", &time);
  EXPECT_STREQ_LEN(written, buffer, "4");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%02W", &time);
  EXPECT_STREQ_LEN(written, buffer, "04");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%05W", &time);
  EXPECT_STREQ_LEN(written, buffer, "00004");

  time.tm_wday = LIBC_NAMESPACE::time_constants::MONDAY;
  time.tm_yday = 78;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01W", &time);
  EXPECT_STREQ_LEN(written, buffer, "12");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%02W", &time);
  EXPECT_STREQ_LEN(written, buffer, "12");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%05W", &time);
  EXPECT_STREQ_LEN(written, buffer, "00012");
}

TEST(LlvmLibcStrftimeTest, YearOfCentury) {
  // this tests %y, which reads: [tm_year]
  struct tm time;
  char buffer[100];
  size_t written = 0;
  SimplePaddedNum spn;

  time.tm_year = get_adjusted_year(2000);

  // iterate through the year, starting on first_weekday.
  for (size_t year = 1900; year < 2001; ++year) {
    time.tm_year = get_adjusted_year(year);

    written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%y", &time);
    char *result = spn.get_padded_num(year % 100, 2);

    ASSERT_STREQ(buffer, result);
    ASSERT_EQ(written, spn.get_str_len());
  }

  // padding is technically undefined for this conversion, but we support it, so
  // we need to test it.
  time.tm_year = get_adjusted_year(2004);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01y", &time);
  EXPECT_STREQ_LEN(written, buffer, "4");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%02y", &time);
  EXPECT_STREQ_LEN(written, buffer, "04");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%05y", &time);
  EXPECT_STREQ_LEN(written, buffer, "00004");

  time.tm_year = get_adjusted_year(12345);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01y", &time);
  EXPECT_STREQ_LEN(written, buffer, "45");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%02y", &time);
  EXPECT_STREQ_LEN(written, buffer, "45");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%05y", &time);
  EXPECT_STREQ_LEN(written, buffer, "00045");
}

TEST(LlvmLibcStrftimeTest, FullYearTests) {
  // this tests %Y, which reads: [tm_year]
  struct tm time;
  char buffer[100];
  size_t written = 0;
  SimplePaddedNum spn;

  // Test the easy cases
  for (int i = 1; i < 10000; ++i) {
    time.tm_year = get_adjusted_year(i);
    written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%Y", &time);
    char *result = spn.get_padded_num(i, 4);

    ASSERT_STREQ(buffer, result);
    ASSERT_EQ(written, spn.get_str_len());
  }

  time.tm_year = get_adjusted_year(11900);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "11900");

  time.tm_year = get_adjusted_year(0);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "0000");

  time.tm_year = get_adjusted_year(-1);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%Y", &time);
  // TODO: should this be what we standardize? Posix doesn't specify what to do
  // about negative numbers
  EXPECT_STREQ_LEN(written, buffer, "-001");

  time.tm_year = get_adjusted_year(-9001);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "-9001");

  time.tm_year = get_adjusted_year(-10001);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "-10001");

  // width tests (with the 0 flag, since the default padding is undefined).
  time.tm_year = get_adjusted_year(2023);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "2023");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%04Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "2023");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%05Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "02023");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%010Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "0000002023");

  time.tm_year = get_adjusted_year(900);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "900");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%04Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "0900");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%05Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "00900");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%010Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "0000000900");

  time.tm_year = get_adjusted_year(12345);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "12345");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%04Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "12345");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%05Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "12345");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%010Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "0000012345");

  time.tm_year = get_adjusted_year(-123);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "-123");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%04Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "-123");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%05Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "-0123");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%010Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "-000000123");

  // '+' flag tests
  time.tm_year = get_adjusted_year(2023);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+1Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "2023");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+4Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "2023");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+5Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "+2023");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+10Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "+000002023");

  time.tm_year = get_adjusted_year(900);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+1Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "900");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+4Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "0900");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+5Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "+0900");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+10Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "+000000900");

  time.tm_year = get_adjusted_year(12345);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+1Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "+12345");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+4Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "+12345");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+5Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "+12345");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+10Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "+000012345");

  time.tm_year = get_adjusted_year(-123);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+1Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "-123");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+4Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "-123");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+5Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "-0123");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+10Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "-000000123");

  // Posix specified tests:
  time.tm_year = get_adjusted_year(1970);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "1970");

  time.tm_year = get_adjusted_year(1970);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+4Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "1970");

  time.tm_year = get_adjusted_year(27);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "0027");

  time.tm_year = get_adjusted_year(270);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "0270");

  time.tm_year = get_adjusted_year(270);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+4Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "0270");

  time.tm_year = get_adjusted_year(12345);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "12345");

  time.tm_year = get_adjusted_year(12345);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+4Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "+12345");

  time.tm_year = get_adjusted_year(12345);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%05Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "12345");

  time.tm_year = get_adjusted_year(270);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+5Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "+0270");

  time.tm_year = get_adjusted_year(12345);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+5Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "+12345");

  time.tm_year = get_adjusted_year(12345);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%06Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "012345");

  time.tm_year = get_adjusted_year(12345);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+6Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "+12345");

  time.tm_year = get_adjusted_year(123456);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%08Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "00123456");

  time.tm_year = get_adjusted_year(123456);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+8Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "+0123456");
}

// String conversions

struct num_str_pair {
  int num;
  LIBC_NAMESPACE::cpp::string_view str;
};

TEST(LlvmLibcStrftimeTest, ShortWeekdayName) {
  // this tests %a, which reads: [tm_wday]
  struct tm time;
  char buffer[100];
  size_t written = 0;

  constexpr LIBC_NAMESPACE::cpp::array<
      num_str_pair, LIBC_NAMESPACE::time_constants::DAYS_PER_WEEK>
      WEEKDAY_PAIRS = {{
          {LIBC_NAMESPACE::time_constants::SUNDAY, "Sun"},
          {LIBC_NAMESPACE::time_constants::MONDAY, "Mon"},
          {LIBC_NAMESPACE::time_constants::TUESDAY, "Tue"},
          {LIBC_NAMESPACE::time_constants::WEDNESDAY, "Wed"},
          {LIBC_NAMESPACE::time_constants::THURSDAY, "Thu"},
          {LIBC_NAMESPACE::time_constants::FRIDAY, "Fri"},
          {LIBC_NAMESPACE::time_constants::SATURDAY, "Sat"},
      }};

  for (size_t i = 0; i < WEEKDAY_PAIRS.size(); ++i) {
    time.tm_wday = WEEKDAY_PAIRS[i].num;
    written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%a", &time);
    EXPECT_STREQ(buffer, WEEKDAY_PAIRS[i].str.data());
    EXPECT_EQ(written, WEEKDAY_PAIRS[i].str.size());
  }

  // check invalid weekdays
  time.tm_wday = -1;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%a", &time);
  EXPECT_STREQ_LEN(written, buffer, "?");

  time.tm_wday = LIBC_NAMESPACE::time_constants::SATURDAY + 1;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%a", &time);
  EXPECT_STREQ_LEN(written, buffer, "?");

  // padding is technically undefined for this conversion, but we support it, so
  // we need to test it.
  time.tm_wday = LIBC_NAMESPACE::time_constants::THURSDAY;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%1a", &time);
  EXPECT_STREQ_LEN(written, buffer, "Thu");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%3a", &time);
  EXPECT_STREQ_LEN(written, buffer, "Thu");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%10a", &time);
  EXPECT_STREQ_LEN(written, buffer, "       Thu");

  time.tm_wday = -1;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%1a", &time);
  EXPECT_STREQ_LEN(written, buffer, "?");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%3a", &time);
  EXPECT_STREQ_LEN(written, buffer, "  ?");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%10a", &time);
  EXPECT_STREQ_LEN(written, buffer, "         ?");
}

TEST(LlvmLibcStrftimeTest, FullWeekdayName) {
  // this tests %a, which reads: [tm_wday]
  struct tm time;
  char buffer[100];
  size_t written = 0;

  constexpr LIBC_NAMESPACE::cpp::array<
      num_str_pair, LIBC_NAMESPACE::time_constants::DAYS_PER_WEEK>
      WEEKDAY_PAIRS = {{
          {LIBC_NAMESPACE::time_constants::SUNDAY, "Sunday"},
          {LIBC_NAMESPACE::time_constants::MONDAY, "Monday"},
          {LIBC_NAMESPACE::time_constants::TUESDAY, "Tuesday"},
          {LIBC_NAMESPACE::time_constants::WEDNESDAY, "Wednesday"},
          {LIBC_NAMESPACE::time_constants::THURSDAY, "Thursday"},
          {LIBC_NAMESPACE::time_constants::FRIDAY, "Friday"},
          {LIBC_NAMESPACE::time_constants::SATURDAY, "Saturday"},
      }};

  for (size_t i = 0; i < WEEKDAY_PAIRS.size(); ++i) {
    time.tm_wday = WEEKDAY_PAIRS[i].num;
    written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%A", &time);
    EXPECT_STREQ(buffer, WEEKDAY_PAIRS[i].str.data());
    EXPECT_EQ(written, WEEKDAY_PAIRS[i].str.size());
  }

  // check invalid weekdays
  time.tm_wday = -1;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%A", &time);
  EXPECT_STREQ_LEN(written, buffer, "?");

  time.tm_wday = LIBC_NAMESPACE::time_constants::SATURDAY + 1;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%A", &time);
  EXPECT_STREQ_LEN(written, buffer, "?");

  // padding is technically undefined for this conversion, but we support it, so
  // we need to test it.
  time.tm_wday = LIBC_NAMESPACE::time_constants::THURSDAY;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%1A", &time);
  EXPECT_STREQ_LEN(written, buffer, "Thursday");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%3A", &time);
  EXPECT_STREQ_LEN(written, buffer, "Thursday");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%10A", &time);
  EXPECT_STREQ_LEN(written, buffer, "  Thursday");

  time.tm_wday = -1;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%1A", &time);
  EXPECT_STREQ_LEN(written, buffer, "?");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%3A", &time);
  EXPECT_STREQ_LEN(written, buffer, "  ?");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%10A", &time);
  EXPECT_STREQ_LEN(written, buffer, "         ?");
}

TEST(LlvmLibcStrftimeTest, ShortMonthName) {
  // this tests %b, which reads: [tm_mon]
  struct tm time;
  char buffer[100];
  size_t written = 0;

  constexpr LIBC_NAMESPACE::cpp::array<
      num_str_pair, LIBC_NAMESPACE::time_constants::MONTHS_PER_YEAR>
      MONTH_PAIRS = {{
          {LIBC_NAMESPACE::time_constants::JANUARY, "Jan"},
          {LIBC_NAMESPACE::time_constants::FEBRUARY, "Feb"},
          {LIBC_NAMESPACE::time_constants::MARCH, "Mar"},
          {LIBC_NAMESPACE::time_constants::APRIL, "Apr"},
          {LIBC_NAMESPACE::time_constants::MAY, "May"},
          {LIBC_NAMESPACE::time_constants::JUNE, "Jun"},
          {LIBC_NAMESPACE::time_constants::JULY, "Jul"},
          {LIBC_NAMESPACE::time_constants::AUGUST, "Aug"},
          {LIBC_NAMESPACE::time_constants::SEPTEMBER, "Sep"},
          {LIBC_NAMESPACE::time_constants::OCTOBER, "Oct"},
          {LIBC_NAMESPACE::time_constants::NOVEMBER, "Nov"},
          {LIBC_NAMESPACE::time_constants::DECEMBER, "Dec"},
      }};

  for (size_t i = 0; i < MONTH_PAIRS.size(); ++i) {
    time.tm_mon = MONTH_PAIRS[i].num;
    written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%b", &time);
    EXPECT_STREQ(buffer, MONTH_PAIRS[i].str.data());
    EXPECT_EQ(written, MONTH_PAIRS[i].str.size());
  }

  // check invalid weekdays
  time.tm_mon = -1;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%b", &time);
  EXPECT_STREQ_LEN(written, buffer, "?");

  time.tm_mon = LIBC_NAMESPACE::time_constants::DECEMBER + 1;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%b", &time);
  EXPECT_STREQ_LEN(written, buffer, "?");

  // Also test %h, which is identical to %b
  time.tm_mon = LIBC_NAMESPACE::time_constants::OCTOBER;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%h", &time);
  EXPECT_STREQ_LEN(written, buffer, "Oct");

  // padding is technically undefined for this conversion, but we support it, so
  // we need to test it.
  time.tm_mon = LIBC_NAMESPACE::time_constants::OCTOBER;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%1b", &time);
  EXPECT_STREQ_LEN(written, buffer, "Oct");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%3b", &time);
  EXPECT_STREQ_LEN(written, buffer, "Oct");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%10b", &time);
  EXPECT_STREQ_LEN(written, buffer, "       Oct");

  time.tm_mon = -1;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%1b", &time);
  EXPECT_STREQ_LEN(written, buffer, "?");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%3b", &time);
  EXPECT_STREQ_LEN(written, buffer, "  ?");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%10b", &time);
  EXPECT_STREQ_LEN(written, buffer, "         ?");
}

TEST(LlvmLibcStrftimeTest, FullMonthName) {
  // this tests %B, which reads: [tm_mon]
  struct tm time;
  char buffer[100];
  size_t written = 0;

  constexpr LIBC_NAMESPACE::cpp::array<
      num_str_pair, LIBC_NAMESPACE::time_constants::MONTHS_PER_YEAR>
      MONTH_PAIRS = {{
          {LIBC_NAMESPACE::time_constants::JANUARY, "January"},
          {LIBC_NAMESPACE::time_constants::FEBRUARY, "February"},
          {LIBC_NAMESPACE::time_constants::MARCH, "March"},
          {LIBC_NAMESPACE::time_constants::APRIL, "April"},
          {LIBC_NAMESPACE::time_constants::MAY, "May"},
          {LIBC_NAMESPACE::time_constants::JUNE, "June"},
          {LIBC_NAMESPACE::time_constants::JULY, "July"},
          {LIBC_NAMESPACE::time_constants::AUGUST, "August"},
          {LIBC_NAMESPACE::time_constants::SEPTEMBER, "September"},
          {LIBC_NAMESPACE::time_constants::OCTOBER, "October"},
          {LIBC_NAMESPACE::time_constants::NOVEMBER, "November"},
          {LIBC_NAMESPACE::time_constants::DECEMBER, "December"},
      }};

  for (size_t i = 0; i < MONTH_PAIRS.size(); ++i) {
    time.tm_mon = MONTH_PAIRS[i].num;
    written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%B", &time);
    EXPECT_STREQ(buffer, MONTH_PAIRS[i].str.data());
    EXPECT_EQ(written, MONTH_PAIRS[i].str.size());
  }

  // check invalid weekdays
  time.tm_mon = -1;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%B", &time);
  EXPECT_STREQ_LEN(written, buffer, "?");

  time.tm_mon = LIBC_NAMESPACE::time_constants::DECEMBER + 1;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%B", &time);
  EXPECT_STREQ_LEN(written, buffer, "?");

  // padding is technically undefined for this conversion, but we support it, so
  // we need to test it.
  time.tm_mon = LIBC_NAMESPACE::time_constants::OCTOBER;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%1B", &time);
  EXPECT_STREQ_LEN(written, buffer, "October");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%3B", &time);
  EXPECT_STREQ_LEN(written, buffer, "October");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%10B", &time);
  EXPECT_STREQ_LEN(written, buffer, "   October");

  time.tm_mon = -1;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%1B", &time);
  EXPECT_STREQ_LEN(written, buffer, "?");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%3B", &time);
  EXPECT_STREQ_LEN(written, buffer, "  ?");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%10B", &time);
  EXPECT_STREQ_LEN(written, buffer, "         ?");
}

TEST(LlvmLibcStrftimeTest, AM_PM) {
  // this tests %p, which reads: [tm_hour]
  struct tm time;
  char buffer[100];
  size_t written = 0;

  time.tm_hour = 0;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%p", &time);
  EXPECT_STREQ_LEN(written, buffer, "AM");

  time.tm_hour = 6;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%p", &time);
  EXPECT_STREQ_LEN(written, buffer, "AM");

  time.tm_hour = 12;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%p", &time);
  EXPECT_STREQ_LEN(written, buffer, "PM");

  time.tm_hour = 18;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%p", &time);
  EXPECT_STREQ_LEN(written, buffer, "PM");

  // padding is technically undefined for this conversion, but we support it, so
  // we need to test it.
  time.tm_hour = 6;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%1p", &time);
  EXPECT_STREQ_LEN(written, buffer, "AM");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%2p", &time);
  EXPECT_STREQ_LEN(written, buffer, "AM");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%10p", &time);
  EXPECT_STREQ_LEN(written, buffer, "        AM");

  time.tm_hour = 18;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%1p", &time);
  EXPECT_STREQ_LEN(written, buffer, "PM");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%2p", &time);
  EXPECT_STREQ_LEN(written, buffer, "PM");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%10p", &time);
  EXPECT_STREQ_LEN(written, buffer, "        PM");
}

TEST(LlvmLibcStrftimeTest, DateFormatUS) {
  // this tests %D, which reads: [tm_mon, tm_mday, tm_year]
  // This is equivalent to "%m/%d/%y"
  struct tm time;
  char buffer[100];
  size_t written = 0;

  // each of %m, %d, and %y have their own tests, so this test won't cover all
  // values of those. Instead it will do basic tests and focus on the specific
  // padding behavior.

  time.tm_mon = 0;  // 0 indexed, so 0 is january
  time.tm_mday = 2; // 1 indexed, so 2 is the 2nd
  time.tm_year = get_adjusted_year(1903);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%D", &time);
  EXPECT_STREQ_LEN(written, buffer, "01/02/03");

  time.tm_mon = 11;
  time.tm_mday = 31;
  time.tm_year = get_adjusted_year(1999);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%D", &time);
  EXPECT_STREQ_LEN(written, buffer, "12/31/99");

  // The day LLVM-libc started
  time.tm_mon = 8;
  time.tm_mday = 16;
  time.tm_year = get_adjusted_year(2019);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%D", &time);
  EXPECT_STREQ_LEN(written, buffer, "09/16/19");

  // %x is equivalent to %D in default locale
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%x", &time);
  EXPECT_STREQ_LEN(written, buffer, "09/16/19");

  // padding is technically undefined for this conversion, but we support it, so
  // we need to test it.
  // Padding is handled in the same way as POSIX describes for %F
  time.tm_mon = 1;
  time.tm_mday = 5;
  time.tm_year = get_adjusted_year(2025);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01D", &time);
  EXPECT_STREQ_LEN(written, buffer, "2/05/25");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%07D", &time);
  EXPECT_STREQ_LEN(written, buffer, "2/05/25");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%010D", &time);
  EXPECT_STREQ_LEN(written, buffer, "0002/05/25");

  time.tm_mon = 9;
  time.tm_mday = 2;
  time.tm_year = get_adjusted_year(2000);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01D", &time);
  EXPECT_STREQ_LEN(written, buffer, "10/02/00");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%07D", &time);
  EXPECT_STREQ_LEN(written, buffer, "10/02/00");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%010D", &time);
  EXPECT_STREQ_LEN(written, buffer, "0010/02/00");
}

TEST(LlvmLibcStrftimeTest, DateFormatISO) {
  // this tests %F, which reads: [tm_year, tm_mon, tm_mday]
  // This is equivalent to "%Y-%m-%d"
  struct tm time;
  char buffer[100];
  size_t written = 0;

  // each of %Y, %m, and %d have their own tests, so this test won't cover all
  // values of those. Instead it will do basic tests and focus on the specific
  // padding behavior.

  time.tm_year = get_adjusted_year(1901);
  time.tm_mon = 1;  // 0 indexed, so 1 is february
  time.tm_mday = 3; // 1 indexed, so 2 is the 2nd
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%F", &time);
  EXPECT_STREQ_LEN(written, buffer, "1901-02-03");

  time.tm_year = get_adjusted_year(1999);
  time.tm_mon = 11;
  time.tm_mday = 31;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%F", &time);
  EXPECT_STREQ_LEN(written, buffer, "1999-12-31");

  time.tm_year = get_adjusted_year(2019);
  time.tm_mon = 8;
  time.tm_mday = 16;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%F", &time);
  EXPECT_STREQ_LEN(written, buffer, "2019-09-16");

  time.tm_year = get_adjusted_year(123);
  time.tm_mon = 3;
  time.tm_mday = 5;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%F", &time);
  EXPECT_STREQ_LEN(written, buffer, "0123-04-05");

  time.tm_year = get_adjusted_year(67);
  time.tm_mon = 7;
  time.tm_mday = 9;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%F", &time);
  EXPECT_STREQ_LEN(written, buffer, "0067-08-09");

  time.tm_year = get_adjusted_year(2);
  time.tm_mon = 1;
  time.tm_mday = 14;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%F", &time);
  EXPECT_STREQ_LEN(written, buffer, "0002-02-14");

  time.tm_year = get_adjusted_year(-543);
  time.tm_mon = 1;
  time.tm_mday = 1;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%F", &time);
  EXPECT_STREQ_LEN(written, buffer, "-543-02-01");

  // padding tests
  time.tm_year = get_adjusted_year(2025);
  time.tm_mon = 1;
  time.tm_mday = 5;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01F", &time);
  EXPECT_STREQ_LEN(written, buffer, "2025-02-05");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%010F", &time);
  EXPECT_STREQ_LEN(written, buffer, "2025-02-05");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%012F", &time);
  EXPECT_STREQ_LEN(written, buffer, "002025-02-05");

  time.tm_year = get_adjusted_year(12345);
  time.tm_mon = 11;
  time.tm_mday = 25;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01F", &time);
  EXPECT_STREQ_LEN(written, buffer, "12345-12-25");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%010F", &time);
  EXPECT_STREQ_LEN(written, buffer, "12345-12-25");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%012F", &time);
  EXPECT_STREQ_LEN(written, buffer, "012345-12-25");

  time.tm_year = get_adjusted_year(476);
  time.tm_mon = 8;
  time.tm_mday = 4;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01F", &time);
  EXPECT_STREQ_LEN(written, buffer, "476-09-04");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%010F", &time);
  EXPECT_STREQ_LEN(written, buffer, "0476-09-04");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%012F", &time);
  EXPECT_STREQ_LEN(written, buffer, "000476-09-04");

  time.tm_year = get_adjusted_year(-100);
  time.tm_mon = 9;
  time.tm_mday = 31;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01F", &time);
  EXPECT_STREQ_LEN(written, buffer, "-100-10-31");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%010F", &time);
  EXPECT_STREQ_LEN(written, buffer, "-100-10-31");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%012F", &time);
  EXPECT_STREQ_LEN(written, buffer, "-00100-10-31");

  // '+' flag tests
  time.tm_year = get_adjusted_year(2025);
  time.tm_mon = 1;
  time.tm_mday = 5;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+1F", &time);
  EXPECT_STREQ_LEN(written, buffer, "2025-02-05");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+10F", &time);
  EXPECT_STREQ_LEN(written, buffer, "2025-02-05");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+12F", &time);
  EXPECT_STREQ_LEN(written, buffer, "+02025-02-05");

  time.tm_year = get_adjusted_year(12345);
  time.tm_mon = 11;
  time.tm_mday = 25;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+1F", &time);
  EXPECT_STREQ_LEN(written, buffer, "+12345-12-25");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+10F", &time);
  EXPECT_STREQ_LEN(written, buffer, "+12345-12-25");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+12F", &time);
  EXPECT_STREQ_LEN(written, buffer, "+12345-12-25");

  time.tm_year = get_adjusted_year(476);
  time.tm_mon = 8;
  time.tm_mday = 4;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+1F", &time);
  EXPECT_STREQ_LEN(written, buffer, "476-09-04");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+10F", &time);
  EXPECT_STREQ_LEN(written, buffer, "0476-09-04");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+12F", &time);
  EXPECT_STREQ_LEN(written, buffer, "+00476-09-04");

  time.tm_year = get_adjusted_year(-100);
  time.tm_mon = 9;
  time.tm_mday = 31;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+1F", &time);
  EXPECT_STREQ_LEN(written, buffer, "-100-10-31");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+10F", &time);
  EXPECT_STREQ_LEN(written, buffer, "-100-10-31");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+12F", &time);
  EXPECT_STREQ_LEN(written, buffer, "-00100-10-31");
}

TEST(LlvmLibcStrftimeTest, TimeFormatAMPM) {
  // this tests %r, which reads: [tm_hour, tm_min, tm_sec]
  // This is equivalent to "%I:%M:%S %p"
  struct tm time;
  char buffer[100];
  size_t written = 0;

  // each of %I, %M, %S, and %p have their own tests, so this test won't cover
  // all values of those. Instead it will do basic tests and focus on the
  // specific padding behavior.

  time.tm_hour = 0;
  time.tm_min = 0;
  time.tm_sec = 0;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%r", &time);
  EXPECT_STREQ_LEN(written, buffer, "12:00:00 AM");

  time.tm_hour = 1;
  time.tm_min = 23;
  time.tm_sec = 45;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%r", &time);
  EXPECT_STREQ_LEN(written, buffer, "01:23:45 AM");

  time.tm_hour = 18;
  time.tm_min = 6;
  time.tm_sec = 2;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%r", &time);
  EXPECT_STREQ_LEN(written, buffer, "06:06:02 PM");

  // padding is technically undefined for this conversion, but we support it, so
  // we need to test it.
  // Padding is handled in the same way as POSIX describes for %F
  time.tm_hour = 10;
  time.tm_min = 9;
  time.tm_sec = 59;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01r", &time);
  EXPECT_STREQ_LEN(written, buffer, "10:09:59 AM");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%011r", &time);
  EXPECT_STREQ_LEN(written, buffer, "10:09:59 AM");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%013r", &time);
  EXPECT_STREQ_LEN(written, buffer, "0010:09:59 AM");

  time.tm_hour = 16;
  time.tm_min = 56;
  time.tm_sec = 9;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01r", &time);
  EXPECT_STREQ_LEN(written, buffer, "4:56:09 PM");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%011r", &time);
  EXPECT_STREQ_LEN(written, buffer, "04:56:09 PM");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%013r", &time);
  EXPECT_STREQ_LEN(written, buffer, "0004:56:09 PM");
}

TEST(LlvmLibcStrftimeTest, TimeFormatMinute) {
  // this tests %R, which reads: [tm_hour, tm_min]
  // This is equivalent to "%H:%M"
  struct tm time;
  char buffer[100];
  size_t written = 0;

  // each of %H and %M have their own tests, so this test won't cover
  // all values of those. Instead it will do basic tests and focus on the
  // specific padding behavior.

  time.tm_hour = 0;
  time.tm_min = 0;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%R", &time);
  EXPECT_STREQ_LEN(written, buffer, "00:00");

  time.tm_hour = 1;
  time.tm_min = 23;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%R", &time);
  EXPECT_STREQ_LEN(written, buffer, "01:23");

  time.tm_hour = 18;
  time.tm_min = 6;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%R", &time);
  EXPECT_STREQ_LEN(written, buffer, "18:06");

  // padding is technically undefined for this conversion, but we support it, so
  // we need to test it.
  // Padding is handled in the same way as POSIX describes for %F
  time.tm_hour = 10;
  time.tm_min = 9;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01R", &time);
  EXPECT_STREQ_LEN(written, buffer, "10:09");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%05R", &time);
  EXPECT_STREQ_LEN(written, buffer, "10:09");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%07R", &time);
  EXPECT_STREQ_LEN(written, buffer, "0010:09");

  time.tm_hour = 4;
  time.tm_min = 56;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01R", &time);
  EXPECT_STREQ_LEN(written, buffer, "4:56");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%05R", &time);
  EXPECT_STREQ_LEN(written, buffer, "04:56");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%07R", &time);
  EXPECT_STREQ_LEN(written, buffer, "0004:56");
}

TEST(LlvmLibcStrftimeTest, TimeFormatSecond) {
  // this tests %T, which reads: [tm_hour, tm_min, tm_sec]
  // This is equivalent to "%H:%M:%S"
  struct tm time;
  char buffer[100];
  size_t written = 0;

  // each of %H, %M, and %S have their own tests, so this test won't cover
  // all values of those. Instead it will do basic tests and focus on the
  // specific padding behavior.

  time.tm_hour = 0;
  time.tm_min = 0;
  time.tm_sec = 0;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%T", &time);
  EXPECT_STREQ_LEN(written, buffer, "00:00:00");

  time.tm_hour = 1;
  time.tm_min = 23;
  time.tm_sec = 45;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%T", &time);
  EXPECT_STREQ_LEN(written, buffer, "01:23:45");

  time.tm_hour = 18;
  time.tm_min = 6;
  time.tm_sec = 2;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%T", &time);
  EXPECT_STREQ_LEN(written, buffer, "18:06:02");

  // %X is equivalent to %T in default locale
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%X", &time);
  EXPECT_STREQ_LEN(written, buffer, "18:06:02");

  // padding is technically undefined for this conversion, but we support it, so
  // we need to test it.
  // Padding is handled in the same way as POSIX describes for %F
  time.tm_hour = 10;
  time.tm_min = 9;
  time.tm_sec = 59;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01T", &time);
  EXPECT_STREQ_LEN(written, buffer, "10:09:59");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%08T", &time);
  EXPECT_STREQ_LEN(written, buffer, "10:09:59");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%010T", &time);
  EXPECT_STREQ_LEN(written, buffer, "0010:09:59");

  time.tm_hour = 4;
  time.tm_min = 56;
  time.tm_sec = 9;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01T", &time);
  EXPECT_STREQ_LEN(written, buffer, "4:56:09");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%08T", &time);
  EXPECT_STREQ_LEN(written, buffer, "04:56:09");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%010T", &time);
  EXPECT_STREQ_LEN(written, buffer, "0004:56:09");
}

TEST(LlvmLibcStrftimeTest, TimeFormatFullDateTime) {
  // this tests %c, which reads:
  //  [tm_wday, tm_mon, tm_mday, tm_hour, tm_min, tm_sec, tm_year]
  // This is equivalent to "%a %b %e %T %Y"
  struct tm time;
  char buffer[100];
  size_t written = 0;

  // each of the individual conversions have their own tests, so this test won't
  // cover all values of those. Instead it will do basic tests and focus on the
  // specific padding behavior.

  time.tm_wday = 0;
  time.tm_mon = 0;
  time.tm_mday = 1;
  time.tm_hour = 0;
  time.tm_min = 0;
  time.tm_sec = 0;
  time.tm_year = get_adjusted_year(1900);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%c", &time);
  EXPECT_STREQ_LEN(written, buffer, "Sun Jan  1 00:00:00 1900");

  time.tm_wday = 3;
  time.tm_mon = 5;
  time.tm_mday = 15;
  time.tm_hour = 14;
  time.tm_min = 13;
  time.tm_sec = 12;
  time.tm_year = get_adjusted_year(2011);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%c", &time);
  EXPECT_STREQ_LEN(written, buffer, "Wed Jun 15 14:13:12 2011");

  // now, as of the writing of this test
  time.tm_wday = 4;
  time.tm_mon = 1;
  time.tm_mday = 6;
  time.tm_hour = 12;
  time.tm_min = 57;
  time.tm_sec = 50;
  time.tm_year = get_adjusted_year(2025);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%c", &time);
  EXPECT_STREQ_LEN(written, buffer, "Thu Feb  6 12:57:50 2025");

  time.tm_wday = 5;
  time.tm_mon = 8;
  time.tm_mday = 4;
  time.tm_hour = 16;
  time.tm_min = 57;
  time.tm_sec = 18;
  time.tm_year = get_adjusted_year(476);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%c", &time);
  EXPECT_STREQ_LEN(written, buffer, "Fri Sep  4 16:57:18 0476");

  // padding is technically undefined for this conversion, but we support it, so
  // we need to test it.
  // Padding is handled in the same way as POSIX describes for %F.
  // This includes assuming the trailing conversions are of a fixed width, which
  // isn't true for years. For simplicity, we format years (%Y) to be padded to
  // 4 digits when possible, which means padding will work as expected for years
  // -999 to 9999. If the current year is large enough to trigger this bug,
  // congrats on making it another ~8000 years!
  time.tm_wday = 5;
  time.tm_mon = 8;
  time.tm_mday = 4;
  time.tm_hour = 16;
  time.tm_min = 57;
  time.tm_sec = 18;
  time.tm_year = get_adjusted_year(476);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%1c", &time);
  EXPECT_STREQ_LEN(written, buffer, "Fri Sep  4 16:57:18 0476");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%24c", &time);
  EXPECT_STREQ_LEN(written, buffer, "Fri Sep  4 16:57:18 0476");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%26c", &time);
  EXPECT_STREQ_LEN(written, buffer, "  Fri Sep  4 16:57:18 0476");

  // '0' flag has no effect on the string part of the conversion, only the
  // numbers, and the only one of those that defaults to spaces is day of month.
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%026c", &time);
  EXPECT_STREQ_LEN(written, buffer, "  Fri Sep 04 16:57:18 0476");

  time.tm_wday = 3;
  time.tm_mon = 5;
  time.tm_mday = 15;
  time.tm_hour = 14;
  time.tm_min = 13;
  time.tm_sec = 12;
  time.tm_year = get_adjusted_year(2011);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%1c", &time);
  EXPECT_STREQ_LEN(written, buffer, "Wed Jun 15 14:13:12 2011");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%24c", &time);
  EXPECT_STREQ_LEN(written, buffer, "Wed Jun 15 14:13:12 2011");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%26c", &time);
  EXPECT_STREQ_LEN(written, buffer, "  Wed Jun 15 14:13:12 2011");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%026c", &time);
  EXPECT_STREQ_LEN(written, buffer, "  Wed Jun 15 14:13:12 2011");
}

// TODO: implement %z and %Z when timezones are implemented.
//  TEST(LlvmLibcStrftimeTest, TimezoneOffset) {
//    // this tests %z, which reads: [tm_isdst, tm_zone]
//    struct tm time;
//    char buffer[100];
//    size_t written = 0;
//    SimplePaddedNum spn;
//  }

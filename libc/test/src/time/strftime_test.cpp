//===-- Unittests for strftime --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/types/struct_tm.h"
#include "src/__support/integer_to_string.h"
#include "src/time/strftime.h"
#include "test/UnitTest/Test.h"

// Copied from sprintf_test.cpp.
// TODO: put this somewhere more reusable, it's handy.
// Subtract 1 from sizeof(expected_str) to account for the null byte.
#define EXPECT_STREQ_LEN(actual_written, actual_str, expected_str)             \
  EXPECT_EQ(actual_written, sizeof(expected_str) - 1);                         \
  EXPECT_STREQ(actual_str, expected_str);

TEST(LlvmLibcStrftimeTest, FullYearTests) {
  // this tests %Y, which reads: [tm_year]
  struct tm time;
  char buffer[100];
  size_t written = 0;

  // basic tests
  time.tm_year = 2022 - 1900; // tm_year counts years since 1900, so 122 -> 2022
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "2022");

  time.tm_year = 11900 - 1900;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "11900");

  time.tm_year = 1900 - 1900;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "1900");

  time.tm_year = 900 - 1900;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "900");

  time.tm_year = 0 - 1900;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "0");

  time.tm_year = -1 - 1900;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "-1");

  time.tm_year = -9001 - 1900;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "-9001");

  time.tm_year = -10001 - 1900;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "-10001");

  // width tests (with the 0 flag, since the default padding is undefined).
  time.tm_year = 2023 - 1900;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "2023");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%04Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "2023");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%05Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "02023");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%010Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "0000002023");

  time.tm_year = 900 - 1900;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "900");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%04Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "0900");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%05Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "00900");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%010Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "0000000900");

  time.tm_year = 12345 - 1900;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "12345");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%04Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "12345");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%05Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "12345");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%010Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "0000012345");

  time.tm_year = -123 - 1900;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "-123");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%04Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "-123");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%05Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "-0123");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%010Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "-000000123");

  // '+' flag tests
  time.tm_year = 2023 - 1900;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+1Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "2023");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+4Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "2023");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+5Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "+2023");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+10Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "+000002023");

  time.tm_year = 900 - 1900;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+1Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "900");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+4Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "0900");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+5Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "+0900");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+10Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "+000000900");

  time.tm_year = 12345 - 1900;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+1Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "+12345");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+4Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "+12345");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+5Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "+12345");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+10Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "+000012345");

  time.tm_year = -123 - 1900;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+1Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "-123");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+4Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "-123");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+5Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "-0123");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+10Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "-000000123");

  // Posix specified tests:
  time.tm_year = 1970 - 1900;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "1970");

  time.tm_year = 1970 - 1900;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+4Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "1970");

  time.tm_year = 27 - 1900;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "27");

  time.tm_year = 270 - 1900;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "270");

  time.tm_year = 270 - 1900;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+4Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "0270");

  time.tm_year = 12345 - 1900;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "12345");

  time.tm_year = 12345 - 1900;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+4Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "+12345");

  time.tm_year = 12345 - 1900;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%05Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "12345");

  time.tm_year = 270 - 1900;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+5Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "+0270");

  time.tm_year = 12345 - 1900;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+5Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "+12345");

  time.tm_year = 12345 - 1900;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%06Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "012345");

  time.tm_year = 12345 - 1900;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+6Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "+12345");

  time.tm_year = 123456 - 1900;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%08Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "00123456");

  time.tm_year = 123456 - 1900;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+8Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "+0123456");
}

TEST(LlvmLibcStrftimeTest, CenturyTests) {
  // this tests %C, which reads: [tm_year]
  struct tm time;
  char buffer[100];
  size_t written = 0;

  // basic tests
  time.tm_year = 2022 - 1900; // tm_year counts years since 1900, so 122 -> 2022
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%C", &time);
  EXPECT_STREQ_LEN(written, buffer, "20");

  time.tm_year = 11900 - 1900;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%C", &time);
  EXPECT_STREQ_LEN(written, buffer, "119");

  time.tm_year = 1900 - 1900;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%C", &time);
  EXPECT_STREQ_LEN(written, buffer, "19");

  time.tm_year = 900 - 1900;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%C", &time);
  EXPECT_STREQ_LEN(written, buffer, "09");

  time.tm_year = 0 - 1900;
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
  time.tm_year = -1 - 1900;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%C", &time);
  EXPECT_STREQ_LEN(written, buffer, "00");

  time.tm_year = -101 - 1900;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%C", &time);
  EXPECT_STREQ_LEN(written, buffer, "-1");

  time.tm_year = -9001 - 1900;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%C", &time);
  EXPECT_STREQ_LEN(written, buffer, "-90");

  time.tm_year = -10001 - 1900;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%C", &time);
  EXPECT_STREQ_LEN(written, buffer, "-100");

  // width tests (with the 0 flag, since the default padding is undefined).
  time.tm_year = 2023 - 1900;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01C", &time);
  EXPECT_STREQ_LEN(written, buffer, "20");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%02C", &time);
  EXPECT_STREQ_LEN(written, buffer, "20");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%05C", &time);
  EXPECT_STREQ_LEN(written, buffer, "00020");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%010C", &time);
  EXPECT_STREQ_LEN(written, buffer, "0000000020");

  time.tm_year = 900 - 1900;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01C", &time);
  EXPECT_STREQ_LEN(written, buffer, "9");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%02C", &time);
  EXPECT_STREQ_LEN(written, buffer, "09");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%05C", &time);
  EXPECT_STREQ_LEN(written, buffer, "00009");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%010C", &time);
  EXPECT_STREQ_LEN(written, buffer, "0000000009");

  time.tm_year = 12345 - 1900;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01C", &time);
  EXPECT_STREQ_LEN(written, buffer, "123");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%02C", &time);
  EXPECT_STREQ_LEN(written, buffer, "123");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%05C", &time);
  EXPECT_STREQ_LEN(written, buffer, "00123");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%010C", &time);
  EXPECT_STREQ_LEN(written, buffer, "0000000123");

  time.tm_year = -123 - 1900;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%01C", &time);
  EXPECT_STREQ_LEN(written, buffer, "-1");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%02C", &time);
  EXPECT_STREQ_LEN(written, buffer, "-1");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%05C", &time);
  EXPECT_STREQ_LEN(written, buffer, "-0001");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%010C", &time);
  EXPECT_STREQ_LEN(written, buffer, "-000000001");

  // '+' flag tests
  time.tm_year = 2023 - 1900;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+1C", &time);
  EXPECT_STREQ_LEN(written, buffer, "20");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+2C", &time);
  EXPECT_STREQ_LEN(written, buffer, "20");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+5C", &time);
  EXPECT_STREQ_LEN(written, buffer, "+0020");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+10C", &time);
  EXPECT_STREQ_LEN(written, buffer, "+000000020");

  time.tm_year = 900 - 1900;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+1C", &time);
  EXPECT_STREQ_LEN(written, buffer, "9");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+2C", &time);
  EXPECT_STREQ_LEN(written, buffer, "09");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+5C", &time);
  EXPECT_STREQ_LEN(written, buffer, "+0009");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+10C", &time);
  EXPECT_STREQ_LEN(written, buffer, "+000000009");

  time.tm_year = 12345 - 1900;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+1C", &time);
  EXPECT_STREQ_LEN(written, buffer, "+123");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+2C", &time);
  EXPECT_STREQ_LEN(written, buffer, "+123");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+5C", &time);
  EXPECT_STREQ_LEN(written, buffer, "+0123");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+10C", &time);
  EXPECT_STREQ_LEN(written, buffer, "+000000123");

  time.tm_year = -123 - 1900;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+1C", &time);
  EXPECT_STREQ_LEN(written, buffer, "-1");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+2C", &time);
  EXPECT_STREQ_LEN(written, buffer, "-1");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+5C", &time);
  EXPECT_STREQ_LEN(written, buffer, "-0001");

  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+10C", &time);
  EXPECT_STREQ_LEN(written, buffer, "-000000001");

  // Posix specified tests:
  time.tm_year = 17 - 1900;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%C", &time);
  EXPECT_STREQ_LEN(written, buffer, "00");

  time.tm_year = 270 - 1900;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%C", &time);
  EXPECT_STREQ_LEN(written, buffer, "02");

  time.tm_year = 270 - 1900;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+3C", &time);
  EXPECT_STREQ_LEN(written, buffer, "+02");

  time.tm_year = 12345 - 1900;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+3C", &time);
  EXPECT_STREQ_LEN(written, buffer, "+123");

  time.tm_year = 12345 - 1900;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%04C", &time);
  EXPECT_STREQ_LEN(written, buffer, "0123");

  time.tm_year = 12345 - 1900;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+4C", &time);
  EXPECT_STREQ_LEN(written, buffer, "+123");

  time.tm_year = 123456 - 1900;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%06C", &time);
  EXPECT_STREQ_LEN(written, buffer, "001234");

  time.tm_year = 123456 - 1900;
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%+6C", &time);
  EXPECT_STREQ_LEN(written, buffer, "+01234");
}

// TODO: Move this somewhere it can be reused. It seems like a useful tool to
// have.
// A helper class to generate simple padded numbers. It places the result in its
// internal buffer, which is cleared on every call.
class SimplePaddedNum {
  static constexpr size_t BUFF_LEN = 8;
  char buff[BUFF_LEN];
  size_t cur_len; // length of string currently in buff

  void clear_buff() {
    // TODO: builtin_memset?
    for (size_t i = 0; i < BUFF_LEN; ++i)
      buff[i] = '\0';
  }

public:
  SimplePaddedNum() = default;

  // PRECONDITIONS: num < 999, min_width < 3
  // Returns: Pointer to the start of the padded number as a string, stored in
  // the internal buffer.
  char *get_padded_num(int num, size_t min_width) {
    clear_buff();

    LIBC_NAMESPACE::IntegerToString<int> raw(num);
    auto str = raw.view();
    int leading_zeroes = min_width - raw.size();

    size_t i = 0;
    for (; static_cast<int>(i) < leading_zeroes; ++i)
      buff[i] = '0';
    for (size_t str_cur = 0; str_cur < str.size(); ++i, ++str_cur)
      buff[i] = str[str_cur];
    cur_len = i;
    return buff;
  }

  size_t get_str_len() { return cur_len; }
};

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
    char *result = spn.get_padded_num(i, 1);

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
  for (size_t i = 1; i < 32; ++i) {
    time.tm_year = i;
    written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%g", &time);
    char *result = spn.get_padded_num(i, 2);

    ASSERT_STREQ(buffer, result);
    ASSERT_EQ(written, spn.get_str_len());
  }

  // Test the harder to round cases

  // not a leap year. Not relevant for the start-of-year tests, but it does
  // matter for the end-of-year tests.
  time.tm_year = 99;

  // check the first days of the year
  for (size_t wday = 0; wday < 7; ++wday) {
    for (size_t yday = 1; yday < 5; ++yday) {
      time.tm_wday = wday;
      time.tm_yday = yday;

      written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%g", &time);
      if (yday == 4) {
        // Since the first ISO week must contain the 4th yday, this must always
        // return the current year.
        EXPECT_STREQ_LEN(written, buffer, "99");
      } else if (yday == 3) {
        // Only sunday the 3rd can be in a week that doesn't contain the 4th.
        if (wday == 0) {
          EXPECT_STREQ_LEN(written, buffer, "98");
        } else {
          EXPECT_STREQ_LEN(written, buffer, "99");
        }
      } else if (yday == 2) {
        // Sunday or Monday the 2nd can be in a week that doesn't contain the
        // 4th.
        if (wday == 0 || wday == 6) {
          EXPECT_STREQ_LEN(written, buffer, "98");
        } else {
          EXPECT_STREQ_LEN(written, buffer, "99");
        }
      } else {
        // Sunday, Monday, or tuesday the 1st can be in a week that doesn't
        // contain the 4th.
        if (wday == 0 || wday == 6 || wday == 5) {
          EXPECT_STREQ_LEN(written, buffer, "98");
        } else {
          EXPECT_STREQ_LEN(written, buffer, "99");
        }
      }
    }
  }


  // check the last days of the year
  for (size_t wday = 0; wday < 7; ++wday) {
    for (size_t yday = 363; yday < 5; ++yday) {
      time.tm_wday = wday;
      time.tm_yday = yday;

      written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%g", &time);
      if (yday == 4) {
        // Since the first ISO week must contain the 4th yday, this must always
        // return the current year.
        EXPECT_STREQ_LEN(written, buffer, "99");
      } else if (yday == 3) {
        // Only sunday the 3rd can be in a week that doesn't contain the 4th.
        if (wday == 0) {
          EXPECT_STREQ_LEN(written, buffer, "98");
        } else {
          EXPECT_STREQ_LEN(written, buffer, "99");
        }
      } else if (yday == 2) {
        // Sunday or Monday the 2nd can be in a week that doesn't contain the
        // 4th.
        if (wday == 0 || wday == 6) {
          EXPECT_STREQ_LEN(written, buffer, "98");
        } else {
          EXPECT_STREQ_LEN(written, buffer, "99");
        }
      } else {
        // Sunday, Monday, or tuesday the 1st can be in a week that doesn't
        // contain the 4th.
        if (wday == 0 || wday == 6 || wday == 5) {
          EXPECT_STREQ_LEN(written, buffer, "98");
        } else {
          EXPECT_STREQ_LEN(written, buffer, "99");
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

// TODO: tests for each other conversion.

TEST(LlvmLibcStrftimeTest, CompositeTests) {
  struct tm time;
  time.tm_year = 122; // Year since 1900, so 2022
  time.tm_mon = 9;    // October (0-indexed)
  time.tm_mday = 15;  // 15th day

  char buffer[100];
  LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%Y-%m-%d", &time);
  EXPECT_STREQ(buffer, "2022-10-15");
}

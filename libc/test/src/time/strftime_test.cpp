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

TEST(LlvmLibcStrftimeTest, FullYearTests) {
  // this tests %Y, which reads: [tm_year]
  struct tm time;
  char buffer[100];
  size_t written = 0;

  // basic tests
  time.tm_year = get_adjusted_year(2022);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "2022");

  time.tm_year = get_adjusted_year(11900);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "11900");

  time.tm_year = get_adjusted_year(1900);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "1900");

  time.tm_year = get_adjusted_year(900);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "900");

  time.tm_year = get_adjusted_year(0);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "0");

  time.tm_year = get_adjusted_year(-1);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "-1");

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
  EXPECT_STREQ_LEN(written, buffer, "27");

  time.tm_year = get_adjusted_year(270);
  written = LIBC_NAMESPACE::strftime(buffer, sizeof(buffer), "%Y", &time);
  EXPECT_STREQ_LEN(written, buffer, "270");

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
  char *get_padded_num(int num, size_t min_width) {
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
    // apparently %G doesn't pad by default.
    char *result = spn.get_padded_num(i, 0);

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
  time.tm_year = 99;

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

// TEST(LlvmLibcStrftimeTest, SecondsSinceEpoch) {
//   // this tests %s, which reads: [tm_year, tm_mon, tm_mday, tm_hour, tm_min,
//   // tm_sec, tm_isdst]
//   struct tm time;
//   char buffer[100];
//   size_t written = 0;
//   SimplePaddedNum spn;
//   // TODO: Test this once the conversion is done.
// }

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

      if (result[1] != buffer[1]) {
        written++;
      }
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

//===-- Unittests for strftime --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/types/struct_tm.h"
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

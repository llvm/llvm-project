//===-- Unittests for sprintf ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/sprintf.h"

#include "utils/UnitTest/Test.h"

TEST(LlvmLibcSPrintfTest, SimpleNoConv) {
  char buff[64];
  int written;

  written = __llvm_libc::sprintf(buff, "A simple string with no conversions.");
  EXPECT_EQ(written, 36);
  ASSERT_STREQ(buff, "A simple string with no conversions.");
}

TEST(LlvmLibcSPrintfTest, PercentConv) {
  char buff[64];
  int written;

  written = __llvm_libc::sprintf(buff, "%%");
  EXPECT_EQ(written, 1);
  ASSERT_STREQ(buff, "%");

  written = __llvm_libc::sprintf(buff, "abc %% def");
  EXPECT_EQ(written, 9);
  ASSERT_STREQ(buff, "abc % def");

  written = __llvm_libc::sprintf(buff, "%%%%%%");
  EXPECT_EQ(written, 3);
  ASSERT_STREQ(buff, "%%%");
}

TEST(LlvmLibcSPrintfTest, CharConv) {
  char buff[64];
  int written;

  written = __llvm_libc::sprintf(buff, "%c", 'a');
  EXPECT_EQ(written, 1);
  ASSERT_STREQ(buff, "a");

  written = __llvm_libc::sprintf(buff, "%3c %-3c", '1', '2');
  EXPECT_EQ(written, 7);
  ASSERT_STREQ(buff, "  1 2  ");

  written = __llvm_libc::sprintf(buff, "%*c", 2, '3');
  EXPECT_EQ(written, 2);
  ASSERT_STREQ(buff, " 3");
}

TEST(LlvmLibcSPrintfTest, StringConv) {
  char buff[64];
  int written;

  written = __llvm_libc::sprintf(buff, "%s", "abcDEF123");
  EXPECT_EQ(written, 9);
  ASSERT_STREQ(buff, "abcDEF123");

  written = __llvm_libc::sprintf(buff, "%10s %-10s", "centered", "title");
  EXPECT_EQ(written, 21);
  ASSERT_STREQ(buff, "  centered title     ");

  written = __llvm_libc::sprintf(buff, "%-5.4s%-4.4s", "words can describe",
                                 "soups most delicious");
  EXPECT_EQ(written, 9);
  ASSERT_STREQ(buff, "word soup");

  written = __llvm_libc::sprintf(buff, "%*s %.*s %*.*s", 10, "beginning", 2,
                                 "isn't", 12, 10, "important. Ever.");
  EXPECT_EQ(written, 26);
  ASSERT_STREQ(buff, " beginning is   important.");
}

TEST(LlvmLibcSPrintfTest, IntConv) {
  char buff[64];
  int written;

  // Basic Tests.

  written = __llvm_libc::sprintf(buff, "%d", 123);
  EXPECT_EQ(written, 3);
  ASSERT_STREQ(buff, "123");

  written = __llvm_libc::sprintf(buff, "%i", -456);
  EXPECT_EQ(written, 4);
  ASSERT_STREQ(buff, "-456");

  // Length Modifier Tests.

  written = __llvm_libc::sprintf(buff, "%hhu", 257); // 0x10001
  EXPECT_EQ(written, 1);
  ASSERT_STREQ(buff, "1");

  written = __llvm_libc::sprintf(buff, "%llu", 18446744073709551615ull);
  EXPECT_EQ(written, 20);
  ASSERT_STREQ(buff, "18446744073709551615"); // ull max

  written = __llvm_libc::sprintf(buff, "%tu", ~ptrdiff_t(0));
  if (sizeof(ptrdiff_t) == 8) {
    EXPECT_EQ(written, 20);
    ASSERT_STREQ(buff, "18446744073709551615");
  } else if (sizeof(ptrdiff_t) == 4) {
    EXPECT_EQ(written, 10);
    ASSERT_STREQ(buff, "4294967296");
  }

  written = __llvm_libc::sprintf(buff, "%lld", -9223372036854775807ll - 1ll);
  EXPECT_EQ(written, 20);
  ASSERT_STREQ(buff, "-9223372036854775808"); // ll min

  // Min Width Tests.

  written = __llvm_libc::sprintf(buff, "%4d", 789);
  EXPECT_EQ(written, 4);
  ASSERT_STREQ(buff, " 789");

  written = __llvm_libc::sprintf(buff, "%2d", 987);
  EXPECT_EQ(written, 3);
  ASSERT_STREQ(buff, "987");

  // Precision Tests.

  written = __llvm_libc::sprintf(buff, "%d", 0);
  EXPECT_EQ(written, 1);
  ASSERT_STREQ(buff, "0");

  written = __llvm_libc::sprintf(buff, "%.0d", 0);
  EXPECT_EQ(written, 0);
  ASSERT_STREQ(buff, "");

  written = __llvm_libc::sprintf(buff, "%.5d", 654);
  EXPECT_EQ(written, 5);
  ASSERT_STREQ(buff, "00654");

  written = __llvm_libc::sprintf(buff, "%.5d", -321);
  EXPECT_EQ(written, 6);
  ASSERT_STREQ(buff, "-00321");

  written = __llvm_libc::sprintf(buff, "%.2d", 135);
  EXPECT_EQ(written, 3);
  ASSERT_STREQ(buff, "135");

  // Flag Tests.

  written = __llvm_libc::sprintf(buff, "%.5d", -321);
  EXPECT_EQ(written, 6);
  ASSERT_STREQ(buff, "-00321");

  written = __llvm_libc::sprintf(buff, "%-5d", 246);
  EXPECT_EQ(written, 5);
  ASSERT_STREQ(buff, "246  ");

  written = __llvm_libc::sprintf(buff, "%-5d", -147);
  EXPECT_EQ(written, 5);
  ASSERT_STREQ(buff, "-147 ");

  written = __llvm_libc::sprintf(buff, "%+d", 258);
  EXPECT_EQ(written, 4);
  ASSERT_STREQ(buff, "+258");

  written = __llvm_libc::sprintf(buff, "% d", 369);
  EXPECT_EQ(written, 4);
  ASSERT_STREQ(buff, " 369");

  written = __llvm_libc::sprintf(buff, "%05d", 470);
  EXPECT_EQ(written, 5);
  ASSERT_STREQ(buff, "00470");

  written = __llvm_libc::sprintf(buff, "%05d", -581);
  EXPECT_EQ(written, 5);
  ASSERT_STREQ(buff, "-0581");

  // Combined Tests.

  written = __llvm_libc::sprintf(buff, "%+ u", 692);
  EXPECT_EQ(written, 3);
  ASSERT_STREQ(buff, "692");

  written = __llvm_libc::sprintf(buff, "%+ -05d", 703);
  EXPECT_EQ(written, 5);
  ASSERT_STREQ(buff, "+703 ");

  written = __llvm_libc::sprintf(buff, "%7.5d", 814);
  EXPECT_EQ(written, 7);
  ASSERT_STREQ(buff, "  00814");

  written = __llvm_libc::sprintf(buff, "%7.5d", -925);
  EXPECT_EQ(written, 7);
  ASSERT_STREQ(buff, " -00925");

  written = __llvm_libc::sprintf(buff, "%7.5d", 159);
  EXPECT_EQ(written, 7);
  ASSERT_STREQ(buff, "  00159");

  written = __llvm_libc::sprintf(buff, "% -7.5d", 260);
  EXPECT_EQ(written, 7);
  ASSERT_STREQ(buff, " 00260 ");

  written = __llvm_libc::sprintf(buff, "%5.4d", 10000);
  EXPECT_EQ(written, 5);
  ASSERT_STREQ(buff, "10000");

  // Multiple Conversion Tests.

  written = __llvm_libc::sprintf(buff, "%10d %-10d", 456, -789);
  EXPECT_EQ(written, 21);
  ASSERT_STREQ(buff, "       456 -789      ");

  written = __llvm_libc::sprintf(buff, "%-5.4d%+.4u", 75, 25);
  EXPECT_EQ(written, 9);
  ASSERT_STREQ(buff, "0075 0025");

  written = __llvm_libc::sprintf(buff, "% 05hhi %+-0.5llu %-+ 06.3zd",
                                 256 + 127, 68719476736ll, size_t(2));
  EXPECT_EQ(written, 24);
  ASSERT_STREQ(buff, " 0127 68719476736 +002  ");
}

#ifndef LLVM_LIBC_PRINTF_DISABLE_INDEX_MODE
TEST(LlvmLibcSPrintfTest, IndexModeParsing) {
  char buff[64];
  int written;

  written = __llvm_libc::sprintf(buff, "%1$s", "abcDEF123");
  EXPECT_EQ(written, 9);
  ASSERT_STREQ(buff, "abcDEF123");

  written = __llvm_libc::sprintf(buff, "%1$s %%", "abcDEF123");
  EXPECT_EQ(written, 11);
  ASSERT_STREQ(buff, "abcDEF123 %");

  written =
      __llvm_libc::sprintf(buff, "%3$s %1$s %2$s", "is", "hard", "ordering");
  EXPECT_EQ(written, 16);
  ASSERT_STREQ(buff, "ordering is hard");

  written = __llvm_libc::sprintf(
      buff, "%10$s %9$s %8$c %7$s %6$s, %6$s %5$s %4$-*1$s %3$.*11$s %2$s. %%",
      6, "pain", "alphabetical", "such", "is", "this", "do", 'u', "would",
      "why", 1);
  EXPECT_EQ(written, 45);
  ASSERT_STREQ(buff, "why would u do this, this is such   a pain. %");
}
#endif // LLVM_LIBC_PRINTF_DISABLE_INDEX_MODE

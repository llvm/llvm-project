//===-- Unittests for sprintf ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/macros/config.h"
#include "src/stdio/sprintf.h"

#include "src/__support/FPUtil/FPBits.h"
#include "src/errno/libc_errno.h"
#include "test/UnitTest/RoundingModeUtils.h"
#include "test/UnitTest/Test.h"
#include <inttypes.h>

// TODO: Add a comment here explaining the printf format string.

// #include <stdio.h>
// namespace LIBC_NAMESPACE_DECL {
// using ::sprintf;
// }

class LlvmLibcSPrintfTest : public LIBC_NAMESPACE::testing::Test {
protected:
  char buff[1000];
  int written;
};

using LIBC_NAMESPACE::fputil::testing::ForceRoundingMode;
using LIBC_NAMESPACE::fputil::testing::RoundingMode;

// Subtract 1 from sizeof(expected_str) to account for the null byte.
#define ASSERT_STREQ_LEN(actual_written, actual_str, expected_str)             \
  EXPECT_EQ(actual_written, static_cast<int>(sizeof(expected_str) - 1));       \
  EXPECT_STREQ(actual_str, expected_str);

#define macro_test(FMT, X, expected)                                           \
  do {                                                                         \
    for (char &c : buff) {                                                     \
      c = 0;                                                                   \
    }                                                                          \
    written = LIBC_NAMESPACE::sprintf(buff, "%" FMT, X);                       \
    ASSERT_STREQ_LEN(written, buff, expected);                                 \
  } while (0)

TEST(LlvmLibcSPrintfTest, Macros) {
  char buff[128];
  int written;
  macro_test(PRIu8, 1, "1");
  macro_test(PRIX16, 0xAA, "AA");
  macro_test(PRId32, -123, "-123");
  macro_test(PRIX32, 0xFFFFFF85, "FFFFFF85");
  macro_test(PRIo8, 0xFF, "377");
  macro_test(PRIo64, 0123456712345671234567ll, "123456712345671234567");
}

TEST(LlvmLibcSPrintfTest, SimpleNoConv) {
  char buff[64];
  int written;

  written =
      LIBC_NAMESPACE::sprintf(buff, "A simple string with no conversions.");
  EXPECT_EQ(written, 36);
  ASSERT_STREQ(buff, "A simple string with no conversions.");
}

TEST(LlvmLibcSPrintfTest, PercentConv) {
  char buff[64];
  int written;

  written = LIBC_NAMESPACE::sprintf(buff, "%%");
  EXPECT_EQ(written, 1);
  ASSERT_STREQ(buff, "%");

  written = LIBC_NAMESPACE::sprintf(buff, "abc %% def");
  EXPECT_EQ(written, 9);
  ASSERT_STREQ(buff, "abc % def");

  written = LIBC_NAMESPACE::sprintf(buff, "%%%%%%");
  EXPECT_EQ(written, 3);
  ASSERT_STREQ(buff, "%%%");
}

TEST(LlvmLibcSPrintfTest, CharConv) {
  char buff[64];
  int written;

  written = LIBC_NAMESPACE::sprintf(buff, "%c", 'a');
  EXPECT_EQ(written, 1);
  ASSERT_STREQ(buff, "a");

  written = LIBC_NAMESPACE::sprintf(buff, "%3c %-3c", '1', '2');
  EXPECT_EQ(written, 7);
  ASSERT_STREQ(buff, "  1 2  ");

  written = LIBC_NAMESPACE::sprintf(buff, "%*c", 2, '3');
  EXPECT_EQ(written, 2);
  ASSERT_STREQ(buff, " 3");
}

TEST(LlvmLibcSPrintfTest, StringConv) {
  char buff[64];
  int written;

  written = LIBC_NAMESPACE::sprintf(buff, "%s", "abcDEF123");
  EXPECT_EQ(written, 9);
  ASSERT_STREQ(buff, "abcDEF123");

  written = LIBC_NAMESPACE::sprintf(buff, "%10s %-10s", "centered", "title");
  EXPECT_EQ(written, 21);
  ASSERT_STREQ(buff, "  centered title     ");

  written = LIBC_NAMESPACE::sprintf(buff, "%-5.4s%-4.4s", "words can describe",
                                    "soups most delicious");
  EXPECT_EQ(written, 9);
  ASSERT_STREQ(buff, "word soup");

  written = LIBC_NAMESPACE::sprintf(buff, "%*s %.*s %*.*s", 10, "beginning", 2,
                                    "isn't", 12, 10, "important. Ever.");
  EXPECT_EQ(written, 26);
  ASSERT_STREQ(buff, " beginning is   important.");

#ifndef LIBC_COPT_PRINTF_NO_NULLPTR_CHECKS
  written = LIBC_NAMESPACE::sprintf(buff, "%s", nullptr);
  EXPECT_EQ(written, 6);
  ASSERT_STREQ(buff, "(null)");
#endif // LIBC_COPT_PRINTF_NO_NULLPTR_CHECKS
}

TEST(LlvmLibcSPrintfTest, IntConv) {
  char buff[64];
  int written;

  // Basic Tests.

  written = LIBC_NAMESPACE::sprintf(buff, "%d", 123);
  EXPECT_EQ(written, 3);
  ASSERT_STREQ(buff, "123");

  written = LIBC_NAMESPACE::sprintf(buff, "%i", -456);
  EXPECT_EQ(written, 4);
  ASSERT_STREQ(buff, "-456");

  // Length Modifier Tests.

  written = LIBC_NAMESPACE::sprintf(buff, "%hhu", 257); // 0x101
  EXPECT_EQ(written, 1);
  ASSERT_STREQ(buff, "1");

  written = LIBC_NAMESPACE::sprintf(buff, "%llu", 18446744073709551615ull);
  EXPECT_EQ(written, 20);
  ASSERT_STREQ(buff, "18446744073709551615"); // ull max

  written = LIBC_NAMESPACE::sprintf(buff, "%u", ~0);
  if (sizeof(int) == 4) {
    EXPECT_EQ(written, 10);
    ASSERT_STREQ(buff, "4294967295");
  }

  written = LIBC_NAMESPACE::sprintf(buff, "%tu", ~ptrdiff_t(0));
  if (sizeof(ptrdiff_t) == 8) {
    EXPECT_EQ(written, 20);
    ASSERT_STREQ(buff, "18446744073709551615");
  } else if (sizeof(ptrdiff_t) == 4) {
    EXPECT_EQ(written, 10);
    ASSERT_STREQ(buff, "4294967295");
  }

  written = LIBC_NAMESPACE::sprintf(buff, "%lld", -9223372036854775807ll - 1ll);
  EXPECT_EQ(written, 20);
  ASSERT_STREQ(buff, "-9223372036854775808"); // ll min

  written = LIBC_NAMESPACE::sprintf(buff, "%w3d", 5807);
  EXPECT_EQ(written, 1);
  ASSERT_STREQ(buff, "7");

  written = LIBC_NAMESPACE::sprintf(buff, "%w3d", 1);
  EXPECT_EQ(written, 1);
  ASSERT_STREQ(buff, "1");

  written = LIBC_NAMESPACE::sprintf(buff, "%w64d", 9223372036854775807ll);
  EXPECT_EQ(written, 19);
  ASSERT_STREQ(buff, "9223372036854775807");

  written = LIBC_NAMESPACE::sprintf(buff, "%w-1d", 5807);
  EXPECT_EQ(written, 5);
  ASSERT_STREQ(buff, "%w-1d");

  written = LIBC_NAMESPACE::sprintf(buff, "%w0d", 5807);
  EXPECT_EQ(written, 4);
  ASSERT_STREQ(buff, "%w0d");

  written = LIBC_NAMESPACE::sprintf(buff, "%w999d", 9223372036854775807ll);
  EXPECT_EQ(written, 19);
  ASSERT_STREQ(buff, "9223372036854775807");

  written = LIBC_NAMESPACE::sprintf(buff, "%winvalid%w1d", 5807, 5807);
  EXPECT_EQ(written, 10);
  ASSERT_STREQ(buff, "%winvalid1");

  written = LIBC_NAMESPACE::sprintf(buff, "%w-1d%w1d", 5807, 5807);
  EXPECT_EQ(written, 6);
  ASSERT_STREQ(buff, "%w-1d1");

  char format[64];
  char uintmax[128];
  LIBC_NAMESPACE::sprintf(format, "%%w%du", sizeof(uintmax_t) * CHAR_BIT);
  const int uintmax_len =
      LIBC_NAMESPACE::sprintf(uintmax, "%ju", sizeof(uintmax_t) * CHAR_BIT);
  written = LIBC_NAMESPACE::sprintf(buff, format, sizeof(uintmax_t) * CHAR_BIT);
  EXPECT_EQ(written, uintmax_len);
  ASSERT_STREQ(buff, uintmax);

  written = LIBC_NAMESPACE::sprintf(buff, "%w64u", 18446744073709551615ull);
  EXPECT_EQ(written, 20);
  ASSERT_STREQ(buff, "18446744073709551615"); // ull max

  written =
      LIBC_NAMESPACE::sprintf(buff, "%w64d", -9223372036854775807ll - 1ll);
  EXPECT_EQ(written, 20);
  ASSERT_STREQ(buff, "-9223372036854775808"); // ll min

  written = LIBC_NAMESPACE::sprintf(buff, "%wf3d", 5807);
  EXPECT_EQ(written, 1);
  ASSERT_STREQ(buff, "7");

  written = LIBC_NAMESPACE::sprintf(buff, "%wf3d", 1);
  EXPECT_EQ(written, 1);
  ASSERT_STREQ(buff, "1");

  written = LIBC_NAMESPACE::sprintf(buff, "%wf64u", 18446744073709551615ull);
  EXPECT_EQ(written, 20);
  ASSERT_STREQ(buff, "18446744073709551615"); // ull max

  written =
      LIBC_NAMESPACE::sprintf(buff, "%wf64d", -9223372036854775807ll - 1ll);
  EXPECT_EQ(written, 20);
  ASSERT_STREQ(buff, "-9223372036854775808"); // ll min

  written = LIBC_NAMESPACE::sprintf(buff, "%wf0d", 5807);
  EXPECT_EQ(written, 5);
  ASSERT_STREQ(buff, "%wf0d");

  written = LIBC_NAMESPACE::sprintf(buff, "%wf-1d", 5807);
  EXPECT_EQ(written, 6);
  ASSERT_STREQ(buff, "%wf-1d");

  written = LIBC_NAMESPACE::sprintf(buff, "%wfinvalid%wf1d", 5807, 5807);
  EXPECT_EQ(written, 11);
  ASSERT_STREQ(buff, "%wfinvalid1");

  written = LIBC_NAMESPACE::sprintf(buff, "%wf-1d%wf1d", 5807, 5807);
  EXPECT_EQ(written, 7);
  ASSERT_STREQ(buff, "%wf-1d1");

  written = LIBC_NAMESPACE::sprintf(buff, "%wf999d", 9223372036854775807ll);
  EXPECT_EQ(written, 19);
  ASSERT_STREQ(buff, "9223372036854775807");

  // Min Width Tests.

  written = LIBC_NAMESPACE::sprintf(buff, "%4d", 789);
  EXPECT_EQ(written, 4);
  ASSERT_STREQ(buff, " 789");

  written = LIBC_NAMESPACE::sprintf(buff, "%2d", 987);
  EXPECT_EQ(written, 3);
  ASSERT_STREQ(buff, "987");

  // Precision Tests.

  written = LIBC_NAMESPACE::sprintf(buff, "%d", 0);
  EXPECT_EQ(written, 1);
  ASSERT_STREQ(buff, "0");

  written = LIBC_NAMESPACE::sprintf(buff, "%.0d", 0);
  EXPECT_EQ(written, 0);
  ASSERT_STREQ(buff, "");

  written = LIBC_NAMESPACE::sprintf(buff, "%.5d", 654);
  EXPECT_EQ(written, 5);
  ASSERT_STREQ(buff, "00654");

  written = LIBC_NAMESPACE::sprintf(buff, "%.5d", -321);
  EXPECT_EQ(written, 6);
  ASSERT_STREQ(buff, "-00321");

  written = LIBC_NAMESPACE::sprintf(buff, "%.2d", 135);
  EXPECT_EQ(written, 3);
  ASSERT_STREQ(buff, "135");

  // Flag Tests.

  written = LIBC_NAMESPACE::sprintf(buff, "%.5d", -321);
  EXPECT_EQ(written, 6);
  ASSERT_STREQ(buff, "-00321");

  written = LIBC_NAMESPACE::sprintf(buff, "%-5d", 246);
  EXPECT_EQ(written, 5);
  ASSERT_STREQ(buff, "246  ");

  written = LIBC_NAMESPACE::sprintf(buff, "%-5d", -147);
  EXPECT_EQ(written, 5);
  ASSERT_STREQ(buff, "-147 ");

  written = LIBC_NAMESPACE::sprintf(buff, "%+d", 258);
  EXPECT_EQ(written, 4);
  ASSERT_STREQ(buff, "+258");

  written = LIBC_NAMESPACE::sprintf(buff, "% d", 369);
  EXPECT_EQ(written, 4);
  ASSERT_STREQ(buff, " 369");

  written = LIBC_NAMESPACE::sprintf(buff, "%05d", 470);
  EXPECT_EQ(written, 5);
  ASSERT_STREQ(buff, "00470");

  written = LIBC_NAMESPACE::sprintf(buff, "%05d", -581);
  EXPECT_EQ(written, 5);
  ASSERT_STREQ(buff, "-0581");

  // Combined Tests.

  written = LIBC_NAMESPACE::sprintf(buff, "%+ u", 692);
  EXPECT_EQ(written, 3);
  ASSERT_STREQ(buff, "692");

  written = LIBC_NAMESPACE::sprintf(buff, "%+ -05d", 703);
  EXPECT_EQ(written, 5);
  ASSERT_STREQ(buff, "+703 ");

  written = LIBC_NAMESPACE::sprintf(buff, "%7.5d", 814);
  EXPECT_EQ(written, 7);
  ASSERT_STREQ(buff, "  00814");

  written = LIBC_NAMESPACE::sprintf(buff, "%7.5d", -925);
  EXPECT_EQ(written, 7);
  ASSERT_STREQ(buff, " -00925");

  written = LIBC_NAMESPACE::sprintf(buff, "%7.5d", 159);
  EXPECT_EQ(written, 7);
  ASSERT_STREQ(buff, "  00159");

  written = LIBC_NAMESPACE::sprintf(buff, "% -7.5d", 260);
  EXPECT_EQ(written, 7);
  ASSERT_STREQ(buff, " 00260 ");

  written = LIBC_NAMESPACE::sprintf(buff, "%5.4d", 10000);
  EXPECT_EQ(written, 5);
  ASSERT_STREQ(buff, "10000");

  // Multiple Conversion Tests.

  written = LIBC_NAMESPACE::sprintf(buff, "%10d %-10d", 456, -789);
  EXPECT_EQ(written, 21);
  ASSERT_STREQ(buff, "       456 -789      ");

  written = LIBC_NAMESPACE::sprintf(buff, "%-5.4d%+.4u", 75, 25);
  EXPECT_EQ(written, 9);
  ASSERT_STREQ(buff, "0075 0025");

  written = LIBC_NAMESPACE::sprintf(buff, "% 05hhi %+-0.5llu %-+ 06.3zd",
                                    256 + 127, 68719476736ll, size_t(2));
  EXPECT_EQ(written, 24);
  ASSERT_STREQ(buff, " 0127 68719476736 +002  ");
}

TEST(LlvmLibcSPrintfTest, HexConv) {
  char buff[64];
  int written;

  // Basic Tests.

  written = LIBC_NAMESPACE::sprintf(buff, "%x", 0x123a);
  EXPECT_EQ(written, 4);
  ASSERT_STREQ(buff, "123a");

  written = LIBC_NAMESPACE::sprintf(buff, "%X", 0x456b);
  EXPECT_EQ(written, 4);
  ASSERT_STREQ(buff, "456B");

  // Length Modifier Tests.

  written = LIBC_NAMESPACE::sprintf(buff, "%hhx", 0x10001);
  EXPECT_EQ(written, 1);
  ASSERT_STREQ(buff, "1");

  written = LIBC_NAMESPACE::sprintf(buff, "%llx", 0xffffffffffffffffull);
  EXPECT_EQ(written, 16);
  ASSERT_STREQ(buff, "ffffffffffffffff"); // ull max

  written = LIBC_NAMESPACE::sprintf(buff, "%tX", ~ptrdiff_t(0));
  if (sizeof(ptrdiff_t) == 8) {
    EXPECT_EQ(written, 16);
    ASSERT_STREQ(buff, "FFFFFFFFFFFFFFFF");
  } else if (sizeof(ptrdiff_t) == 4) {
    EXPECT_EQ(written, 8);
    ASSERT_STREQ(buff, "FFFFFFFF");
  }

  // Min Width Tests.

  written = LIBC_NAMESPACE::sprintf(buff, "%4x", 0x789);
  EXPECT_EQ(written, 4);
  ASSERT_STREQ(buff, " 789");

  written = LIBC_NAMESPACE::sprintf(buff, "%2X", 0x987);
  EXPECT_EQ(written, 3);
  ASSERT_STREQ(buff, "987");

  // Precision Tests.

  written = LIBC_NAMESPACE::sprintf(buff, "%x", 0);
  EXPECT_EQ(written, 1);
  ASSERT_STREQ(buff, "0");

  written = LIBC_NAMESPACE::sprintf(buff, "%.0x", 0);
  EXPECT_EQ(written, 0);
  ASSERT_STREQ(buff, "");

  written = LIBC_NAMESPACE::sprintf(buff, "%.5x", 0x1F3);
  EXPECT_EQ(written, 5);
  ASSERT_STREQ(buff, "001f3");

  written = LIBC_NAMESPACE::sprintf(buff, "%.2x", 0x135);
  EXPECT_EQ(written, 3);
  ASSERT_STREQ(buff, "135");

  // Flag Tests.

  written = LIBC_NAMESPACE::sprintf(buff, "%-5x", 0x246);
  EXPECT_EQ(written, 5);
  ASSERT_STREQ(buff, "246  ");

  written = LIBC_NAMESPACE::sprintf(buff, "%#x", 0xd3f);
  EXPECT_EQ(written, 5);
  ASSERT_STREQ(buff, "0xd3f");

  written = LIBC_NAMESPACE::sprintf(buff, "%#x", 0);
  EXPECT_EQ(written, 1);
  ASSERT_STREQ(buff, "0");

  written = LIBC_NAMESPACE::sprintf(buff, "%#X", 0xE40);
  EXPECT_EQ(written, 5);
  ASSERT_STREQ(buff, "0XE40");

  written = LIBC_NAMESPACE::sprintf(buff, "%05x", 0x470);
  EXPECT_EQ(written, 5);
  ASSERT_STREQ(buff, "00470");

  written = LIBC_NAMESPACE::sprintf(buff, "%0#6x", 0x8c3);
  EXPECT_EQ(written, 6);
  ASSERT_STREQ(buff, "0x08c3");

  written = LIBC_NAMESPACE::sprintf(buff, "%-#6x", 0x5f0);
  EXPECT_EQ(written, 6);
  ASSERT_STREQ(buff, "0x5f0 ");

  // Combined Tests.

  written = LIBC_NAMESPACE::sprintf(buff, "%#-07x", 0x703);
  EXPECT_EQ(written, 7);
  ASSERT_STREQ(buff, "0x703  ");

  written = LIBC_NAMESPACE::sprintf(buff, "%7.5x", 0x814);
  EXPECT_EQ(written, 7);
  ASSERT_STREQ(buff, "  00814");

  written = LIBC_NAMESPACE::sprintf(buff, "%#9.5X", 0x9d4);
  EXPECT_EQ(written, 9);
  ASSERT_STREQ(buff, "  0X009D4");

  written = LIBC_NAMESPACE::sprintf(buff, "%#.x", 0);
  EXPECT_EQ(written, 0);
  ASSERT_STREQ(buff, "");

  written = LIBC_NAMESPACE::sprintf(buff, "%-7.5x", 0x260);
  EXPECT_EQ(written, 7);
  ASSERT_STREQ(buff, "00260  ");

  written = LIBC_NAMESPACE::sprintf(buff, "%5.4x", 0x10000);
  EXPECT_EQ(written, 5);
  ASSERT_STREQ(buff, "10000");

  // Multiple Conversion Tests.

  written = LIBC_NAMESPACE::sprintf(buff, "%10X %-#10x", 0x45b, 0x789);
  EXPECT_EQ(written, 21);
  ASSERT_STREQ(buff, "       45B 0x789     ");

  written = LIBC_NAMESPACE::sprintf(buff, "%-5.4x%#.4x", 0x75, 0x25);
  EXPECT_EQ(written, 11);
  ASSERT_STREQ(buff, "0075 0x0025");

  written = LIBC_NAMESPACE::sprintf(buff, "%04hhX %#.5llx %-6.3zX", 256 + 0x7f,
                                    0x1000000000ll, size_t(2));
  EXPECT_EQ(written, 24);
  ASSERT_STREQ(buff, "007F 0x1000000000 002   ");
}

TEST(LlvmLibcSPrintfTest, BinConv) {
  char buff[64];
  int written;

  // Basic Tests.

  written = LIBC_NAMESPACE::sprintf(buff, "%b", 42);
  EXPECT_EQ(written, 6);
  ASSERT_STREQ(buff, "101010");

  written = LIBC_NAMESPACE::sprintf(buff, "%B", 12081991);
  EXPECT_EQ(written, 24);
  ASSERT_STREQ(buff, "101110000101101101000111");

  // Min Width Tests.

  written = LIBC_NAMESPACE::sprintf(buff, "%10b", 0b101010);
  EXPECT_EQ(written, 10);
  ASSERT_STREQ(buff, "    101010");

  written = LIBC_NAMESPACE::sprintf(buff, "%2B", 0b101010);
  EXPECT_EQ(written, 6);
  ASSERT_STREQ(buff, "101010");

  // Precision Tests.

  written = LIBC_NAMESPACE::sprintf(buff, "%b", 0);
  EXPECT_EQ(written, 1);
  ASSERT_STREQ(buff, "0");

  written = LIBC_NAMESPACE::sprintf(buff, "%.0b", 0);
  EXPECT_EQ(written, 0);
  ASSERT_STREQ(buff, "");

  written = LIBC_NAMESPACE::sprintf(buff, "%.5b", 0b111);
  EXPECT_EQ(written, 5);
  ASSERT_STREQ(buff, "00111");

  written = LIBC_NAMESPACE::sprintf(buff, "%.2b", 0b111);
  EXPECT_EQ(written, 3);
  ASSERT_STREQ(buff, "111");

  written = LIBC_NAMESPACE::sprintf(buff, "%3b", 0b111);
  EXPECT_EQ(written, 3);
  ASSERT_STREQ(buff, "111");

  // Flag Tests.

  written = LIBC_NAMESPACE::sprintf(buff, "%-5b", 0b111);
  EXPECT_EQ(written, 5);
  ASSERT_STREQ(buff, "111  ");

  written = LIBC_NAMESPACE::sprintf(buff, "%#b", 0b111);
  EXPECT_EQ(written, 5);
  ASSERT_STREQ(buff, "0b111");

  written = LIBC_NAMESPACE::sprintf(buff, "%#b", 0);
  EXPECT_EQ(written, 1);
  ASSERT_STREQ(buff, "0");

  written = LIBC_NAMESPACE::sprintf(buff, "%#B", 0b111);
  EXPECT_EQ(written, 5);
  ASSERT_STREQ(buff, "0B111");

  written = LIBC_NAMESPACE::sprintf(buff, "%05b", 0b111);
  EXPECT_EQ(written, 5);
  ASSERT_STREQ(buff, "00111");

  written = LIBC_NAMESPACE::sprintf(buff, "%0#6b", 0b111);
  EXPECT_EQ(written, 6);
  ASSERT_STREQ(buff, "0b0111");

  written = LIBC_NAMESPACE::sprintf(buff, "%-#6b", 0b111);
  EXPECT_EQ(written, 6);
  ASSERT_STREQ(buff, "0b111 ");

  // Combined Tests.

  written = LIBC_NAMESPACE::sprintf(buff, "%#-07b", 0b111);
  EXPECT_EQ(written, 7);
  ASSERT_STREQ(buff, "0b111  ");

  written = LIBC_NAMESPACE::sprintf(buff, "%7.5b", 0b111);
  EXPECT_EQ(written, 7);
  ASSERT_STREQ(buff, "  00111");

  written = LIBC_NAMESPACE::sprintf(buff, "%#9.5B", 0b111);
  EXPECT_EQ(written, 9);
  ASSERT_STREQ(buff, "  0B00111");

  written = LIBC_NAMESPACE::sprintf(buff, "%#.b", 0);
  EXPECT_EQ(written, 0);
  ASSERT_STREQ(buff, "");

  written = LIBC_NAMESPACE::sprintf(buff, "%-7.5b", 0b111);
  EXPECT_EQ(written, 7);
  ASSERT_STREQ(buff, "00111  ");

  written = LIBC_NAMESPACE::sprintf(buff, "%5.4b", 0b1111);
  EXPECT_EQ(written, 5);
  ASSERT_STREQ(buff, " 1111");

  // Multiple Conversion Tests.

  written = LIBC_NAMESPACE::sprintf(buff, "%10B %-#10b", 0b101, 0b110);
  EXPECT_EQ(written, 21);
  ASSERT_STREQ(buff, "       101 0b110     ");

  written = LIBC_NAMESPACE::sprintf(buff, "%-5.4b%#.4b", 0b101, 0b110);
  EXPECT_EQ(written, 11);
  ASSERT_STREQ(buff, "0101 0b0110");
}

TEST(LlvmLibcSPrintfTest, PointerConv) {
  char buff[64];
  int written;

  written = LIBC_NAMESPACE::sprintf(buff, "%p", nullptr);
  EXPECT_EQ(written, 9);
  ASSERT_STREQ(buff, "(nullptr)");

  written = LIBC_NAMESPACE::sprintf(buff, "%p", 0x1a2b3c4d);
  EXPECT_EQ(written, 10);
  ASSERT_STREQ(buff, "0x1a2b3c4d");

  if constexpr (sizeof(void *) > 4) {
    written = LIBC_NAMESPACE::sprintf(buff, "%p", 0x1a2b3c4d5e6f7081);
    EXPECT_EQ(written, 18);
    ASSERT_STREQ(buff, "0x1a2b3c4d5e6f7081");
  }

  written = LIBC_NAMESPACE::sprintf(buff, "%p", &written);
  EXPECT_GT(written, 0);

  // Width tests:

  written = LIBC_NAMESPACE::sprintf(buff, "%20p", nullptr);
  EXPECT_EQ(written, 20);
  ASSERT_STREQ(buff, "           (nullptr)");

  written = LIBC_NAMESPACE::sprintf(buff, "%20p", 0x1a2b3c4d);
  EXPECT_EQ(written, 20);
  ASSERT_STREQ(buff, "          0x1a2b3c4d");

  // Flag tests:

  written = LIBC_NAMESPACE::sprintf(buff, "%-20p", nullptr);
  EXPECT_EQ(written, 20);
  ASSERT_STREQ(buff, "(nullptr)           ");

  written = LIBC_NAMESPACE::sprintf(buff, "%-20p", 0x1a2b3c4d);
  EXPECT_EQ(written, 20);
  ASSERT_STREQ(buff, "0x1a2b3c4d          ");

  // Using the 0 flag is technically undefined, but here we're following the
  // convention of matching the behavior of %#x.
  written = LIBC_NAMESPACE::sprintf(buff, "%020p", 0x1a2b3c4d);
  EXPECT_EQ(written, 20);
  ASSERT_STREQ(buff, "0x00000000001a2b3c4d");

  // Precision tests:
  // These are all undefined behavior. The precision option is undefined for %p.

  // Precision specifies the number of characters for a string conversion.
  written = LIBC_NAMESPACE::sprintf(buff, "%.5p", nullptr);
  EXPECT_EQ(written, 5);
  ASSERT_STREQ(buff, "(null");

  // Precision specifies the number of digits to be written for %x conversions,
  // and the "0x" doesn't count as part of the digits.
  written = LIBC_NAMESPACE::sprintf(buff, "%.20p", 0x1a2b3c4d);
  EXPECT_EQ(written, 22);
  ASSERT_STREQ(buff, "0x0000000000001a2b3c4d");
}

TEST(LlvmLibcSPrintfTest, OctConv) {
  char buff[64];
  int written;

  // Basic Tests.

  written = LIBC_NAMESPACE::sprintf(buff, "%o", 01234);
  EXPECT_EQ(written, 4);
  ASSERT_STREQ(buff, "1234");

  written = LIBC_NAMESPACE::sprintf(buff, "%o", 04567);
  EXPECT_EQ(written, 4);
  ASSERT_STREQ(buff, "4567");

  // Length Modifier Tests.

  written = LIBC_NAMESPACE::sprintf(buff, "%hho", 0401);
  EXPECT_EQ(written, 1);
  ASSERT_STREQ(buff, "1");

  written = LIBC_NAMESPACE::sprintf(buff, "%llo", 01777777777777777777777ull);
  EXPECT_EQ(written, 22);
  ASSERT_STREQ(buff, "1777777777777777777777"); // ull max

  written = LIBC_NAMESPACE::sprintf(buff, "%to", ~ptrdiff_t(0));
  if (sizeof(ptrdiff_t) == 8) {
    EXPECT_EQ(written, 22);
    ASSERT_STREQ(buff, "1777777777777777777777");
  } else if (sizeof(ptrdiff_t) == 4) {
    EXPECT_EQ(written, 11);
    ASSERT_STREQ(buff, "37777777777");
  }

  // Min Width Tests.

  written = LIBC_NAMESPACE::sprintf(buff, "%4o", 0701);
  EXPECT_EQ(written, 4);
  ASSERT_STREQ(buff, " 701");

  written = LIBC_NAMESPACE::sprintf(buff, "%2o", 0107);
  EXPECT_EQ(written, 3);
  ASSERT_STREQ(buff, "107");

  // Precision Tests.

  written = LIBC_NAMESPACE::sprintf(buff, "%o", 0);
  EXPECT_EQ(written, 1);
  ASSERT_STREQ(buff, "0");

  written = LIBC_NAMESPACE::sprintf(buff, "%.0o", 0);
  EXPECT_EQ(written, 0);
  ASSERT_STREQ(buff, "");

  written = LIBC_NAMESPACE::sprintf(buff, "%.5o", 0153);
  EXPECT_EQ(written, 5);
  ASSERT_STREQ(buff, "00153");

  written = LIBC_NAMESPACE::sprintf(buff, "%.2o", 0135);
  EXPECT_EQ(written, 3);
  ASSERT_STREQ(buff, "135");

  // Flag Tests.

  written = LIBC_NAMESPACE::sprintf(buff, "%-5o", 0246);
  EXPECT_EQ(written, 5);
  ASSERT_STREQ(buff, "246  ");

  written = LIBC_NAMESPACE::sprintf(buff, "%#o", 0234);
  EXPECT_EQ(written, 4);
  ASSERT_STREQ(buff, "0234");

  written = LIBC_NAMESPACE::sprintf(buff, "%#o", 0);
  EXPECT_EQ(written, 1);
  ASSERT_STREQ(buff, "0");

  written = LIBC_NAMESPACE::sprintf(buff, "%05o", 0470);
  EXPECT_EQ(written, 5);
  ASSERT_STREQ(buff, "00470");

  written = LIBC_NAMESPACE::sprintf(buff, "%0#6o", 0753);
  EXPECT_EQ(written, 6);
  ASSERT_STREQ(buff, "000753");

  written = LIBC_NAMESPACE::sprintf(buff, "%-#6o", 0642);
  EXPECT_EQ(written, 6);
  ASSERT_STREQ(buff, "0642  ");

  // Combined Tests.

  written = LIBC_NAMESPACE::sprintf(buff, "%#-07o", 0703);
  EXPECT_EQ(written, 7);
  ASSERT_STREQ(buff, "0703   ");

  written = LIBC_NAMESPACE::sprintf(buff, "%#.o", 0);
  EXPECT_EQ(written, 1);
  ASSERT_STREQ(buff, "0");

  written = LIBC_NAMESPACE::sprintf(buff, "%7.5o", 0314);
  EXPECT_EQ(written, 7);
  ASSERT_STREQ(buff, "  00314");

  written = LIBC_NAMESPACE::sprintf(buff, "%#9.5o", 0234);
  EXPECT_EQ(written, 9);
  ASSERT_STREQ(buff, "    00234");

  written = LIBC_NAMESPACE::sprintf(buff, "%-7.5o", 0260);
  EXPECT_EQ(written, 7);
  ASSERT_STREQ(buff, "00260  ");

  written = LIBC_NAMESPACE::sprintf(buff, "%5.4o", 010000);
  EXPECT_EQ(written, 5);
  ASSERT_STREQ(buff, "10000");

  // Multiple Conversion Tests.

  written = LIBC_NAMESPACE::sprintf(buff, "%10o %-#10o", 0456, 0123);
  EXPECT_EQ(written, 21);
  ASSERT_STREQ(buff, "       456 0123      ");

  written = LIBC_NAMESPACE::sprintf(buff, "%-5.4o%#.4o", 075, 025);
  EXPECT_EQ(written, 9);
  ASSERT_STREQ(buff, "0075 0025");

  written = LIBC_NAMESPACE::sprintf(buff, "%04hho %#.5llo %-6.3zo", 256 + 077,
                                    01000000000000ll, size_t(2));
  EXPECT_EQ(written, 26);
  ASSERT_STREQ(buff, "0077 01000000000000 002   ");
}

#ifndef LIBC_COPT_PRINTF_DISABLE_FLOAT

TEST_F(LlvmLibcSPrintfTest, FloatHexExpConv) {
  ForceRoundingMode r(RoundingMode::Nearest);
  double inf = LIBC_NAMESPACE::fputil::FPBits<double>::inf().get_val();
  double nan = LIBC_NAMESPACE::fputil::FPBits<double>::quiet_nan().get_val();
  written = LIBC_NAMESPACE::sprintf(buff, "%a", 1.0);
  ASSERT_STREQ_LEN(written, buff, "0x1p+0");

  written = LIBC_NAMESPACE::sprintf(buff, "%A", -1.0);
  ASSERT_STREQ_LEN(written, buff, "-0X1P+0");

  written = LIBC_NAMESPACE::sprintf(buff, "%a", -0x1.abcdef12345p0);
  ASSERT_STREQ_LEN(written, buff, "-0x1.abcdef12345p+0");

  written = LIBC_NAMESPACE::sprintf(buff, "%A", 0x1.abcdef12345p0);
  ASSERT_STREQ_LEN(written, buff, "0X1.ABCDEF12345P+0");

  written = LIBC_NAMESPACE::sprintf(buff, "%a", 0.0);
  ASSERT_STREQ_LEN(written, buff, "0x0p+0");

  written = LIBC_NAMESPACE::sprintf(buff, "%a", 1.0e100);
  ASSERT_STREQ_LEN(written, buff, "0x1.249ad2594c37dp+332");

  written = LIBC_NAMESPACE::sprintf(buff, "%a", 0.1);
  ASSERT_STREQ_LEN(written, buff, "0x1.999999999999ap-4");

  // Subnormal Tests.

  written = LIBC_NAMESPACE::sprintf(buff, "%a", 0x1.0p-1027);
  ASSERT_STREQ_LEN(written, buff, "0x0.08p-1022");

  written = LIBC_NAMESPACE::sprintf(buff, "%a", 0x1.0p-1025);
  ASSERT_STREQ_LEN(written, buff, "0x0.2p-1022");

  written = LIBC_NAMESPACE::sprintf(buff, "%a", 0x1.0p-1023);
  ASSERT_STREQ_LEN(written, buff, "0x0.8p-1022");

  written = LIBC_NAMESPACE::sprintf(buff, "%a", 0x1.0p-1022);
  ASSERT_STREQ_LEN(written, buff, "0x1p-1022");

  written = LIBC_NAMESPACE::sprintf(buff, "%a", 0x1.0p-1074);
  ASSERT_STREQ_LEN(written, buff, "0x0.0000000000001p-1022");

  // Inf/Nan Tests.

  written = LIBC_NAMESPACE::sprintf(buff, "%a", inf);
  ASSERT_STREQ_LEN(written, buff, "inf");

  written = LIBC_NAMESPACE::sprintf(buff, "%A", -inf);
  ASSERT_STREQ_LEN(written, buff, "-INF");

  written = LIBC_NAMESPACE::sprintf(buff, "%a", nan);
  ASSERT_STREQ_LEN(written, buff, "nan");

  written = LIBC_NAMESPACE::sprintf(buff, "%A", -nan);
  ASSERT_STREQ_LEN(written, buff, "-NAN");

  // Length Modifier Tests.

  written = LIBC_NAMESPACE::sprintf(buff, "%La", 0.1L);
#if defined(LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80)
  ASSERT_STREQ_LEN(written, buff, "0xc.ccccccccccccccdp-7");
#elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT64)
  ASSERT_STREQ_LEN(written, buff, "0x1.999999999999ap-4");
#elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT128)
  ASSERT_STREQ_LEN(written, buff, "0x1.999999999999999999999999999ap-4");
#endif

  written = LIBC_NAMESPACE::sprintf(buff, "%La", 1.0e1000L);
#if defined(LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80)
  ASSERT_STREQ_LEN(written, buff, "0xf.38db1f9dd3dac05p+3318");
#elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT64)
  ASSERT_STREQ_LEN(written, buff, "inf");
#elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT128)
  ASSERT_STREQ_LEN(written, buff, "0x1.e71b63f3ba7b580af1a52d2a7379p+3321");
#endif

  written = LIBC_NAMESPACE::sprintf(buff, "%La", 1.0e-1000L);
#if defined(LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80)
  ASSERT_STREQ_LEN(written, buff, "0x8.68a9188a89e1467p-3325");
#elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT64)
  ASSERT_STREQ_LEN(written, buff, "0x0p+0");
#elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT128)
  ASSERT_STREQ_LEN(written, buff, "0x1.0d152311513c28ce202627c06ec2p-3322");
#endif

  // Min Width Tests.

  written = LIBC_NAMESPACE::sprintf(buff, "%15a", 1.0);
  ASSERT_STREQ_LEN(written, buff, "         0x1p+0");

  written = LIBC_NAMESPACE::sprintf(buff, "%15a", -1.0);
  ASSERT_STREQ_LEN(written, buff, "        -0x1p+0");

  written = LIBC_NAMESPACE::sprintf(buff, "%15a", 1.0e10);
  ASSERT_STREQ_LEN(written, buff, " 0x1.2a05f2p+33");

  written = LIBC_NAMESPACE::sprintf(buff, "%15a", -1.0e10);
  ASSERT_STREQ_LEN(written, buff, "-0x1.2a05f2p+33");

  written = LIBC_NAMESPACE::sprintf(buff, "%10a", 1.0e10);
  ASSERT_STREQ_LEN(written, buff, "0x1.2a05f2p+33");

  written = LIBC_NAMESPACE::sprintf(buff, "%5a", inf);
  ASSERT_STREQ_LEN(written, buff, "  inf");

  written = LIBC_NAMESPACE::sprintf(buff, "%5a", -nan);
  ASSERT_STREQ_LEN(written, buff, " -nan");

  // Precision Tests.

  written = LIBC_NAMESPACE::sprintf(buff, "%.1a", 1.0);
  ASSERT_STREQ_LEN(written, buff, "0x1.0p+0");

  written = LIBC_NAMESPACE::sprintf(buff, "%.1a", 0.0);
  ASSERT_STREQ_LEN(written, buff, "0x0.0p+0");

  written = LIBC_NAMESPACE::sprintf(buff, "%.1a", 0.1);
  ASSERT_STREQ_LEN(written, buff, "0x1.ap-4");

  written = LIBC_NAMESPACE::sprintf(buff, "%.1a", 0x1.0fp0);
  ASSERT_STREQ_LEN(written, buff, "0x1.1p+0");

  written = LIBC_NAMESPACE::sprintf(buff, "%.1a", 0x1.07p0);
  ASSERT_STREQ_LEN(written, buff, "0x1.0p+0");

  written = LIBC_NAMESPACE::sprintf(buff, "%.1a", 0x1.08p0);
  ASSERT_STREQ_LEN(written, buff, "0x1.0p+0");

  written = LIBC_NAMESPACE::sprintf(buff, "%.1a", 0x1.18p0);
  ASSERT_STREQ_LEN(written, buff, "0x1.2p+0");

  written = LIBC_NAMESPACE::sprintf(buff, "%.1a", 0x1.ffp0);
  ASSERT_STREQ_LEN(written, buff, "0x2.0p+0");

  written = LIBC_NAMESPACE::sprintf(buff, "%.5a", 1.25);
  ASSERT_STREQ_LEN(written, buff, "0x1.40000p+0");

  written = LIBC_NAMESPACE::sprintf(buff, "%.0a", 1.25);
  ASSERT_STREQ_LEN(written, buff, "0x1p+0");

  written = LIBC_NAMESPACE::sprintf(buff, "%.0a", 1.75);
  ASSERT_STREQ_LEN(written, buff, "0x2p+0");

  written = LIBC_NAMESPACE::sprintf(buff, "%.1a", 0x1.0p-1023);
  ASSERT_STREQ_LEN(written, buff, "0x0.8p-1022");

  written = LIBC_NAMESPACE::sprintf(buff, "%.1a", 0x1.8p-1023);
  ASSERT_STREQ_LEN(written, buff, "0x0.cp-1022");

  written = LIBC_NAMESPACE::sprintf(buff, "%.1a", 0x1.0p-1024);
  ASSERT_STREQ_LEN(written, buff, "0x0.4p-1022");

  written = LIBC_NAMESPACE::sprintf(buff, "%.0a", 0x1.0p-1023);
  ASSERT_STREQ_LEN(written, buff, "0x0p-1022");

  written = LIBC_NAMESPACE::sprintf(buff, "%.0a", 0x1.8p-1023);
  ASSERT_STREQ_LEN(written, buff, "0x1p-1022");

  written = LIBC_NAMESPACE::sprintf(buff, "%.0a", 0x1.0p-1024);
  ASSERT_STREQ_LEN(written, buff, "0x0p-1022");

  written = LIBC_NAMESPACE::sprintf(buff, "%.2a", 0x1.0p-1027);
  ASSERT_STREQ_LEN(written, buff, "0x0.08p-1022");

  written = LIBC_NAMESPACE::sprintf(buff, "%.1a", 0x1.0p-1027);
  ASSERT_STREQ_LEN(written, buff, "0x0.0p-1022");

  written = LIBC_NAMESPACE::sprintf(buff, "%.5a", 0.0);
  ASSERT_STREQ_LEN(written, buff, "0x0.00000p+0");

  written = LIBC_NAMESPACE::sprintf(buff, "%.5a", 0x1.008p0);
  ASSERT_STREQ_LEN(written, buff, "0x1.00800p+0");

  written = LIBC_NAMESPACE::sprintf(buff, "%.5a", 0x1.008p10);
  ASSERT_STREQ_LEN(written, buff, "0x1.00800p+10");

  written = LIBC_NAMESPACE::sprintf(buff, "%.5a", nan);
  ASSERT_STREQ_LEN(written, buff, "nan");

  written = LIBC_NAMESPACE::sprintf(buff, "%La", 0.0L);
  ASSERT_STREQ_LEN(written, buff, "0x0p+0");

  written = LIBC_NAMESPACE::sprintf(buff, "%.1La", 0.1L);
#if defined(LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80)
  ASSERT_STREQ_LEN(written, buff, "0xc.dp-7");
#elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT64)
  ASSERT_STREQ_LEN(written, buff, "0x1.ap-4");
#elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT128)
  ASSERT_STREQ_LEN(written, buff, "0x1.ap-4");
#endif

  written = LIBC_NAMESPACE::sprintf(buff, "%.1La", 0xf.fffffffffffffffp16380L);
#if defined(LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80)
  ASSERT_STREQ_LEN(written, buff, "0x1.0p+16384");
#elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT64)
  ASSERT_STREQ_LEN(written, buff, "inf");
#elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT128)
  ASSERT_STREQ_LEN(written, buff, "0x2.0p+16383");
#endif

  // Rounding Mode Tests.

  if (ForceRoundingMode r(RoundingMode::Nearest); r.success) {
    written = LIBC_NAMESPACE::sprintf(buff, "%.1a", 0x1.08p0);
    ASSERT_STREQ_LEN(written, buff, "0x1.0p+0");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1a", 0x1.18p0);
    ASSERT_STREQ_LEN(written, buff, "0x1.2p+0");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1a", 0x1.04p0);
    ASSERT_STREQ_LEN(written, buff, "0x1.0p+0");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1a", 0x1.14p0);
    ASSERT_STREQ_LEN(written, buff, "0x1.1p+0");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1a", -0x1.08p0);
    ASSERT_STREQ_LEN(written, buff, "-0x1.0p+0");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1a", -0x1.18p0);
    ASSERT_STREQ_LEN(written, buff, "-0x1.2p+0");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1a", -0x1.04p0);
    ASSERT_STREQ_LEN(written, buff, "-0x1.0p+0");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1a", -0x1.14p0);
    ASSERT_STREQ_LEN(written, buff, "-0x1.1p+0");
  }

  if (ForceRoundingMode r(RoundingMode::Upward); r.success) {
    written = LIBC_NAMESPACE::sprintf(buff, "%.1a", 0x1.08p0);
    ASSERT_STREQ_LEN(written, buff, "0x1.1p+0");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1a", 0x1.18p0);
    ASSERT_STREQ_LEN(written, buff, "0x1.2p+0");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1a", 0x1.04p0);
    ASSERT_STREQ_LEN(written, buff, "0x1.1p+0");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1a", 0x1.14p0);
    ASSERT_STREQ_LEN(written, buff, "0x1.2p+0");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1a", -0x1.08p0);
    ASSERT_STREQ_LEN(written, buff, "-0x1.0p+0");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1a", -0x1.18p0);
    ASSERT_STREQ_LEN(written, buff, "-0x1.1p+0");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1a", -0x1.04p0);
    ASSERT_STREQ_LEN(written, buff, "-0x1.0p+0");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1a", -0x1.14p0);
    ASSERT_STREQ_LEN(written, buff, "-0x1.1p+0");
  }

  if (ForceRoundingMode r(RoundingMode::Downward); r.success) {
    written = LIBC_NAMESPACE::sprintf(buff, "%.1a", 0x1.08p0);
    ASSERT_STREQ_LEN(written, buff, "0x1.0p+0");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1a", 0x1.18p0);
    ASSERT_STREQ_LEN(written, buff, "0x1.1p+0");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1a", 0x1.04p0);
    ASSERT_STREQ_LEN(written, buff, "0x1.0p+0");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1a", 0x1.14p0);
    ASSERT_STREQ_LEN(written, buff, "0x1.1p+0");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1a", -0x1.08p0);
    ASSERT_STREQ_LEN(written, buff, "-0x1.1p+0");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1a", -0x1.18p0);
    ASSERT_STREQ_LEN(written, buff, "-0x1.2p+0");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1a", -0x1.04p0);
    ASSERT_STREQ_LEN(written, buff, "-0x1.1p+0");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1a", -0x1.14p0);
    ASSERT_STREQ_LEN(written, buff, "-0x1.2p+0");
  }

  if (ForceRoundingMode r(RoundingMode::TowardZero); r.success) {
    written = LIBC_NAMESPACE::sprintf(buff, "%.1a", 0x1.08p0);
    ASSERT_STREQ_LEN(written, buff, "0x1.0p+0");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1a", 0x1.18p0);
    ASSERT_STREQ_LEN(written, buff, "0x1.1p+0");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1a", 0x1.04p0);
    ASSERT_STREQ_LEN(written, buff, "0x1.0p+0");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1a", 0x1.14p0);
    ASSERT_STREQ_LEN(written, buff, "0x1.1p+0");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1a", -0x1.08p0);
    ASSERT_STREQ_LEN(written, buff, "-0x1.0p+0");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1a", -0x1.18p0);
    ASSERT_STREQ_LEN(written, buff, "-0x1.1p+0");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1a", -0x1.04p0);
    ASSERT_STREQ_LEN(written, buff, "-0x1.0p+0");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1a", -0x1.14p0);
    ASSERT_STREQ_LEN(written, buff, "-0x1.1p+0");
  }

  // Flag Tests.

  written = LIBC_NAMESPACE::sprintf(buff, "%+a", nan);
  ASSERT_STREQ_LEN(written, buff, "+nan");

  written = LIBC_NAMESPACE::sprintf(buff, "% A", inf);
  ASSERT_STREQ_LEN(written, buff, " INF");

  written = LIBC_NAMESPACE::sprintf(buff, "%-5a", inf);
  ASSERT_STREQ_LEN(written, buff, "inf  ");

  written = LIBC_NAMESPACE::sprintf(buff, "%+-5A", nan);
  ASSERT_STREQ_LEN(written, buff, "+NAN ");

  written = LIBC_NAMESPACE::sprintf(buff, "%+a", 1.0);
  ASSERT_STREQ_LEN(written, buff, "+0x1p+0");

  written = LIBC_NAMESPACE::sprintf(buff, "% a", 0.0);
  ASSERT_STREQ_LEN(written, buff, " 0x0p+0");

  written = LIBC_NAMESPACE::sprintf(buff, "%-10a", 1.5);
  ASSERT_STREQ_LEN(written, buff, "0x1.8p+0  ");

  written = LIBC_NAMESPACE::sprintf(buff, "%#a", 1.0);
  ASSERT_STREQ_LEN(written, buff, "0x1.p+0");

  written = LIBC_NAMESPACE::sprintf(buff, "%#.0a", 1.5);
  ASSERT_STREQ_LEN(written, buff, "0x2.p+0");

  written = LIBC_NAMESPACE::sprintf(buff, "%010a", 1.5);
  ASSERT_STREQ_LEN(written, buff, "0x001.8p+0");

  written = LIBC_NAMESPACE::sprintf(buff, "%+- #0a", 0.0);
  ASSERT_STREQ_LEN(written, buff, "+0x0.p+0");

  // Combined Tests.

  written = LIBC_NAMESPACE::sprintf(buff, "%12.3a %-12.3A", 0.1, 256.0);
  ASSERT_STREQ_LEN(written, buff, "  0x1.99ap-4 0X1.000P+8  ");

  written = LIBC_NAMESPACE::sprintf(buff, "%+-#12.3a % 012.3a", 0.1256, 1256.0);
  ASSERT_STREQ_LEN(written, buff, "+0x1.014p-3   0x1.3a0p+10");

  // These tests check that the padding is properly calculated based on the
  // min_width field. Specifically, they check that the extra zeroes added by
  // the high precision are accounted for correctly.
  written = LIBC_NAMESPACE::sprintf(buff, "%50.50a", 0x1.0p0);
  ASSERT_STREQ_LEN(written, buff,
                   "0x1.00000000000000000000000000000000000000000000000000p+0");

  // The difference with this test is that the formatted number is exactly 57
  // characters, so padding to 58 adds a space.
  written = LIBC_NAMESPACE::sprintf(buff, "%58.50a", 0x1.0p0);
  ASSERT_STREQ_LEN(
      written, buff,
      " 0x1.00000000000000000000000000000000000000000000000000p+0");
}

TEST_F(LlvmLibcSPrintfTest, FloatDecimalConv) {
  ForceRoundingMode r(RoundingMode::Nearest);
  double inf = LIBC_NAMESPACE::fputil::FPBits<double>::inf().get_val();
  double nan = LIBC_NAMESPACE::fputil::FPBits<double>::quiet_nan().get_val();
  long double ld_inf =
      LIBC_NAMESPACE::fputil::FPBits<long double>::inf().get_val();
  long double ld_nan =
      LIBC_NAMESPACE::fputil::FPBits<long double>::quiet_nan().get_val();

  char big_buff[10000]; // Used for long doubles and other extremely wide
                        // numbers.

  written = LIBC_NAMESPACE::sprintf(buff, "%f", 1.0);
  ASSERT_STREQ_LEN(written, buff, "1.000000");

  written = LIBC_NAMESPACE::sprintf(buff, "%F", -1.0);
  ASSERT_STREQ_LEN(written, buff, "-1.000000");

  written = LIBC_NAMESPACE::sprintf(buff, "%f", -1.234567);
  ASSERT_STREQ_LEN(written, buff, "-1.234567");

  written = LIBC_NAMESPACE::sprintf(buff, "%f", 0.0);
  ASSERT_STREQ_LEN(written, buff, "0.000000");

  written = LIBC_NAMESPACE::sprintf(buff, "%f", 1.5);
  ASSERT_STREQ_LEN(written, buff, "1.500000");

  written = LIBC_NAMESPACE::sprintf(buff, "%f", 1e300);
  ASSERT_STREQ_LEN(
      written, buff,
      "100000000000000005250476025520442024870446858110815915491585411551180245"
      "798890819578637137508044786404370444383288387817694252323536043057564479"
      "218478670698284838720092657580373783023379478809005936895323497079994508"
      "111903896764088007465274278014249457925878882005684283811566947219638686"
      "5459400540160.000000");

  written = LIBC_NAMESPACE::sprintf(buff, "%f", 0.1);
  ASSERT_STREQ_LEN(written, buff, "0.100000");

  written = LIBC_NAMESPACE::sprintf(buff, "%f", 1234567890123456789.0);
  ASSERT_STREQ_LEN(written, buff, "1234567890123456768.000000");

  written = LIBC_NAMESPACE::sprintf(buff, "%f", 9999999999999.99);
  ASSERT_STREQ_LEN(written, buff, "9999999999999.990234");

  // Simple Subnormal Tests.

  written = LIBC_NAMESPACE::sprintf(buff, "%f", 0x1.0p-1027);
  ASSERT_STREQ_LEN(written, buff, "0.000000");

  written = LIBC_NAMESPACE::sprintf(buff, "%f", 0x1.0p-1074);
  ASSERT_STREQ_LEN(written, buff, "0.000000");

  // Inf/Nan Tests.

  written = LIBC_NAMESPACE::sprintf(buff, "%f", inf);
  ASSERT_STREQ_LEN(written, buff, "inf");

  written = LIBC_NAMESPACE::sprintf(buff, "%F", -inf);
  ASSERT_STREQ_LEN(written, buff, "-INF");

  written = LIBC_NAMESPACE::sprintf(buff, "%f", nan);
  ASSERT_STREQ_LEN(written, buff, "nan");

  written = LIBC_NAMESPACE::sprintf(buff, "%F", -nan);
  ASSERT_STREQ_LEN(written, buff, "-NAN");

  written = LIBC_NAMESPACE::sprintf(buff, "%Lf", ld_inf);
  ASSERT_STREQ_LEN(written, buff, "inf");

  written = LIBC_NAMESPACE::sprintf(buff, "%LF", -ld_inf);
  ASSERT_STREQ_LEN(written, buff, "-INF");

  written = LIBC_NAMESPACE::sprintf(buff, "%Lf", ld_nan);
  ASSERT_STREQ_LEN(written, buff, "nan");

// Some float128 systems (specifically the ones used for aarch64 buildbots)
// don't respect signs for long double NaNs.
#if defined(LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80) ||                          \
    defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT64)
  written = LIBC_NAMESPACE::sprintf(buff, "%LF", -ld_nan);
  ASSERT_STREQ_LEN(written, buff, "-NAN");
#endif

  // Min Width Tests.

  written = LIBC_NAMESPACE::sprintf(buff, "%15f", 1.0);
  ASSERT_STREQ_LEN(written, buff, "       1.000000");

  written = LIBC_NAMESPACE::sprintf(buff, "%15f", -1.0);
  ASSERT_STREQ_LEN(written, buff, "      -1.000000");

  written = LIBC_NAMESPACE::sprintf(buff, "%15f", 1.0e5);
  ASSERT_STREQ_LEN(written, buff, "  100000.000000");

  written = LIBC_NAMESPACE::sprintf(buff, "%15f", -1.0e5);
  ASSERT_STREQ_LEN(written, buff, " -100000.000000");

  written = LIBC_NAMESPACE::sprintf(buff, "%10f", 1.0e5);
  ASSERT_STREQ_LEN(written, buff, "100000.000000");

  // Precision Tests.

  written = LIBC_NAMESPACE::sprintf(buff, "%.1f", 1.0);
  ASSERT_STREQ_LEN(written, buff, "1.0");

  written = LIBC_NAMESPACE::sprintf(buff, "%.1f", 0.0);
  ASSERT_STREQ_LEN(written, buff, "0.0");

  written = LIBC_NAMESPACE::sprintf(buff, "%.0f", 0.0);
  ASSERT_STREQ_LEN(written, buff, "0");

  written = LIBC_NAMESPACE::sprintf(buff, "%.1f", 0.1);
  ASSERT_STREQ_LEN(written, buff, "0.1");

  written = LIBC_NAMESPACE::sprintf(buff, "%.1f", 1.09);
  ASSERT_STREQ_LEN(written, buff, "1.1");

  written = LIBC_NAMESPACE::sprintf(buff, "%.1f", 1.04);
  ASSERT_STREQ_LEN(written, buff, "1.0");

  written = LIBC_NAMESPACE::sprintf(buff, "%.1f", 1.19);
  ASSERT_STREQ_LEN(written, buff, "1.2");

  written = LIBC_NAMESPACE::sprintf(buff, "%.1f", 1.99);
  ASSERT_STREQ_LEN(written, buff, "2.0");

  written = LIBC_NAMESPACE::sprintf(buff, "%.1f", 9.99);
  ASSERT_STREQ_LEN(written, buff, "10.0");

  written = LIBC_NAMESPACE::sprintf(buff, "%.2f", 9999999999999.99);
  ASSERT_STREQ_LEN(written, buff, "9999999999999.99");

  written = LIBC_NAMESPACE::sprintf(buff, "%.1f", 9999999999999.99);
  ASSERT_STREQ_LEN(written, buff, "10000000000000.0");

  written = LIBC_NAMESPACE::sprintf(buff, "%.5f", 1.25);
  ASSERT_STREQ_LEN(written, buff, "1.25000");

  written = LIBC_NAMESPACE::sprintf(buff, "%.0f", 1.25);
  ASSERT_STREQ_LEN(written, buff, "1");

  written = LIBC_NAMESPACE::sprintf(buff, "%.0f", 1.75);
  ASSERT_STREQ_LEN(written, buff, "2");

  written = LIBC_NAMESPACE::sprintf(buff, "%.20f", 1.234e-10);
  ASSERT_STREQ_LEN(written, buff, "0.00000000012340000000");

  written = LIBC_NAMESPACE::sprintf(buff, "%.2f", -9.99);
  ASSERT_STREQ_LEN(written, buff, "-9.99");

  written = LIBC_NAMESPACE::sprintf(buff, "%.1f", -9.99);
  ASSERT_STREQ_LEN(written, buff, "-10.0");

  written = LIBC_NAMESPACE::sprintf(buff, "%.5f", 0.0);
  ASSERT_STREQ_LEN(written, buff, "0.00000");

  written = LIBC_NAMESPACE::sprintf(buff, "%.5f", 1.008);
  ASSERT_STREQ_LEN(written, buff, "1.00800");

  written = LIBC_NAMESPACE::sprintf(buff, "%.5f", 1.008e3);
  ASSERT_STREQ_LEN(written, buff, "1008.00000");

  // Found with the help of Fred Tydeman's tbin2dec test.
  written = LIBC_NAMESPACE::sprintf(buff, "%.1f", 0x1.1000000000006p+3);
  ASSERT_STREQ_LEN(written, buff, "8.5");

  written = LIBC_NAMESPACE::sprintf(buff, "%.0f", 0x1.1000000000006p+3);
  ASSERT_STREQ_LEN(written, buff, "9");

  // Most of these tests are checking rounding behavior when the precision is
  // set. As an example, %.9f has a precision of 9, meaning it should be rounded
  // to 9 digits after the decimal point. In this case, that means that it
  // should be rounded up. Many of these tests have precisions divisible by 9
  // since when printing the floating point numbers are broken up into "blocks"
  // of 9 digits. They often also have a 5 after the end of what's printed,
  // since in round to nearest mode, that requires checking additional digits.
  written = LIBC_NAMESPACE::sprintf(buff, "%.9f", 1.9999999999999514);
  ASSERT_STREQ_LEN(written, buff, "2.000000000");

  // The number continues after the literal because floating point numbers can't
  // represent every value. The printed value is the closest value a double can
  // represent, rounded to the requested precision.
  written = LIBC_NAMESPACE::sprintf(buff, "%.238f", 1.131959884853339E-72);
  ASSERT_STREQ_LEN(
      written, buff,
      "0."
      "000000000000000000000000000000000000000000000000000000000000000000000001"
      "131959884853339045938639911360973972585316399767392273697826861241937664"
      "824105639342441431495119762431744054912109728706985341609159156917030486"
      "5110665559768676757812");

  written = LIBC_NAMESPACE::sprintf(buff, "%.36f", 9.9e-77);
  ASSERT_STREQ_LEN(written, buff, "0.000000000000000000000000000000000000");

  written =
      LIBC_NAMESPACE::sprintf(big_buff, "%.1071f", 2.0226568751604562E-314);
  ASSERT_STREQ_LEN(
      written, big_buff,
      "0."
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000020226568751604561683387695750739190248658016786"
      "876938365740768295004457513021760887468117675879956193821375945376632621"
      "367998639317487303530427946024002091961988296562516210434394107910027236"
      "308233439098296717697919471698168200340836487924061502604112643734560622"
      "258525943451473162532620033398739382796482175564084902819878893430369431"
      "907237673154867595954110791891883281880339550955455702452422857027182100"
      "606009588295886640782228837851739241290179512817803196347460636150182981"
      "085084829941917048152725177119574542042352896161225179181967347829576272"
      "242480201291872969114441104973910102402751449901108484914924879541248714"
      "939096548775588293353689592872854495101242645279589976452453829724479805"
      "750016448075109469332839157162950982637994457036256790161132812");

  // If no precision is specified it defaults to 6 for %f.
  written = LIBC_NAMESPACE::sprintf(buff, "%f", 2325885.4901960781);
  ASSERT_STREQ_LEN(written, buff, "2325885.490196");

  // Subnormal Precision Tests

  written = LIBC_NAMESPACE::sprintf(buff, "%.310f", 0x1.0p-1022);
  ASSERT_STREQ_LEN(
      written, buff,
      "0."
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "0000000000000000000223");

  written = LIBC_NAMESPACE::sprintf(buff, "%.310f", 0x1.0p-1023);
  ASSERT_STREQ_LEN(
      written, buff,
      "0."
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "0000000000000000000111");

  written = LIBC_NAMESPACE::sprintf(buff, "%.315f", 9.99999e-310);
  ASSERT_STREQ_LEN(
      written, buff,
      "0."
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000999999");

  written = LIBC_NAMESPACE::sprintf(buff, "%.314f", 9.99999e-310);
  ASSERT_STREQ_LEN(
      written, buff,
      "0."
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "00000000000000000000100000");

  written = LIBC_NAMESPACE::sprintf(buff, "%.330f", 0x1.0p-1074);
  ASSERT_STREQ_LEN(
      written, buff,
      "0."
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000004940656");

  /*
    written = LIBC_NAMESPACE::sprintf(buff, "%.1La", 0.1L);
  #if defined(LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80)
    ASSERT_STREQ_LEN(written, buff, "0xc.dp-7");
  #elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT64)
    ASSERT_STREQ_LEN(written, buff, "0x1.ap-4");
  #elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT128)
    ASSERT_STREQ_LEN(written, buff, "0x1.ap-4");
  #endif

    written = LIBC_NAMESPACE::sprintf(buff, "%.1La",
  0xf.fffffffffffffffp16380L); #if
  defined(LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80) ASSERT_STREQ_LEN(written, buff,
  "0x1.0p+16384"); #elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT64)
    ASSERT_STREQ_LEN(written, buff, "inf");
  #elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT128)
    ASSERT_STREQ_LEN(written, buff, "0x2.0p+16383");
  #endif
  */

  // Rounding Mode Tests.

  if (ForceRoundingMode r(RoundingMode::Nearest); r.success) {
    written = LIBC_NAMESPACE::sprintf(buff, "%.1f", 1.75);
    ASSERT_STREQ_LEN(written, buff, "1.8");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1f", 1.25);
    ASSERT_STREQ_LEN(written, buff, "1.2");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1f", 1.125);
    ASSERT_STREQ_LEN(written, buff, "1.1");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1f", 1.625);
    ASSERT_STREQ_LEN(written, buff, "1.6");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1f", 1.375);
    ASSERT_STREQ_LEN(written, buff, "1.4");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1f", 1.875);
    ASSERT_STREQ_LEN(written, buff, "1.9");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1f", -1.75);
    ASSERT_STREQ_LEN(written, buff, "-1.8");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1f", -1.25);
    ASSERT_STREQ_LEN(written, buff, "-1.2");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1f", -1.125);
    ASSERT_STREQ_LEN(written, buff, "-1.1");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1f", -1.625);
    ASSERT_STREQ_LEN(written, buff, "-1.6");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1f", -1.375);
    ASSERT_STREQ_LEN(written, buff, "-1.4");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1f", -1.875);
    ASSERT_STREQ_LEN(written, buff, "-1.9");
  }

  if (ForceRoundingMode r(RoundingMode::Upward); r.success) {
    written = LIBC_NAMESPACE::sprintf(buff, "%.1f", 1.75);
    ASSERT_STREQ_LEN(written, buff, "1.8");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1f", 1.25);
    ASSERT_STREQ_LEN(written, buff, "1.3");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1f", 1.125);
    ASSERT_STREQ_LEN(written, buff, "1.2");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1f", 1.625);
    ASSERT_STREQ_LEN(written, buff, "1.7");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1f", 1.375);
    ASSERT_STREQ_LEN(written, buff, "1.4");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1f", 1.875);
    ASSERT_STREQ_LEN(written, buff, "1.9");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1f", -1.75);
    ASSERT_STREQ_LEN(written, buff, "-1.7");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1f", -1.25);
    ASSERT_STREQ_LEN(written, buff, "-1.2");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1f", -1.125);
    ASSERT_STREQ_LEN(written, buff, "-1.1");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1f", -1.625);
    ASSERT_STREQ_LEN(written, buff, "-1.6");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1f", -1.375);
    ASSERT_STREQ_LEN(written, buff, "-1.3");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1f", -1.875);
    ASSERT_STREQ_LEN(written, buff, "-1.8");
  }

  if (ForceRoundingMode r(RoundingMode::Downward); r.success) {
    written = LIBC_NAMESPACE::sprintf(buff, "%.1f", 1.75);
    ASSERT_STREQ_LEN(written, buff, "1.7");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1f", 1.25);
    ASSERT_STREQ_LEN(written, buff, "1.2");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1f", 1.125);
    ASSERT_STREQ_LEN(written, buff, "1.1");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1f", 1.625);
    ASSERT_STREQ_LEN(written, buff, "1.6");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1f", 1.375);
    ASSERT_STREQ_LEN(written, buff, "1.3");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1f", 1.875);
    ASSERT_STREQ_LEN(written, buff, "1.8");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1f", -1.75);
    ASSERT_STREQ_LEN(written, buff, "-1.8");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1f", -1.25);
    ASSERT_STREQ_LEN(written, buff, "-1.3");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1f", -1.125);
    ASSERT_STREQ_LEN(written, buff, "-1.2");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1f", -1.625);
    ASSERT_STREQ_LEN(written, buff, "-1.7");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1f", -1.375);
    ASSERT_STREQ_LEN(written, buff, "-1.4");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1f", -1.875);
    ASSERT_STREQ_LEN(written, buff, "-1.9");
  }

  if (ForceRoundingMode r(RoundingMode::TowardZero); r.success) {
    written = LIBC_NAMESPACE::sprintf(buff, "%.1f", 1.75);
    ASSERT_STREQ_LEN(written, buff, "1.7");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1f", 1.25);
    ASSERT_STREQ_LEN(written, buff, "1.2");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1f", 1.125);
    ASSERT_STREQ_LEN(written, buff, "1.1");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1f", 1.625);
    ASSERT_STREQ_LEN(written, buff, "1.6");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1f", 1.375);
    ASSERT_STREQ_LEN(written, buff, "1.3");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1f", 1.875);
    ASSERT_STREQ_LEN(written, buff, "1.8");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1f", -1.75);
    ASSERT_STREQ_LEN(written, buff, "-1.7");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1f", -1.25);
    ASSERT_STREQ_LEN(written, buff, "-1.2");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1f", -1.125);
    ASSERT_STREQ_LEN(written, buff, "-1.1");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1f", -1.625);
    ASSERT_STREQ_LEN(written, buff, "-1.6");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1f", -1.375);
    ASSERT_STREQ_LEN(written, buff, "-1.3");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1f", -1.875);
    ASSERT_STREQ_LEN(written, buff, "-1.8");
  }

  // Flag Tests.
  written = LIBC_NAMESPACE::sprintf(buff, "%+f", 1.0);
  ASSERT_STREQ_LEN(written, buff, "+1.000000");

  written = LIBC_NAMESPACE::sprintf(buff, "%+f", -1.0);
  ASSERT_STREQ_LEN(written, buff, "-1.000000");

  written = LIBC_NAMESPACE::sprintf(buff, "% f", 1.0);
  ASSERT_STREQ_LEN(written, buff, " 1.000000");

  written = LIBC_NAMESPACE::sprintf(buff, "% f", -1.0);
  ASSERT_STREQ_LEN(written, buff, "-1.000000");

  written = LIBC_NAMESPACE::sprintf(buff, "%-10f", 1.5);
  ASSERT_STREQ_LEN(written, buff, "1.500000  ");

  written = LIBC_NAMESPACE::sprintf(buff, "%#.f", 1.0);
  ASSERT_STREQ_LEN(written, buff, "1.");

  written = LIBC_NAMESPACE::sprintf(buff, "%#.0f", 1.5);
  ASSERT_STREQ_LEN(written, buff, "2.");

  written = LIBC_NAMESPACE::sprintf(buff, "%010f", 1.5);
  ASSERT_STREQ_LEN(written, buff, "001.500000");

  written = LIBC_NAMESPACE::sprintf(buff, "%010f", -1.5);
  ASSERT_STREQ_LEN(written, buff, "-01.500000");

  written = LIBC_NAMESPACE::sprintf(buff, "%+- #0f", 0.0);
  ASSERT_STREQ_LEN(written, buff, "+0.000000");

  // Combined Tests.

  written = LIBC_NAMESPACE::sprintf(buff, "%10.2f", 9.99);
  ASSERT_STREQ_LEN(written, buff, "      9.99");

  written = LIBC_NAMESPACE::sprintf(buff, "%5.1f", 9.99);
  ASSERT_STREQ_LEN(written, buff, " 10.0");

  written = LIBC_NAMESPACE::sprintf(buff, "%-10.2f", 9.99);
  ASSERT_STREQ_LEN(written, buff, "9.99      ");

  written = LIBC_NAMESPACE::sprintf(buff, "%-5.1f", 9.99);
  ASSERT_STREQ_LEN(written, buff, "10.0 ");

  written = LIBC_NAMESPACE::sprintf(buff, "%-5.1f", 1.0e-50);
  ASSERT_STREQ_LEN(written, buff, "0.0  ");

  written = LIBC_NAMESPACE::sprintf(buff, "%30f", 1234567890123456789.0);
  ASSERT_STREQ_LEN(written, buff, "    1234567890123456768.000000");

  written = LIBC_NAMESPACE::sprintf(buff, "%-30f", 1234567890123456789.0);
  ASSERT_STREQ_LEN(written, buff, "1234567890123456768.000000    ");

  written = LIBC_NAMESPACE::sprintf(buff, "%20.2f", 9999999999999.99);
  ASSERT_STREQ_LEN(written, buff, "    9999999999999.99");

  written = LIBC_NAMESPACE::sprintf(buff, "%20.1f", 9999999999999.99);
  ASSERT_STREQ_LEN(written, buff, "    10000000000000.0");

  written = LIBC_NAMESPACE::sprintf(buff, "%12.3f %-12.3f", 0.1, 256.0);
  ASSERT_STREQ_LEN(written, buff, "       0.100 256.000     ");

  written = LIBC_NAMESPACE::sprintf(buff, "%+-#12.3f % 012.3f", 0.1256, 1256.0);
  ASSERT_STREQ_LEN(written, buff, "+0.126        0001256.000");
}

// The long double tests are separated so that their performance can be directly
// measured.
TEST_F(LlvmLibcSPrintfTest, FloatDecimalLongDoubleConv) {
  ForceRoundingMode r(RoundingMode::Nearest);

  // Length Modifier Tests.

  // TODO(michaelrj): Add tests for LIBC_TYPES_LONG_DOUBLE_IS_FLOAT64 and 128
  // bit long double systems.
  // TODO(michaelrj): Fix the tests to only depend on the digits the long double
  // is accurate for.

  written = LIBC_NAMESPACE::sprintf(buff, "%Lf", 1.0L);
  ASSERT_STREQ_LEN(written, buff, "1.000000");

  written = LIBC_NAMESPACE::sprintf(buff, "%.Lf", -2.5L);
  ASSERT_STREQ_LEN(written, buff, "-2");

#if defined(LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80)

  written = LIBC_NAMESPACE::sprintf(buff, "%Lf", 1e100L);
  ASSERT_STREQ_LEN(written, buff,
                   "99999999999999999996693535322073426194986990198284960792713"
                   "91541752018669482644324418977840117055488.000000");

  written = LIBC_NAMESPACE::sprintf(buff, "%Lf", 0xd.96ed1192687859ap-24L);
  ASSERT_STREQ_LEN(written, buff, "0.000001");

  written = LIBC_NAMESPACE::sprintf(buff, "%Lf", 10000000000000000.25L);
  ASSERT_STREQ_LEN(written, buff, "10000000000000000.250000");

  written = LIBC_NAMESPACE::sprintf(buff, "%.510Lf", 0x8p-503L);
  ASSERT_STREQ_LEN(
      written, buff,
      "0."
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000305493636349960468205197939321361769978940274057232666389361390928"
      "129162652472045770185723510801522825687515269359046715531785342780428396"
      "973513311420091788963072442053377285222203558881953188370081650866793017"
      "948791366338993705251636497892270212003524508209121908744820211960149463"
      "721109340307985507678283651836204093399373959982767701148986816406250000"
      "000000");

  written = LIBC_NAMESPACE::sprintf(buff, "%.500Lf", -4327677766926336.0L);
  ASSERT_STREQ_LEN(
      written, buff,
      "-4327677766926336."
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "00000000000000000000000000000000000000000000000000000000000000000000");

  char big_buff[10000]; // Used for extremely wide numbers.

  written = LIBC_NAMESPACE::sprintf(big_buff, "%Lf", 1e1000L);
  ASSERT_STREQ_LEN(
      written, big_buff,
      "999999999999999999973107317669562353428234857594552594925899449376328728"
      "202461036775511405481186963193066642191664822065529414252060696836533522"
      "387143501724276282079456797058697369889056407118642873669166717313763499"
      "277025985141177344925615052465165938514140943010597323750202561187880136"
      "174810574553749194614479541820148407958204853833697063267336294787191005"
      "628217462261955103745349844675732989944229689277833828743730290177882029"
      "042613704915899149603539993716885598351951895974316347947147507970269673"
      "097709017164643598452451201499004104341931127294141495501309305995449742"
      "273419524803597130450457553871345958049837885085168840317195672271085085"
      "950520957945970913451088104971436093671776829538796532762184174216651692"
      "640931965387852083906784898823494867055070322768919156031682291829761007"
      "101483799978382119231551218582499361996919560548090784230386907125151658"
      "086767207295524036170321059257942621398084478974000973622199163292708506"
      "2431457550909271560663602154947063707982236377366647567795879936."
      "000000");

  written = LIBC_NAMESPACE::sprintf(big_buff, "%Lf", 1e4900L);
  ASSERT_STREQ_LEN(
      written, big_buff,
      "100000000000000000002708312230690349833224052504078834346502930111959028"
      "517260692666637048230414374897655201843766090626319971729765251179632020"
      "313912652522792711197087872698264530532442630109549129842736280196919130"
      "242615101228133188193853826983121366159061148351354364472807590931218045"
      "387490935930967150336231085015126034696883068553581691802388371635128003"
      "615577299166097675723780877126495909902479233742826339471026068806070433"
      "075629449530819183550315434973800271862658869400009022028602967197463980"
      "126881829804282202449930132940824361207087494829502385835258094836304011"
      "876250359661206802659650567866176246063987902366800491980400341950657151"
      "370854446585517805253310195469184699955519312761482572080479702840420595"
      "377369017651259376039167277822106875560385309101650382998482652792335482"
      "865443482342801545877390859444282105890147577937366066315975231014810320"
      "888482059656248277607763361589359794524314002443575149260630989130103550"
      "443177966380769341050735632338583912575890190136462629316287947355057647"
      "111088565611192544631519843618778618820046304429723908484879583579178075"
      "456701368334212923379389029311286386996015804122917416008806233549005183"
      "152461084266176543129004016414959261473645240454289630182591200574019087"
      "358223489767381636349719510715487188747217311279465814538495924567014916"
      "238565628036285599497236493491668884212847699052761266207598941300449276"
      "447201387520841811835583254242213093566548778954711633721122784159793843"
      "766802019309395771984693609426401362800013936338891483689127845928572536"
      "790651156184721483511507878883282891696900630100211914227950790472211403"
      "392549466062537498185758854079775888444518306635752468713312357556380082"
      "275500658967283696421824354930077523691855699312544373220921962817907078"
      "445538421941800259027487429330768616490865438859612697367766323925013940"
      "918384858952407145253573823848733994146335416209309233074165707437420756"
      "438833918763109580759409985573826485055208965115587885226774453455112406"
      "581351429640282227888764449360534584421929291565334894907337572527922691"
      "473242328379737396430908523008687037407295838014450772162091496534584696"
      "605157436893236842602956298545594095307060870397506421786236892553632163"
      "491468601982681381011940409602294892199042638682530687578982576819839451"
      "907594697546439533559153604700750696252355362322662219852740143212566818"
      "745528402265116534684566273868361460640280523251242059850044328669692159"
      "629900374576027104298177006629276014371540945261309319363704125592775129"
      "543526908667388673739382491147471395192495459318806593271282662311169392"
      "196897003517840025298267505925987901751541005546610016067658227181318892"
      "914686508281007582655667597441346214499847364272258631922040641860333431"
      "409838623713258383681350233064164940590695888300919626215847587544298023"
      "636416943680102708406086295669759876682046839368574433996997648445207805"
      "615784339667691231286807666753972942872019850432610318031627872612657513"
      "588188267160616660825719678199868371370527508463011236193719286066916786"
      "169956541349011494927225747024994619057884118692213564790598702879596058"
      "672338334720925179141906809470606964896245458600635183723159228561689808"
      "246141482736625197373238197777325580142168245885279594913851700941789475"
      "252421784152262567254611571822468808675893407728003047921107885664474662"
      "930921581384003950729114103689170603748380178682003976896397305836815761"
      "717676338115866650889936516794601457549097578905329423919798362140648664"
      "569177147076571576101649257502509463877402424847669830852345415301684820"
      "395813946416649808062227494112874521812750160935760825922220707178083076"
      "380203450993589198835885505461509442443773367592842795410339065860781804"
      "024975272228687688301824830333940416256885455008512598774611538878683158"
      "183931461086893832255176926531299425504132104728730288984598001187854507"
      "900417184206801359847651992484444933900133130832052346600926424167009902"
      "829803553087005800387704758687923428053612864451456596148162238935900033"
      "917094683141205188616000211702577553792389670853917118547527592495253773"
      "028135298405566315903922235989614934474805789300370437580494193066066314"
      "056627605207631392651010580925826419831250810981343093764403877594495896"
      "516881097415880926429607388979497471571321217205535961262051641426436441"
      "668989765107456413733909427384182109285933511623871034309722437967253289"
      "084018145083721513211807496392673952789642893241520398827805325610653506"
      "029060153153064455898648607959013571280930834475689835845791849456112104"
      "462337569019001580859906425911782967213265389744605395555069797947978230"
      "708108432086217134763779632408473684293543722127232658767439906910370146"
      "716836295909075482355827087389127370874842532825987593970846704144140471"
      "956027276735614286138656432085771988513977140957180090146798065497158947"
      "229765733489703157617307078835099906185890777007500964162371428641176460"
      "739074789794941408428328217107759915202650066155868439585510978709442590"
      "231934194956788626761834746430104077432547436359522462253411168467463134"
      "24896.000000");

  written =
      LIBC_NAMESPACE::sprintf(big_buff, "%Lf", 0xf.fffffffffffffffp+16380L);
  ASSERT_STREQ_LEN(
      written, big_buff,
      "118973149535723176502126385303097020516906332229462420044032373389173700"
      "552297072261641029033652888285354569780749557731442744315367028843419812"
      "557385374367867359320070697326320191591828296152436552951064679108661431"
      "179063216977883889613478656060039914875343321145491116008867984515486651"
      "285234014977303760000912547939396622315138362241783854274391783813871780"
      "588948754057516822634765923557697480511372564902088485522249479139937758"
      "502601177354918009979622602685950855888360815984690023564513234659447638"
      "493985927645628457966177293040780660922910271504608538808795932778162298"
      "682754783076808004015069494230341172895777710033571401055977524212405734"
      "700738625166011082837911962300846927720096515350020847447079244384854591"
      "288672300061908512647211195136146752763351956292759795725027800298079590"
      "419313960302147099703527646744553092202267965628099149823208332964124103"
      "850923918473478612192169721054348428704835340811304257300221642134891734"
      "717423480071488075100206439051723424765600472176809648610799494341570347"
      "632064355862420744350442438056613601760883747816538902780957697597728686"
      "007148702828795556714140463261583262360276289631617397848425448686060994"
      "827086796804807870251185893083854658422304090880599629459458620190376604"
      "844679092600222541053077590106576067134720012584640695703025713896098375"
      "799892695455305236856075868317922311363951946885088077187210470520395758"
      "748001314313144425494391994017575316933939236688185618912993172910425292"
      "123683515992232205099800167710278403536014082929639811512287776813570604"
      "578934353545169653956125404884644716978689321167108722908808277835051822"
      "885764606221873970285165508372099234948333443522898475123275372663606621"
      "390228126470623407535207172405866507951821730346378263135339370677490195"
      "019784169044182473806316282858685774143258116536404021840272491339332094"
      "921949842244273042701987304453662035026238695780468200360144729199712309"
      "553005720614186697485284685618651483271597448120312194675168637934309618"
      "961510733006555242148519520176285859509105183947250286387163249416761380"
      "499631979144187025430270675849519200883791516940158174004671147787720145"
      "964446117520405945350476472180797576111172084627363927960033967047003761"
      "337450955318415007379641260504792325166135484129188421134082301547330475"
      "406707281876350361733290800595189632520707167390454777712968226520622565"
      "143991937680440029238090311243791261477625596469422198137514696707944687"
      "035800439250765945161837981185939204954403611491531078225107269148697980"
      "924094677214272701240437718740921675661363493890045123235166814608932240"
      "069799317601780533819184998193300841098599393876029260139091141452600372"
      "028487213241195542428210183120421610446740462163533690058366460659115629"
      "876474552506814500393294140413149540067760295100596225302282300363147382"
      "468105964844244132486457313743759509641616804802412935187620466813563687"
      "753281467553879887177183651289394719533506188500326760735438867336800207"
      "438784965701457609034985757124304510203873049485425670247933932280911052"
      "604153852899484920399109194612991249163328991799809438033787952209313146"
      "694614970593966415237594928589096048991612194498998638483702248667224914"
      "892467841020618336462741696957630763248023558797524525373703543388296086"
      "275342774001633343405508353704850737454481975472222897528108302089868263"
      "302028525992308416805453968791141829762998896457648276528750456285492426"
      "516521775079951625966922911497778896235667095662713848201819134832168799"
      "586365263762097828507009933729439678463987902491451422274252700636394232"
      "799848397673998715441855420156224415492665301451550468548925862027608576"
      "183712976335876121538256512963353814166394951655600026415918655485005705"
      "261143195291991880795452239464962763563017858089669222640623538289853586"
      "759599064700838568712381032959192649484625076899225841930548076362021508"
      "902214922052806984201835084058693849381549890944546197789302911357651677"
      "540623227829831403347327660395223160342282471752818181884430488092132193"
      "355086987339586127607367086665237555567580317149010847732009642431878007"
      "000879734603290627894355374356444885190719161645514115576193939969076741"
      "515640282654366402676009508752394550734155613586793306603174472092444651"
      "353236664764973540085196704077110364053815007348689179836404957060618953"
      "500508984091382686953509006678332447257871219660441528492484004185093281"
      "190896363417573989716659600075948780061916409485433875852065711654107226"
      "099628815012314437794400874930194474433078438899570184271000480830501217"
      "712356062289507626904285680004771889315808935851559386317665294808903126"
      "774702966254511086154895839508779675546413794489596052797520987481383976"
      "257859210575628440175934932416214833956535018919681138909184379573470326"
      "940634289008780584694035245347939808067427323629788710086717580253156130"
      "235606487870925986528841635097252953709111431720488774740553905400942537"
      "542411931794417513706468964386151771884986701034153254238591108962471088"
      "538580868883777725864856414593426212108664758848926003176234596076950884"
      "9149662444156604419552086811989770240.000000");

  written = LIBC_NAMESPACE::sprintf(big_buff, "%.10Lf", 1e-10L);
  ASSERT_STREQ_LEN(written, big_buff, "0.0000000001");

  written = LIBC_NAMESPACE::sprintf(big_buff, "%.7500Lf", 1e-4900L);
  ASSERT_STREQ_LEN(
      written, big_buff,
      "0."
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000099999999999999999996962764452956071352139203248614920751610856665084"
      "549214352477698417183862158583009348897567779527408501588132175167211539"
      "462139941448204886585901454195352527238724272760638086779284030512649793"
      "039219351187928723378036480041948464946018272171365770411701020666925613"
      "422460317465324758217878522666789603627480490870456508256359180089236338"
      "765625231186929290294207420828927406735690318849109129700396907705735097"
      "663944722727287361650042373203763784830198232253311807069225650324196304"
      "532045014970637489181357566354288111205943347410488298480279857453705249"
      "232862728556860184412369114663536200895729846877559808001004454634804626"
      "541455540260282018142615835686583304903486353937549394736905011798466731"
      "536563240053860118551127061960208467764243724656897127545613968909523389"
      "577188368809623987105800147797280462974804046545425080530020901531407223"
      "191237123282274818236437397994019915368657474589800678444589412286037789"
      "891525464936023205313685584525510094270344601331453730179416773626565262"
      "480345858564672442896904520146956686863172737711483866766404977719744767"
      "834324844875237277613991088218774564658513875732403456058414595576806383"
      "115554713240005982141397577420073082470139244845624915873825746771661332"
      "098677966580506186966978746832443976821987300902957597498388211921362869"
      "017846215557612829071692275292036211064515305528052919611691470945774714"
      "135516559501572279732350629089770249554808690411603894492333360300589658"
      "470898965370892774715815089075170720164713889237058574941489766701880158"
      "060081295483989540170337129032188818293132770882381428397119039835946745"
      "549356649433406617266370644136291924838857814675939156677910783740103207"
      "523299367093130816446415259371931925208362367989095199399211644084543790"
      "110432339056231037520216864358899218874658268610955002763260912337688947"
      "822453100821038299301092582962825965939081817836419126254832772002214908"
      "085575905761843610944187009818156363893015929300295112598059949496854566"
      "638748010633726861510500653821408135845840123073754133549077708843800674"
      "328440913743105608636458354618912183716456158809545183074062249922212944"
      "249667793845728355381309084891765979111348980470647082269921872595470473"
      "719354467594516320911964549508538492057120740224559944452120552719041944"
      "961475548547884309626382512432626380881023756568143060204097921571153170"
      "723817845809196253498326358439807445210362177680590181657555380795450462"
      "223805222580359379367452693270553602179122419370586308101820559214330382"
      "570449525088342437216896462077260223998756027453411520977536701491759878"
      "422771447006016890777855573925295187921971811871399320142563330377888532"
      "179817332113");
#endif // LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80
}

TEST_F(LlvmLibcSPrintfTest, FloatExponentConv) {
  ForceRoundingMode r(RoundingMode::Nearest);
  double inf = LIBC_NAMESPACE::fputil::FPBits<double>::inf().get_val();
  double nan = LIBC_NAMESPACE::fputil::FPBits<double>::quiet_nan().get_val();

  written = LIBC_NAMESPACE::sprintf(buff, "%e", 1.0);
  ASSERT_STREQ_LEN(written, buff, "1.000000e+00");

  written = LIBC_NAMESPACE::sprintf(buff, "%E", -1.0);
  ASSERT_STREQ_LEN(written, buff, "-1.000000E+00");

  written = LIBC_NAMESPACE::sprintf(buff, "%e", -1.234567);
  ASSERT_STREQ_LEN(written, buff, "-1.234567e+00");

  written = LIBC_NAMESPACE::sprintf(buff, "%e", 0.0);
  ASSERT_STREQ_LEN(written, buff, "0.000000e+00");

  written = LIBC_NAMESPACE::sprintf(buff, "%e", 1.5);
  ASSERT_STREQ_LEN(written, buff, "1.500000e+00");

  written = LIBC_NAMESPACE::sprintf(buff, "%e", 1e300);
  ASSERT_STREQ_LEN(written, buff, "1.000000e+300");

  written = LIBC_NAMESPACE::sprintf(buff, "%e", 0.1);
  ASSERT_STREQ_LEN(written, buff, "1.000000e-01");

  written = LIBC_NAMESPACE::sprintf(buff, "%e", 0.001);
  ASSERT_STREQ_LEN(written, buff, "1.000000e-03");

  written = LIBC_NAMESPACE::sprintf(buff, "%e", 0.00001);
  ASSERT_STREQ_LEN(written, buff, "1.000000e-05");

  written = LIBC_NAMESPACE::sprintf(buff, "%e", 0.0000001);
  ASSERT_STREQ_LEN(written, buff, "1.000000e-07");

  written = LIBC_NAMESPACE::sprintf(buff, "%e", 0.000000001);
  ASSERT_STREQ_LEN(written, buff, "1.000000e-09");

  written = LIBC_NAMESPACE::sprintf(buff, "%e", 1.0e-20);
  ASSERT_STREQ_LEN(written, buff, "1.000000e-20");

  written = LIBC_NAMESPACE::sprintf(buff, "%e", 1234567890123456789.0);
  ASSERT_STREQ_LEN(written, buff, "1.234568e+18");

  written = LIBC_NAMESPACE::sprintf(buff, "%e", 9999999000000.00);
  ASSERT_STREQ_LEN(written, buff, "9.999999e+12");

  // Simple Subnormal Tests.

  written = LIBC_NAMESPACE::sprintf(buff, "%e", 0x1.0p-1027);
  ASSERT_STREQ_LEN(written, buff, "6.953356e-310");

  written = LIBC_NAMESPACE::sprintf(buff, "%e", 0x1.0p-1074);
  ASSERT_STREQ_LEN(written, buff, "4.940656e-324");

  // Inf/Nan Tests.

  written = LIBC_NAMESPACE::sprintf(buff, "%e", inf);
  ASSERT_STREQ_LEN(written, buff, "inf");

  written = LIBC_NAMESPACE::sprintf(buff, "%E", -inf);
  ASSERT_STREQ_LEN(written, buff, "-INF");

  written = LIBC_NAMESPACE::sprintf(buff, "%e", nan);
  ASSERT_STREQ_LEN(written, buff, "nan");

  written = LIBC_NAMESPACE::sprintf(buff, "%E", -nan);
  ASSERT_STREQ_LEN(written, buff, "-NAN");

  // Min Width Tests.

  written = LIBC_NAMESPACE::sprintf(buff, "%15e", 1.0);
  ASSERT_STREQ_LEN(written, buff, "   1.000000e+00");

  written = LIBC_NAMESPACE::sprintf(buff, "%15e", -1.0);
  ASSERT_STREQ_LEN(written, buff, "  -1.000000e+00");

  written = LIBC_NAMESPACE::sprintf(buff, "%15e", 1.0e5);
  ASSERT_STREQ_LEN(written, buff, "   1.000000e+05");

  written = LIBC_NAMESPACE::sprintf(buff, "%15e", -1.0e5);
  ASSERT_STREQ_LEN(written, buff, "  -1.000000e+05");

  written = LIBC_NAMESPACE::sprintf(buff, "%10e", 1.0e-5);
  ASSERT_STREQ_LEN(written, buff, "1.000000e-05");

  // Precision Tests.

  written = LIBC_NAMESPACE::sprintf(buff, "%.1e", 1.0);
  ASSERT_STREQ_LEN(written, buff, "1.0e+00");

  written = LIBC_NAMESPACE::sprintf(buff, "%.1e", 0.0);
  ASSERT_STREQ_LEN(written, buff, "0.0e+00");

  written = LIBC_NAMESPACE::sprintf(buff, "%.0e", 0.0);
  ASSERT_STREQ_LEN(written, buff, "0e+00");

  written = LIBC_NAMESPACE::sprintf(buff, "%.1e", 0.1);
  ASSERT_STREQ_LEN(written, buff, "1.0e-01");

  written = LIBC_NAMESPACE::sprintf(buff, "%.1e", 1.09);
  ASSERT_STREQ_LEN(written, buff, "1.1e+00");

  written = LIBC_NAMESPACE::sprintf(buff, "%.1e", 1.04);
  ASSERT_STREQ_LEN(written, buff, "1.0e+00");

  written = LIBC_NAMESPACE::sprintf(buff, "%.1e", 1.19);
  ASSERT_STREQ_LEN(written, buff, "1.2e+00");

  written = LIBC_NAMESPACE::sprintf(buff, "%.1e", 1.99);
  ASSERT_STREQ_LEN(written, buff, "2.0e+00");

  written = LIBC_NAMESPACE::sprintf(buff, "%.1e", 9.99);
  ASSERT_STREQ_LEN(written, buff, "1.0e+01");

  written = LIBC_NAMESPACE::sprintf(buff, "%.2e", 99.9);
  ASSERT_STREQ_LEN(written, buff, "9.99e+01");

  written = LIBC_NAMESPACE::sprintf(buff, "%.1e", 99.9);
  ASSERT_STREQ_LEN(written, buff, "1.0e+02");

  written = LIBC_NAMESPACE::sprintf(buff, "%.5e", 1.25);
  ASSERT_STREQ_LEN(written, buff, "1.25000e+00");

  written = LIBC_NAMESPACE::sprintf(buff, "%.0e", 1.25);
  ASSERT_STREQ_LEN(written, buff, "1e+00");

  written = LIBC_NAMESPACE::sprintf(buff, "%.0e", 1.75);
  ASSERT_STREQ_LEN(written, buff, "2e+00");

  written = LIBC_NAMESPACE::sprintf(buff, "%.20e", 1.234e-10);
  ASSERT_STREQ_LEN(written, buff, "1.23400000000000008140e-10");

  written = LIBC_NAMESPACE::sprintf(buff, "%.2e", -9.99);
  ASSERT_STREQ_LEN(written, buff, "-9.99e+00");

  written = LIBC_NAMESPACE::sprintf(buff, "%.1e", -9.99);
  ASSERT_STREQ_LEN(written, buff, "-1.0e+01");

  written = LIBC_NAMESPACE::sprintf(buff, "%.5e", 0.0);
  ASSERT_STREQ_LEN(written, buff, "0.00000e+00");

  written = LIBC_NAMESPACE::sprintf(buff, "%.5e", 1.008);
  ASSERT_STREQ_LEN(written, buff, "1.00800e+00");

  written = LIBC_NAMESPACE::sprintf(buff, "%.5e", 1.008e3);
  ASSERT_STREQ_LEN(written, buff, "1.00800e+03");

  // These tests also focus on rounding. Almost all of them have a 5 right after
  // the printed string (e.g. 9.5 with precision 0 prints 0 digits after the
  // decimal point). This is again because rounding a number with a 5 after the
  // printed section means that more digits have to be checked to determine if
  // this should be rounded up (if there are non-zero digits after the 5) or to
  // even (if the 5 is the last non-zero digit). Additionally, the algorithm for
  // checking if a number is all 0s after the decimal point may not work since
  // the decimal point moves in this representation.
  written = LIBC_NAMESPACE::sprintf(buff, "%.0e", 2.5812229360061737E+200);
  ASSERT_STREQ_LEN(written, buff, "3e+200");

  written = LIBC_NAMESPACE::sprintf(buff, "%.1e", 9.059E+200);
  ASSERT_STREQ_LEN(written, buff, "9.1e+200");

  written = LIBC_NAMESPACE::sprintf(buff, "%.0e", 9.059E+200);
  ASSERT_STREQ_LEN(written, buff, "9e+200");

  written = LIBC_NAMESPACE::sprintf(buff, "%.166e", 1.131959884853339E-72);
  ASSERT_STREQ_LEN(written, buff,
                   "1."
                   "13195988485333904593863991136097397258531639976739227369782"
                   "68612419376648241056393424414314951197624317440549121097287"
                   "069853416091591569170304865110665559768676757812e-72");

  written = LIBC_NAMESPACE::sprintf(buff, "%.0e", 9.5);
  ASSERT_STREQ_LEN(written, buff, "1e+01");

  written = LIBC_NAMESPACE::sprintf(buff, "%.10e", 1.9999999999890936);
  ASSERT_STREQ_LEN(written, buff, "2.0000000000e+00");

  written = LIBC_NAMESPACE::sprintf(buff, "%.1e", 745362143563.03894);
  ASSERT_STREQ_LEN(written, buff, "7.5e+11");

  written = LIBC_NAMESPACE::sprintf(buff, "%.0e", 45181042688.0);
  ASSERT_STREQ_LEN(written, buff, "5e+10");

  written = LIBC_NAMESPACE::sprintf(buff, "%.35e", 1.3752441369139243);
  ASSERT_STREQ_LEN(written, buff, "1.37524413691392433101157166674965993e+00");

  // Subnormal Precision Tests

  written = LIBC_NAMESPACE::sprintf(buff, "%.310e", 0x1.0p-1022);
  ASSERT_STREQ_LEN(
      written, buff,
      "2."
      "225073858507201383090232717332404064219215980462331830553327416887204434"
      "813918195854283159012511020564067339731035811005152434161553460108856012"
      "385377718821130777993532002330479610147442583636071921565046942503734208"
      "375250806650616658158948720491179968591639648500635908770118304874799780"
      "8877537499494515804516e-308");

  written = LIBC_NAMESPACE::sprintf(buff, "%.30e", 0x1.0p-1022);
  ASSERT_STREQ_LEN(written, buff, "2.225073858507201383090232717332e-308");

  written = LIBC_NAMESPACE::sprintf(buff, "%.310e", 0x1.0p-1023);
  ASSERT_STREQ_LEN(
      written, buff,
      "1."
      "112536929253600691545116358666202032109607990231165915276663708443602217"
      "406959097927141579506255510282033669865517905502576217080776730054428006"
      "192688859410565388996766001165239805073721291818035960782523471251867104"
      "187625403325308329079474360245589984295819824250317954385059152437399890"
      "4438768749747257902258e-308");

  written = LIBC_NAMESPACE::sprintf(buff, "%.6e", 9.99999e-310);
  ASSERT_STREQ_LEN(written, buff, "9.999990e-310");

  written = LIBC_NAMESPACE::sprintf(buff, "%.5e", 9.99999e-310);
  ASSERT_STREQ_LEN(written, buff, "9.99999e-310");

  written = LIBC_NAMESPACE::sprintf(buff, "%.4e", 9.99999e-310);
  ASSERT_STREQ_LEN(written, buff, "1.0000e-309");

  written = LIBC_NAMESPACE::sprintf(buff, "%.3e", 9.99999e-310);
  ASSERT_STREQ_LEN(written, buff, "1.000e-309");

  written = LIBC_NAMESPACE::sprintf(buff, "%.2e", 9.99999e-310);
  ASSERT_STREQ_LEN(written, buff, "1.00e-309");

  written = LIBC_NAMESPACE::sprintf(buff, "%.1e", 9.99999e-310);
  ASSERT_STREQ_LEN(written, buff, "1.0e-309");

  written = LIBC_NAMESPACE::sprintf(buff, "%.0e", 9.99999e-310);
  ASSERT_STREQ_LEN(written, buff, "1e-309");

  written = LIBC_NAMESPACE::sprintf(buff, "%.10e", 0x1.0p-1074);
  ASSERT_STREQ_LEN(written, buff, "4.9406564584e-324");

  /*
    written = LIBC_NAMESPACE::sprintf(buff, "%.1La", 0.1L);
  #if defined(LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80)
    ASSERT_STREQ_LEN(written, buff, "0xc.dp-7");
  #elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT64)
    ASSERT_STREQ_LEN(written, buff, "0x1.ap-4");
  #elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT128)
    ASSERT_STREQ_LEN(written, buff, "0x1.ap-4");
  #endif

    written = LIBC_NAMESPACE::sprintf(buff, "%.1La",
  0xf.fffffffffffffffp16380L); #if
  defined(LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80) ASSERT_STREQ_LEN(written, buff,
  "0x1.0p+16384"); #elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT64)
    ASSERT_STREQ_LEN(written, buff, "inf");
  #elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT128)
    ASSERT_STREQ_LEN(written, buff, "0x2.0p+16383");
  #endif
  */

  // Rounding Mode Tests.

  if (ForceRoundingMode r(RoundingMode::Nearest); r.success) {
    written = LIBC_NAMESPACE::sprintf(buff, "%.1e", 1.75);
    ASSERT_STREQ_LEN(written, buff, "1.8e+00");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1e", 1.25);
    ASSERT_STREQ_LEN(written, buff, "1.2e+00");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1e", 1.125);
    ASSERT_STREQ_LEN(written, buff, "1.1e+00");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1e", 1.625);
    ASSERT_STREQ_LEN(written, buff, "1.6e+00");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1e", 1.375);
    ASSERT_STREQ_LEN(written, buff, "1.4e+00");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1e", 1.875);
    ASSERT_STREQ_LEN(written, buff, "1.9e+00");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1e", -1.75);
    ASSERT_STREQ_LEN(written, buff, "-1.8e+00");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1e", -1.25);
    ASSERT_STREQ_LEN(written, buff, "-1.2e+00");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1e", -1.125);
    ASSERT_STREQ_LEN(written, buff, "-1.1e+00");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1e", -1.625);
    ASSERT_STREQ_LEN(written, buff, "-1.6e+00");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1e", -1.375);
    ASSERT_STREQ_LEN(written, buff, "-1.4e+00");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1e", -1.875);
    ASSERT_STREQ_LEN(written, buff, "-1.9e+00");
  }

  if (ForceRoundingMode r(RoundingMode::Upward); r.success) {
    written = LIBC_NAMESPACE::sprintf(buff, "%.1e", 1.75);
    ASSERT_STREQ_LEN(written, buff, "1.8e+00");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1e", 1.25);
    ASSERT_STREQ_LEN(written, buff, "1.3e+00");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1e", 1.125);
    ASSERT_STREQ_LEN(written, buff, "1.2e+00");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1e", 1.625);
    ASSERT_STREQ_LEN(written, buff, "1.7e+00");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1e", 1.375);
    ASSERT_STREQ_LEN(written, buff, "1.4e+00");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1e", 1.875);
    ASSERT_STREQ_LEN(written, buff, "1.9e+00");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1e", -1.75);
    ASSERT_STREQ_LEN(written, buff, "-1.7e+00");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1e", -1.25);
    ASSERT_STREQ_LEN(written, buff, "-1.2e+00");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1e", -1.125);
    ASSERT_STREQ_LEN(written, buff, "-1.1e+00");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1e", -1.625);
    ASSERT_STREQ_LEN(written, buff, "-1.6e+00");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1e", -1.375);
    ASSERT_STREQ_LEN(written, buff, "-1.3e+00");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1e", -1.875);
    ASSERT_STREQ_LEN(written, buff, "-1.8e+00");
  }

  if (ForceRoundingMode r(RoundingMode::Downward); r.success) {
    written = LIBC_NAMESPACE::sprintf(buff, "%.1e", 1.75);
    ASSERT_STREQ_LEN(written, buff, "1.7e+00");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1e", 1.25);
    ASSERT_STREQ_LEN(written, buff, "1.2e+00");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1e", 1.125);
    ASSERT_STREQ_LEN(written, buff, "1.1e+00");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1e", 1.625);
    ASSERT_STREQ_LEN(written, buff, "1.6e+00");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1e", 1.375);
    ASSERT_STREQ_LEN(written, buff, "1.3e+00");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1e", 1.875);
    ASSERT_STREQ_LEN(written, buff, "1.8e+00");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1e", -1.75);
    ASSERT_STREQ_LEN(written, buff, "-1.8e+00");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1e", -1.25);
    ASSERT_STREQ_LEN(written, buff, "-1.3e+00");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1e", -1.125);
    ASSERT_STREQ_LEN(written, buff, "-1.2e+00");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1e", -1.625);
    ASSERT_STREQ_LEN(written, buff, "-1.7e+00");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1e", -1.375);
    ASSERT_STREQ_LEN(written, buff, "-1.4e+00");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1e", -1.875);
    ASSERT_STREQ_LEN(written, buff, "-1.9e+00");
  }

  if (ForceRoundingMode r(RoundingMode::TowardZero); r.success) {
    written = LIBC_NAMESPACE::sprintf(buff, "%.1e", 1.75);
    ASSERT_STREQ_LEN(written, buff, "1.7e+00");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1e", 1.25);
    ASSERT_STREQ_LEN(written, buff, "1.2e+00");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1e", 1.125);
    ASSERT_STREQ_LEN(written, buff, "1.1e+00");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1e", 1.625);
    ASSERT_STREQ_LEN(written, buff, "1.6e+00");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1e", 1.375);
    ASSERT_STREQ_LEN(written, buff, "1.3e+00");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1e", 1.875);
    ASSERT_STREQ_LEN(written, buff, "1.8e+00");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1e", -1.75);
    ASSERT_STREQ_LEN(written, buff, "-1.7e+00");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1e", -1.25);
    ASSERT_STREQ_LEN(written, buff, "-1.2e+00");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1e", -1.125);
    ASSERT_STREQ_LEN(written, buff, "-1.1e+00");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1e", -1.625);
    ASSERT_STREQ_LEN(written, buff, "-1.6e+00");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1e", -1.375);
    ASSERT_STREQ_LEN(written, buff, "-1.3e+00");

    written = LIBC_NAMESPACE::sprintf(buff, "%.1e", -1.875);
    ASSERT_STREQ_LEN(written, buff, "-1.8e+00");
  }

  // Flag Tests.
  written = LIBC_NAMESPACE::sprintf(buff, "%+e", 1.0);
  ASSERT_STREQ_LEN(written, buff, "+1.000000e+00");

  written = LIBC_NAMESPACE::sprintf(buff, "%+e", -1.0);
  ASSERT_STREQ_LEN(written, buff, "-1.000000e+00");

  written = LIBC_NAMESPACE::sprintf(buff, "% e", 1.0);
  ASSERT_STREQ_LEN(written, buff, " 1.000000e+00");

  written = LIBC_NAMESPACE::sprintf(buff, "% e", -1.0);
  ASSERT_STREQ_LEN(written, buff, "-1.000000e+00");

  written = LIBC_NAMESPACE::sprintf(buff, "%-15e", 1.5);
  ASSERT_STREQ_LEN(written, buff, "1.500000e+00   ");

  written = LIBC_NAMESPACE::sprintf(buff, "%#.e", 1.0);
  ASSERT_STREQ_LEN(written, buff, "1.e+00");

  written = LIBC_NAMESPACE::sprintf(buff, "%#.0e", 1.5);
  ASSERT_STREQ_LEN(written, buff, "2.e+00");

  written = LIBC_NAMESPACE::sprintf(buff, "%015e", 1.5);
  ASSERT_STREQ_LEN(written, buff, "0001.500000e+00");

  written = LIBC_NAMESPACE::sprintf(buff, "%015e", -1.5);
  ASSERT_STREQ_LEN(written, buff, "-001.500000e+00");

  written = LIBC_NAMESPACE::sprintf(buff, "%+- #0e", 0.0);
  ASSERT_STREQ_LEN(written, buff, "+0.000000e+00");

  // Combined Tests.

  written = LIBC_NAMESPACE::sprintf(buff, "%10.2e", 9.99);
  ASSERT_STREQ_LEN(written, buff, "  9.99e+00");

  written = LIBC_NAMESPACE::sprintf(buff, "%10.1e", 9.99);
  ASSERT_STREQ_LEN(written, buff, "   1.0e+01");

  written = LIBC_NAMESPACE::sprintf(buff, "%10.0e", 9.99);
  ASSERT_STREQ_LEN(written, buff, "     1e+01");

  written = LIBC_NAMESPACE::sprintf(buff, "%10.0e", 0.0999);
  ASSERT_STREQ_LEN(written, buff, "     1e-01");

  written = LIBC_NAMESPACE::sprintf(buff, "%-10.2e", 9.99);
  ASSERT_STREQ_LEN(written, buff, "9.99e+00  ");

  written = LIBC_NAMESPACE::sprintf(buff, "%-10.1e", 9.99);
  ASSERT_STREQ_LEN(written, buff, "1.0e+01   ");

  written = LIBC_NAMESPACE::sprintf(buff, "%-10.1e", 1.0e-50);
  ASSERT_STREQ_LEN(written, buff, "1.0e-50   ");

  written = LIBC_NAMESPACE::sprintf(buff, "%30e", 1234567890123456789.0);
  ASSERT_STREQ_LEN(written, buff, "                  1.234568e+18");

  written = LIBC_NAMESPACE::sprintf(buff, "%-30e", 1234567890123456789.0);
  ASSERT_STREQ_LEN(written, buff, "1.234568e+18                  ");

  written = LIBC_NAMESPACE::sprintf(buff, "%25.14e", 9999999999999.99);
  ASSERT_STREQ_LEN(written, buff, "     9.99999999999999e+12");

  written = LIBC_NAMESPACE::sprintf(buff, "%25.13e", 9999999999999.99);
  ASSERT_STREQ_LEN(written, buff, "      1.0000000000000e+13");

  written = LIBC_NAMESPACE::sprintf(buff, "%25.12e", 9999999999999.99);
  ASSERT_STREQ_LEN(written, buff, "       1.000000000000e+13");

  written = LIBC_NAMESPACE::sprintf(buff, "%12.3e %-12.3e", 0.1, 256.0);
  ASSERT_STREQ_LEN(written, buff, "   1.000e-01 2.560e+02   ");

  written = LIBC_NAMESPACE::sprintf(buff, "%+-#12.3e % 012.3e", 0.1256, 1256.0);
  ASSERT_STREQ_LEN(written, buff, "+1.256e-01    001.256e+03");
}

TEST_F(LlvmLibcSPrintfTest, FloatExponentLongDoubleConv) {
  ForceRoundingMode r(RoundingMode::Nearest);
  // Length Modifier Tests.

#if defined(LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80)
  written = LIBC_NAMESPACE::sprintf(buff, "%.9Le", 1000000000500000000.1L);
  ASSERT_STREQ_LEN(written, buff, "1.000000001e+18");

  written = LIBC_NAMESPACE::sprintf(buff, "%.9Le", 1000000000500000000.0L);
  ASSERT_STREQ_LEN(written, buff, "1.000000000e+18");

  written = LIBC_NAMESPACE::sprintf(buff, "%Le", 0xf.fffffffffffffffp+16380L);
  ASSERT_STREQ_LEN(written, buff, "1.189731e+4932");
#endif

  // TODO: Fix long doubles (needs bigger table or alternate algorithm.)
  // Currently the table values are generated, which is very slow.
  /*
  written = LIBC_NAMESPACE::sprintf(buff, "%Lf", 1e100L);
  ASSERT_STREQ_LEN(written, buff,
                   "99999999999999999996693535322073426194986990198284960792713"
                   "91541752018669482644324418977840117055488.000000");

  written = LIBC_NAMESPACE::sprintf(buff, "%Lf", 1.0L);
  ASSERT_STREQ_LEN(written, buff, "1.000000");

  char big_buff[10000];
  written = LIBC_NAMESPACE::sprintf(big_buff, "%Lf", 1e1000L);
  ASSERT_STREQ_LEN(
      written, big_buff,
      "999999999999999999973107317669562353428234857594552594925899449376328728"
      "202461036775511405481186963193066642191664822065529414252060696836533522"
      "387143501724276282079456797058697369889056407118642873669166717313763499"
      "277025985141177344925615052465165938514140943010597323750202561187880136"
      "174810574553749194614479541820148407958204853833697063267336294787191005"
      "628217462261955103745349844675732989944229689277833828743730290177882029"
      "042613704915899149603539993716885598351951895974316347947147507970269673"
      "097709017164643598452451201499004104341931127294141495501309305995449742"
      "273419524803597130450457553871345958049837885085168840317195672271085085"
      "950520957945970913451088104971436093671776829538796532762184174216651692"
      "640931965387852083906784898823494867055070322768919156031682291829761007"
      "101483799978382119231551218582499361996919560548090784230386907125151658"
      "086767207295524036170321059257942621398084478974000973622199163292708506"
      "2431457550909271560663602154947063707982236377366647567795879936."
      "000000");

  written = LIBC_NAMESPACE::sprintf(big_buff, "%Lf", 1e4900L);
  ASSERT_STREQ_LEN(
      written, big_buff,
      "100000000000000000002708312230690349833224052504078834346502930111959028"
      "517260692666637048230414374897655201843766090626319971729765251179632020"
      "313912652522792711197087872698264530532442630109549129842736280196919130"
      "242615101228133188193853826983121366159061148351354364472807590931218045"
      "387490935930967150336231085015126034696883068553581691802388371635128003"
      "615577299166097675723780877126495909902479233742826339471026068806070433"
      "075629449530819183550315434973800271862658869400009022028602967197463980"
      "126881829804282202449930132940824361207087494829502385835258094836304011"
      "876250359661206802659650567866176246063987902366800491980400341950657151"
      "370854446585517805253310195469184699955519312761482572080479702840420595"
      "377369017651259376039167277822106875560385309101650382998482652792335482"
      "865443482342801545877390859444282105890147577937366066315975231014810320"
      "888482059656248277607763361589359794524314002443575149260630989130103550"
      "443177966380769341050735632338583912575890190136462629316287947355057647"
      "111088565611192544631519843618778618820046304429723908484879583579178075"
      "456701368334212923379389029311286386996015804122917416008806233549005183"
      "152461084266176543129004016414959261473645240454289630182591200574019087"
      "358223489767381636349719510715487188747217311279465814538495924567014916"
      "238565628036285599497236493491668884212847699052761266207598941300449276"
      "447201387520841811835583254242213093566548778954711633721122784159793843"
      "766802019309395771984693609426401362800013936338891483689127845928572536"
      "790651156184721483511507878883282891696900630100211914227950790472211403"
      "392549466062537498185758854079775888444518306635752468713312357556380082"
      "275500658967283696421824354930077523691855699312544373220921962817907078"
      "445538421941800259027487429330768616490865438859612697367766323925013940"
      "918384858952407145253573823848733994146335416209309233074165707437420756"
      "438833918763109580759409985573826485055208965115587885226774453455112406"
      "581351429640282227888764449360534584421929291565334894907337572527922691"
      "473242328379737396430908523008687037407295838014450772162091496534584696"
      "605157436893236842602956298545594095307060870397506421786236892553632163"
      "491468601982681381011940409602294892199042638682530687578982576819839451"
      "907594697546439533559153604700750696252355362322662219852740143212566818"
      "745528402265116534684566273868361460640280523251242059850044328669692159"
      "629900374576027104298177006629276014371540945261309319363704125592775129"
      "543526908667388673739382491147471395192495459318806593271282662311169392"
      "196897003517840025298267505925987901751541005546610016067658227181318892"
      "914686508281007582655667597441346214499847364272258631922040641860333431"
      "409838623713258383681350233064164940590695888300919626215847587544298023"
      "636416943680102708406086295669759876682046839368574433996997648445207805"
      "615784339667691231286807666753972942872019850432610318031627872612657513"
      "588188267160616660825719678199868371370527508463011236193719286066916786"
      "169956541349011494927225747024994619057884118692213564790598702879596058"
      "672338334720925179141906809470606964896245458600635183723159228561689808"
      "246141482736625197373238197777325580142168245885279594913851700941789475"
      "252421784152262567254611571822468808675893407728003047921107885664474662"
      "930921581384003950729114103689170603748380178682003976896397305836815761"
      "717676338115866650889936516794601457549097578905329423919798362140648664"
      "569177147076571576101649257502509463877402424847669830852345415301684820"
      "395813946416649808062227494112874521812750160935760825922220707178083076"
      "380203450993589198835885505461509442443773367592842795410339065860781804"
      "024975272228687688301824830333940416256885455008512598774611538878683158"
      "183931461086893832255176926531299425504132104728730288984598001187854507"
      "900417184206801359847651992484444933900133130832052346600926424167009902"
      "829803553087005800387704758687923428053612864451456596148162238935900033"
      "917094683141205188616000211702577553792389670853917118547527592495253773"
      "028135298405566315903922235989614934474805789300370437580494193066066314"
      "056627605207631392651010580925826419831250810981343093764403877594495896"
      "516881097415880926429607388979497471571321217205535961262051641426436441"
      "668989765107456413733909427384182109285933511623871034309722437967253289"
      "084018145083721513211807496392673952789642893241520398827805325610653506"
      "029060153153064455898648607959013571280930834475689835845791849456112104"
      "462337569019001580859906425911782967213265389744605395555069797947978230"
      "708108432086217134763779632408473684293543722127232658767439906910370146"
      "716836295909075482355827087389127370874842532825987593970846704144140471"
      "956027276735614286138656432085771988513977140957180090146798065497158947"
      "229765733489703157617307078835099906185890777007500964162371428641176460"
      "739074789794941408428328217107759915202650066155868439585510978709442590"
      "231934194956788626761834746430104077432547436359522462253411168467463134"
      "24896.000000");
*/
}

TEST_F(LlvmLibcSPrintfTest, FloatAutoConv) {
  ForceRoundingMode r(RoundingMode::Nearest);
  double inf = LIBC_NAMESPACE::fputil::FPBits<double>::inf().get_val();
  double nan = LIBC_NAMESPACE::fputil::FPBits<double>::quiet_nan().get_val();

  written = LIBC_NAMESPACE::sprintf(buff, "%g", 1.0);
  ASSERT_STREQ_LEN(written, buff, "1");

  written = LIBC_NAMESPACE::sprintf(buff, "%G", -1.0);
  ASSERT_STREQ_LEN(written, buff, "-1");

  written = LIBC_NAMESPACE::sprintf(buff, "%g", -1.234567);
  ASSERT_STREQ_LEN(written, buff, "-1.23457");

  written = LIBC_NAMESPACE::sprintf(buff, "%g", 0.0);
  ASSERT_STREQ_LEN(written, buff, "0");

  written = LIBC_NAMESPACE::sprintf(buff, "%g", -0.0);
  ASSERT_STREQ_LEN(written, buff, "-0");

  written = LIBC_NAMESPACE::sprintf(buff, "%g", 1.5);
  ASSERT_STREQ_LEN(written, buff, "1.5");

  written = LIBC_NAMESPACE::sprintf(buff, "%g", 1e300);
  ASSERT_STREQ_LEN(written, buff, "1e+300");

  written = LIBC_NAMESPACE::sprintf(buff, "%g", 0.1);
  ASSERT_STREQ_LEN(written, buff, "0.1");

  written = LIBC_NAMESPACE::sprintf(buff, "%g", 0.001);
  ASSERT_STREQ_LEN(written, buff, "0.001");

  written = LIBC_NAMESPACE::sprintf(buff, "%g", 0.00001);
  ASSERT_STREQ_LEN(written, buff, "1e-05");

  written = LIBC_NAMESPACE::sprintf(buff, "%g", 0.0000001);
  ASSERT_STREQ_LEN(written, buff, "1e-07");

  written = LIBC_NAMESPACE::sprintf(buff, "%g", 0.000000001);
  ASSERT_STREQ_LEN(written, buff, "1e-09");

  written = LIBC_NAMESPACE::sprintf(buff, "%g", 1.0e-20);
  ASSERT_STREQ_LEN(written, buff, "1e-20");

  written = LIBC_NAMESPACE::sprintf(buff, "%g", 1234567890123456789.0);
  ASSERT_STREQ_LEN(written, buff, "1.23457e+18");

  written = LIBC_NAMESPACE::sprintf(buff, "%g", 9999990000000.00);
  ASSERT_STREQ_LEN(written, buff, "9.99999e+12");

  written = LIBC_NAMESPACE::sprintf(buff, "%g", 9999999000000.00);
  ASSERT_STREQ_LEN(written, buff, "1e+13");

  written = LIBC_NAMESPACE::sprintf(buff, "%g", 0xa.aaaaaaaaaaaaaabp-7);
  ASSERT_STREQ_LEN(written, buff, "0.0833333");

  // Simple Subnormal Tests.

  written = LIBC_NAMESPACE::sprintf(buff, "%g", 0x1.0p-1027);
  ASSERT_STREQ_LEN(written, buff, "6.95336e-310");

  written = LIBC_NAMESPACE::sprintf(buff, "%g", 0x1.0p-1074);
  ASSERT_STREQ_LEN(written, buff, "4.94066e-324");

  // Inf/Nan Tests.

  written = LIBC_NAMESPACE::sprintf(buff, "%g", inf);
  ASSERT_STREQ_LEN(written, buff, "inf");

  written = LIBC_NAMESPACE::sprintf(buff, "%G", -inf);
  ASSERT_STREQ_LEN(written, buff, "-INF");

  written = LIBC_NAMESPACE::sprintf(buff, "%g", nan);
  ASSERT_STREQ_LEN(written, buff, "nan");

  written = LIBC_NAMESPACE::sprintf(buff, "%G", -nan);
  ASSERT_STREQ_LEN(written, buff, "-NAN");

  // Min Width Tests.

  written = LIBC_NAMESPACE::sprintf(buff, "%15g", 1.0);
  ASSERT_STREQ_LEN(written, buff, "              1");

  written = LIBC_NAMESPACE::sprintf(buff, "%15g", -1.0);
  ASSERT_STREQ_LEN(written, buff, "             -1");

  written = LIBC_NAMESPACE::sprintf(buff, "%15g", 1.0e5);
  ASSERT_STREQ_LEN(written, buff, "         100000");

  written = LIBC_NAMESPACE::sprintf(buff, "%15g", -1.0e5);
  ASSERT_STREQ_LEN(written, buff, "        -100000");

  written = LIBC_NAMESPACE::sprintf(buff, "%10g", 1.0e-5);
  ASSERT_STREQ_LEN(written, buff, "     1e-05");

  // Precision Tests.

  written = LIBC_NAMESPACE::sprintf(buff, "%.2g", 1.23456789);
  ASSERT_STREQ_LEN(written, buff, "1.2");

  // Trimming trailing zeroes causes the precision to be ignored here.
  written = LIBC_NAMESPACE::sprintf(buff, "%.1g", 0.0);
  ASSERT_STREQ_LEN(written, buff, "0");

  written = LIBC_NAMESPACE::sprintf(buff, "%.0g", 0.0);
  ASSERT_STREQ_LEN(written, buff, "0");

  written = LIBC_NAMESPACE::sprintf(buff, "%.2g", 0.1);
  ASSERT_STREQ_LEN(written, buff, "0.1");

  written = LIBC_NAMESPACE::sprintf(buff, "%.2g", 1.09);
  ASSERT_STREQ_LEN(written, buff, "1.1");

  written = LIBC_NAMESPACE::sprintf(buff, "%.2g", 1.04);
  ASSERT_STREQ_LEN(written, buff, "1");

  written = LIBC_NAMESPACE::sprintf(buff, "%.2g", 1.19);
  ASSERT_STREQ_LEN(written, buff, "1.2");

  written = LIBC_NAMESPACE::sprintf(buff, "%.2g", 1.99);
  ASSERT_STREQ_LEN(written, buff, "2");

  written = LIBC_NAMESPACE::sprintf(buff, "%.2g", 9.99);
  ASSERT_STREQ_LEN(written, buff, "10");

  written = LIBC_NAMESPACE::sprintf(buff, "%.3g", 99.9);
  ASSERT_STREQ_LEN(written, buff, "99.9");

  written = LIBC_NAMESPACE::sprintf(buff, "%.2g", 99.9);
  ASSERT_STREQ_LEN(written, buff, "1e+02");

  written = LIBC_NAMESPACE::sprintf(buff, "%.1g", 99.9);
  ASSERT_STREQ_LEN(written, buff, "1e+02");

  written = LIBC_NAMESPACE::sprintf(buff, "%.5g", 1.25);
  ASSERT_STREQ_LEN(written, buff, "1.25");

  written = LIBC_NAMESPACE::sprintf(buff, "%.0g", 1.25);
  ASSERT_STREQ_LEN(written, buff, "1");

  written = LIBC_NAMESPACE::sprintf(buff, "%.0g", 1.75);
  ASSERT_STREQ_LEN(written, buff, "2");

  written = LIBC_NAMESPACE::sprintf(buff, "%.20g", 1.234e-10);
  ASSERT_STREQ_LEN(written, buff, "1.2340000000000000814e-10");

  written = LIBC_NAMESPACE::sprintf(buff, "%.3g", -9.99);
  ASSERT_STREQ_LEN(written, buff, "-9.99");

  written = LIBC_NAMESPACE::sprintf(buff, "%.2g", -9.99);
  ASSERT_STREQ_LEN(written, buff, "-10");

  written = LIBC_NAMESPACE::sprintf(buff, "%.1g", -9.99);
  ASSERT_STREQ_LEN(written, buff, "-1e+01");

  written = LIBC_NAMESPACE::sprintf(buff, "%.5g", 1.008);
  ASSERT_STREQ_LEN(written, buff, "1.008");

  written = LIBC_NAMESPACE::sprintf(buff, "%.5g", 1.008e3);
  ASSERT_STREQ_LEN(written, buff, "1008");

  written = LIBC_NAMESPACE::sprintf(buff, "%.4g", 9999.0);
  ASSERT_STREQ_LEN(written, buff, "9999");

  written = LIBC_NAMESPACE::sprintf(buff, "%.3g", 9999.0);
  ASSERT_STREQ_LEN(written, buff, "1e+04");

  written = LIBC_NAMESPACE::sprintf(buff, "%.3g", 1256.0);
  ASSERT_STREQ_LEN(written, buff, "1.26e+03");

  // Found through large scale testing.
  written = LIBC_NAMESPACE::sprintf(buff, "%.15g", 22.25);
  ASSERT_STREQ_LEN(written, buff, "22.25");

  // These tests also focus on rounding, but only in how it relates to the base
  // 10 exponent. The %g conversion selects between being a %f or %e conversion
  // based on what the exponent would be if it was %e. If we call the precision
  // P (equal to 6 if the precision is not set, 0 if the provided precision is
  // 0, and provided precision - 1 otherwise) and the exponent X, then the style
  // is %f with an effective precision of P - X + 1 if P > X >= -4, else the
  // style is %e with effective precision P - 1. Additionally, it attempts to
  // trim zeros that would be displayed after the decimal point.
  written = LIBC_NAMESPACE::sprintf(buff, "%.1g", 9.059E+200);
  ASSERT_STREQ_LEN(written, buff, "9e+200");

  written = LIBC_NAMESPACE::sprintf(buff, "%.2g", 9.059E+200);
  ASSERT_STREQ_LEN(written, buff, "9.1e+200");

  // For this test, P = 0 and X = 1, so P > X >= -4 is false, giving a %e style.
  written = LIBC_NAMESPACE::sprintf(buff, "%.0g", 9.5);
  ASSERT_STREQ_LEN(written, buff, "1e+01");

  // Subnormal Precision Tests
  written = LIBC_NAMESPACE::sprintf(buff, "%.310g", 0x1.0p-1022);
  ASSERT_STREQ_LEN(
      written, buff,
      "2."
      "225073858507201383090232717332404064219215980462331830553327416887204434"
      "813918195854283159012511020564067339731035811005152434161553460108856012"
      "385377718821130777993532002330479610147442583636071921565046942503734208"
      "375250806650616658158948720491179968591639648500635908770118304874799780"
      "887753749949451580452e-308");

  written = LIBC_NAMESPACE::sprintf(buff, "%.30g", 0x1.0p-1022);
  ASSERT_STREQ_LEN(written, buff, "2.22507385850720138309023271733e-308");

  written = LIBC_NAMESPACE::sprintf(buff, "%.310g", 0x1.0p-1023);
  ASSERT_STREQ_LEN(
      written, buff,
      "1."
      "112536929253600691545116358666202032109607990231165915276663708443602217"
      "406959097927141579506255510282033669865517905502576217080776730054428006"
      "192688859410565388996766001165239805073721291818035960782523471251867104"
      "187625403325308329079474360245589984295819824250317954385059152437399890"
      "443876874974725790226e-308");

  written = LIBC_NAMESPACE::sprintf(buff, "%.7g", 9.99999e-310);
  ASSERT_STREQ_LEN(written, buff, "9.99999e-310");

  written = LIBC_NAMESPACE::sprintf(buff, "%.6g", 9.99999e-310);
  ASSERT_STREQ_LEN(written, buff, "9.99999e-310");

  written = LIBC_NAMESPACE::sprintf(buff, "%.5g", 9.99999e-310);
  ASSERT_STREQ_LEN(written, buff, "1e-309");

  written = LIBC_NAMESPACE::sprintf(buff, "%.4g", 9.99999e-310);
  ASSERT_STREQ_LEN(written, buff, "1e-309");

  written = LIBC_NAMESPACE::sprintf(buff, "%.3g", 9.99999e-310);
  ASSERT_STREQ_LEN(written, buff, "1e-309");

  written = LIBC_NAMESPACE::sprintf(buff, "%.2g", 9.99999e-310);
  ASSERT_STREQ_LEN(written, buff, "1e-309");

  written = LIBC_NAMESPACE::sprintf(buff, "%.1g", 9.99999e-310);
  ASSERT_STREQ_LEN(written, buff, "1e-309");

  written = LIBC_NAMESPACE::sprintf(buff, "%.0g", 9.99999e-310);
  ASSERT_STREQ_LEN(written, buff, "1e-309");

  written = LIBC_NAMESPACE::sprintf(buff, "%.10g", 0x1.0p-1074);
  ASSERT_STREQ_LEN(written, buff, "4.940656458e-324");

#if defined(LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80)

  written = LIBC_NAMESPACE::sprintf(buff, "%.60Lg", 0xa.aaaaaaaaaaaaaabp-7L);
  ASSERT_STREQ_LEN(
      written, buff,
      "0.0833333333333333333355920878593448009041821933351457118988037");

#endif // LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80

  // Long double precision tests.
  // These are currently commented out because they require long double support
  // that isn't ready yet.
  /*
    written = LIBC_NAMESPACE::sprintf(buff, "%.1La", 0.1L);
  #if defined(LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80)
    ASSERT_STREQ_LEN(written, buff, "0xc.dp-7");
  #elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT64)
    ASSERT_STREQ_LEN(written, buff, "0x1.ap-4");
  #elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT128)
    ASSERT_STREQ_LEN(written, buff, "0x1.ap-4");
  #endif

    written = LIBC_NAMESPACE::sprintf(buff, "%.1La",
  0xf.fffffffffffffffp16380L); #if
  defined(LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80) ASSERT_STREQ_LEN(written, buff,
  "0x1.0p+16384"); #elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT64)
    ASSERT_STREQ_LEN(written, buff, "inf");
  #elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT128)
    ASSERT_STREQ_LEN(written, buff, "0x2.0p+16383");
  #endif
  */

  // Rounding Mode Tests.

  if (ForceRoundingMode r(RoundingMode::Nearest); r.success) {
    written = LIBC_NAMESPACE::sprintf(buff, "%.2g", 1.75);
    ASSERT_STREQ_LEN(written, buff, "1.8");

    written = LIBC_NAMESPACE::sprintf(buff, "%.2g", 1.25);
    ASSERT_STREQ_LEN(written, buff, "1.2");

    written = LIBC_NAMESPACE::sprintf(buff, "%.2g", 1.125);
    ASSERT_STREQ_LEN(written, buff, "1.1");

    written = LIBC_NAMESPACE::sprintf(buff, "%.2g", 1.625);
    ASSERT_STREQ_LEN(written, buff, "1.6");

    written = LIBC_NAMESPACE::sprintf(buff, "%.2g", 1.375);
    ASSERT_STREQ_LEN(written, buff, "1.4");

    written = LIBC_NAMESPACE::sprintf(buff, "%.2g", 1.875);
    ASSERT_STREQ_LEN(written, buff, "1.9");

    written = LIBC_NAMESPACE::sprintf(buff, "%.2g", -1.75);
    ASSERT_STREQ_LEN(written, buff, "-1.8");

    written = LIBC_NAMESPACE::sprintf(buff, "%.2g", -1.25);
    ASSERT_STREQ_LEN(written, buff, "-1.2");

    written = LIBC_NAMESPACE::sprintf(buff, "%.2g", -1.125);
    ASSERT_STREQ_LEN(written, buff, "-1.1");

    written = LIBC_NAMESPACE::sprintf(buff, "%.2g", -1.625);
    ASSERT_STREQ_LEN(written, buff, "-1.6");

    written = LIBC_NAMESPACE::sprintf(buff, "%.2g", -1.375);
    ASSERT_STREQ_LEN(written, buff, "-1.4");

    written = LIBC_NAMESPACE::sprintf(buff, "%.2g", -1.875);
    ASSERT_STREQ_LEN(written, buff, "-1.9");
  }

  if (ForceRoundingMode r(RoundingMode::Upward); r.success) {
    written = LIBC_NAMESPACE::sprintf(buff, "%.2g", 1.75);
    ASSERT_STREQ_LEN(written, buff, "1.8");

    written = LIBC_NAMESPACE::sprintf(buff, "%.2g", 1.25);
    ASSERT_STREQ_LEN(written, buff, "1.3");

    written = LIBC_NAMESPACE::sprintf(buff, "%.2g", 1.125);
    ASSERT_STREQ_LEN(written, buff, "1.2");

    written = LIBC_NAMESPACE::sprintf(buff, "%.2g", 1.625);
    ASSERT_STREQ_LEN(written, buff, "1.7");

    written = LIBC_NAMESPACE::sprintf(buff, "%.2g", 1.375);
    ASSERT_STREQ_LEN(written, buff, "1.4");

    written = LIBC_NAMESPACE::sprintf(buff, "%.2g", 1.875);
    ASSERT_STREQ_LEN(written, buff, "1.9");

    written = LIBC_NAMESPACE::sprintf(buff, "%.2g", -1.75);
    ASSERT_STREQ_LEN(written, buff, "-1.7");

    written = LIBC_NAMESPACE::sprintf(buff, "%.2g", -1.25);
    ASSERT_STREQ_LEN(written, buff, "-1.2");

    written = LIBC_NAMESPACE::sprintf(buff, "%.2g", -1.125);
    ASSERT_STREQ_LEN(written, buff, "-1.1");

    written = LIBC_NAMESPACE::sprintf(buff, "%.2g", -1.625);
    ASSERT_STREQ_LEN(written, buff, "-1.6");

    written = LIBC_NAMESPACE::sprintf(buff, "%.2g", -1.375);
    ASSERT_STREQ_LEN(written, buff, "-1.3");

    written = LIBC_NAMESPACE::sprintf(buff, "%.2g", -1.875);
    ASSERT_STREQ_LEN(written, buff, "-1.8");
  }

  if (ForceRoundingMode r(RoundingMode::Downward); r.success) {
    written = LIBC_NAMESPACE::sprintf(buff, "%.2g", 1.75);
    ASSERT_STREQ_LEN(written, buff, "1.7");

    written = LIBC_NAMESPACE::sprintf(buff, "%.2g", 1.25);
    ASSERT_STREQ_LEN(written, buff, "1.2");

    written = LIBC_NAMESPACE::sprintf(buff, "%.2g", 1.125);
    ASSERT_STREQ_LEN(written, buff, "1.1");

    written = LIBC_NAMESPACE::sprintf(buff, "%.2g", 1.625);
    ASSERT_STREQ_LEN(written, buff, "1.6");

    written = LIBC_NAMESPACE::sprintf(buff, "%.2g", 1.375);
    ASSERT_STREQ_LEN(written, buff, "1.3");

    written = LIBC_NAMESPACE::sprintf(buff, "%.2g", 1.875);
    ASSERT_STREQ_LEN(written, buff, "1.8");

    written = LIBC_NAMESPACE::sprintf(buff, "%.2g", -1.75);
    ASSERT_STREQ_LEN(written, buff, "-1.8");

    written = LIBC_NAMESPACE::sprintf(buff, "%.2g", -1.25);
    ASSERT_STREQ_LEN(written, buff, "-1.3");

    written = LIBC_NAMESPACE::sprintf(buff, "%.2g", -1.125);
    ASSERT_STREQ_LEN(written, buff, "-1.2");

    written = LIBC_NAMESPACE::sprintf(buff, "%.2g", -1.625);
    ASSERT_STREQ_LEN(written, buff, "-1.7");

    written = LIBC_NAMESPACE::sprintf(buff, "%.2g", -1.375);
    ASSERT_STREQ_LEN(written, buff, "-1.4");

    written = LIBC_NAMESPACE::sprintf(buff, "%.2g", -1.875);
    ASSERT_STREQ_LEN(written, buff, "-1.9");
  }

  if (ForceRoundingMode r(RoundingMode::TowardZero); r.success) {
    written = LIBC_NAMESPACE::sprintf(buff, "%.2g", 1.75);
    ASSERT_STREQ_LEN(written, buff, "1.7");

    written = LIBC_NAMESPACE::sprintf(buff, "%.2g", 1.25);
    ASSERT_STREQ_LEN(written, buff, "1.2");

    written = LIBC_NAMESPACE::sprintf(buff, "%.2g", 1.125);
    ASSERT_STREQ_LEN(written, buff, "1.1");

    written = LIBC_NAMESPACE::sprintf(buff, "%.2g", 1.625);
    ASSERT_STREQ_LEN(written, buff, "1.6");

    written = LIBC_NAMESPACE::sprintf(buff, "%.2g", 1.375);
    ASSERT_STREQ_LEN(written, buff, "1.3");

    written = LIBC_NAMESPACE::sprintf(buff, "%.2g", 1.875);
    ASSERT_STREQ_LEN(written, buff, "1.8");

    written = LIBC_NAMESPACE::sprintf(buff, "%.2g", -1.75);
    ASSERT_STREQ_LEN(written, buff, "-1.7");

    written = LIBC_NAMESPACE::sprintf(buff, "%.2g", -1.25);
    ASSERT_STREQ_LEN(written, buff, "-1.2");

    written = LIBC_NAMESPACE::sprintf(buff, "%.2g", -1.125);
    ASSERT_STREQ_LEN(written, buff, "-1.1");

    written = LIBC_NAMESPACE::sprintf(buff, "%.2g", -1.625);
    ASSERT_STREQ_LEN(written, buff, "-1.6");

    written = LIBC_NAMESPACE::sprintf(buff, "%.2g", -1.375);
    ASSERT_STREQ_LEN(written, buff, "-1.3");

    written = LIBC_NAMESPACE::sprintf(buff, "%.2g", -1.875);
    ASSERT_STREQ_LEN(written, buff, "-1.8");
  }

  // Flag Tests.
  written = LIBC_NAMESPACE::sprintf(buff, "%+g", 1.0);
  ASSERT_STREQ_LEN(written, buff, "+1");

  written = LIBC_NAMESPACE::sprintf(buff, "%+g", -1.0);
  ASSERT_STREQ_LEN(written, buff, "-1");

  written = LIBC_NAMESPACE::sprintf(buff, "% g", 1.0);
  ASSERT_STREQ_LEN(written, buff, " 1");

  written = LIBC_NAMESPACE::sprintf(buff, "% g", -1.0);
  ASSERT_STREQ_LEN(written, buff, "-1");

  written = LIBC_NAMESPACE::sprintf(buff, "%-15g", 1.5);
  ASSERT_STREQ_LEN(written, buff, "1.5            ");

  written = LIBC_NAMESPACE::sprintf(buff, "%#.g", 1.0);
  ASSERT_STREQ_LEN(written, buff, "1.");

  written = LIBC_NAMESPACE::sprintf(buff, "%#g", 1.0);
  ASSERT_STREQ_LEN(written, buff, "1.00000");

  written = LIBC_NAMESPACE::sprintf(buff, "%#.0g", 1.5);
  ASSERT_STREQ_LEN(written, buff, "2.");

  written = LIBC_NAMESPACE::sprintf(buff, "%015g", 1.5);
  ASSERT_STREQ_LEN(written, buff, "0000000000001.5");

  written = LIBC_NAMESPACE::sprintf(buff, "%015g", -1.5);
  ASSERT_STREQ_LEN(written, buff, "-000000000001.5");

  written = LIBC_NAMESPACE::sprintf(buff, "%+- #0g", 0.0);
  ASSERT_STREQ_LEN(written, buff, "+0.00000");

  // Combined Tests.

  written = LIBC_NAMESPACE::sprintf(buff, "%10.3g", 9.99);
  ASSERT_STREQ_LEN(written, buff, "      9.99");

  written = LIBC_NAMESPACE::sprintf(buff, "%10.2g", 9.99);
  ASSERT_STREQ_LEN(written, buff, "        10");

  written = LIBC_NAMESPACE::sprintf(buff, "%10.1g", 9.99);
  ASSERT_STREQ_LEN(written, buff, "     1e+01");

  written = LIBC_NAMESPACE::sprintf(buff, "%-10.3g", 9.99);
  ASSERT_STREQ_LEN(written, buff, "9.99      ");

  written = LIBC_NAMESPACE::sprintf(buff, "%-10.2g", 9.99);
  ASSERT_STREQ_LEN(written, buff, "10        ");

  written = LIBC_NAMESPACE::sprintf(buff, "%-10.1g", 9.99);
  ASSERT_STREQ_LEN(written, buff, "1e+01     ");

  written = LIBC_NAMESPACE::sprintf(buff, "%-10.1g", 1.0e-50);
  ASSERT_STREQ_LEN(written, buff, "1e-50     ");

  written = LIBC_NAMESPACE::sprintf(buff, "%30g", 1234567890123456789.0);
  ASSERT_STREQ_LEN(written, buff, "                   1.23457e+18");

  written = LIBC_NAMESPACE::sprintf(buff, "%-30g", 1234567890123456789.0);
  ASSERT_STREQ_LEN(written, buff, "1.23457e+18                   ");

  written = LIBC_NAMESPACE::sprintf(buff, "%25.15g", 9999999999999.99);
  ASSERT_STREQ_LEN(written, buff, "         9999999999999.99");

  written = LIBC_NAMESPACE::sprintf(buff, "%25.14g", 9999999999999.99);
  ASSERT_STREQ_LEN(written, buff, "           10000000000000");

  written = LIBC_NAMESPACE::sprintf(buff, "%25.13g", 9999999999999.99);
  ASSERT_STREQ_LEN(written, buff, "                    1e+13");

  written = LIBC_NAMESPACE::sprintf(buff, "%#12.3g %-12.3g", 0.1, 256.0);
  ASSERT_STREQ_LEN(written, buff, "       0.100 256         ");

  written = LIBC_NAMESPACE::sprintf(buff, "%+-#12.3g % 012.3g", 0.1256, 1256.0);
  ASSERT_STREQ_LEN(written, buff, "+0.126        0001.26e+03");
}

TEST_F(LlvmLibcSPrintfTest, FloatAutoLongDoubleConv) {
  ForceRoundingMode r(RoundingMode::Nearest);

  // Length Modifier Tests.

#if defined(LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80)

  written = LIBC_NAMESPACE::sprintf(buff, "%Lg", 0xf.fffffffffffffffp+16380L);
  ASSERT_STREQ_LEN(written, buff, "1.18973e+4932");

  written = LIBC_NAMESPACE::sprintf(buff, "%Lg", 0xa.aaaaaaaaaaaaaabp-7L);
  ASSERT_STREQ_LEN(written, buff, "0.0833333");

  written = LIBC_NAMESPACE::sprintf(buff, "%Lg", 9.99999999999e-100L);
  ASSERT_STREQ_LEN(written, buff, "1e-99");

#endif // LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80

  // TODO: Uncomment the below tests after long double support is added
  /*
  written = LIBC_NAMESPACE::sprintf(buff, "%Lf", 1e100L);
  ASSERT_STREQ_LEN(written, buff,
                   "99999999999999999996693535322073426194986990198284960792713"
                   "91541752018669482644324418977840117055488.000000");

  written = LIBC_NAMESPACE::sprintf(buff, "%Lf", 1.0L);
  ASSERT_STREQ_LEN(written, buff, "1.000000");

  char big_buff[10000];
  written = LIBC_NAMESPACE::sprintf(big_buff, "%Lf", 1e1000L);
  ASSERT_STREQ_LEN(
      written, big_buff,
      "999999999999999999973107317669562353428234857594552594925899449376328728"
      "202461036775511405481186963193066642191664822065529414252060696836533522"
      "387143501724276282079456797058697369889056407118642873669166717313763499"
      "277025985141177344925615052465165938514140943010597323750202561187880136"
      "174810574553749194614479541820148407958204853833697063267336294787191005"
      "628217462261955103745349844675732989944229689277833828743730290177882029"
      "042613704915899149603539993716885598351951895974316347947147507970269673"
      "097709017164643598452451201499004104341931127294141495501309305995449742"
      "273419524803597130450457553871345958049837885085168840317195672271085085"
      "950520957945970913451088104971436093671776829538796532762184174216651692"
      "640931965387852083906784898823494867055070322768919156031682291829761007"
      "101483799978382119231551218582499361996919560548090784230386907125151658"
      "086767207295524036170321059257942621398084478974000973622199163292708506"
      "2431457550909271560663602154947063707982236377366647567795879936."
      "000000");

  written = LIBC_NAMESPACE::sprintf(big_buff, "%Lf", 1e4900L);
  ASSERT_STREQ_LEN(
      written, big_buff,
      "100000000000000000002708312230690349833224052504078834346502930111959028"
      "517260692666637048230414374897655201843766090626319971729765251179632020"
      "313912652522792711197087872698264530532442630109549129842736280196919130"
      "242615101228133188193853826983121366159061148351354364472807590931218045"
      "387490935930967150336231085015126034696883068553581691802388371635128003"
      "615577299166097675723780877126495909902479233742826339471026068806070433"
      "075629449530819183550315434973800271862658869400009022028602967197463980"
      "126881829804282202449930132940824361207087494829502385835258094836304011"
      "876250359661206802659650567866176246063987902366800491980400341950657151"
      "370854446585517805253310195469184699955519312761482572080479702840420595"
      "377369017651259376039167277822106875560385309101650382998482652792335482"
      "865443482342801545877390859444282105890147577937366066315975231014810320"
      "888482059656248277607763361589359794524314002443575149260630989130103550"
      "443177966380769341050735632338583912575890190136462629316287947355057647"
      "111088565611192544631519843618778618820046304429723908484879583579178075"
      "456701368334212923379389029311286386996015804122917416008806233549005183"
      "152461084266176543129004016414959261473645240454289630182591200574019087"
      "358223489767381636349719510715487188747217311279465814538495924567014916"
      "238565628036285599497236493491668884212847699052761266207598941300449276"
      "447201387520841811835583254242213093566548778954711633721122784159793843"
      "766802019309395771984693609426401362800013936338891483689127845928572536"
      "790651156184721483511507878883282891696900630100211914227950790472211403"
      "392549466062537498185758854079775888444518306635752468713312357556380082"
      "275500658967283696421824354930077523691855699312544373220921962817907078"
      "445538421941800259027487429330768616490865438859612697367766323925013940"
      "918384858952407145253573823848733994146335416209309233074165707437420756"
      "438833918763109580759409985573826485055208965115587885226774453455112406"
      "581351429640282227888764449360534584421929291565334894907337572527922691"
      "473242328379737396430908523008687037407295838014450772162091496534584696"
      "605157436893236842602956298545594095307060870397506421786236892553632163"
      "491468601982681381011940409602294892199042638682530687578982576819839451"
      "907594697546439533559153604700750696252355362322662219852740143212566818"
      "745528402265116534684566273868361460640280523251242059850044328669692159"
      "629900374576027104298177006629276014371540945261309319363704125592775129"
      "543526908667388673739382491147471395192495459318806593271282662311169392"
      "196897003517840025298267505925987901751541005546610016067658227181318892"
      "914686508281007582655667597441346214499847364272258631922040641860333431"
      "409838623713258383681350233064164940590695888300919626215847587544298023"
      "636416943680102708406086295669759876682046839368574433996997648445207805"
      "615784339667691231286807666753972942872019850432610318031627872612657513"
      "588188267160616660825719678199868371370527508463011236193719286066916786"
      "169956541349011494927225747024994619057884118692213564790598702879596058"
      "672338334720925179141906809470606964896245458600635183723159228561689808"
      "246141482736625197373238197777325580142168245885279594913851700941789475"
      "252421784152262567254611571822468808675893407728003047921107885664474662"
      "930921581384003950729114103689170603748380178682003976896397305836815761"
      "717676338115866650889936516794601457549097578905329423919798362140648664"
      "569177147076571576101649257502509463877402424847669830852345415301684820"
      "395813946416649808062227494112874521812750160935760825922220707178083076"
      "380203450993589198835885505461509442443773367592842795410339065860781804"
      "024975272228687688301824830333940416256885455008512598774611538878683158"
      "183931461086893832255176926531299425504132104728730288984598001187854507"
      "900417184206801359847651992484444933900133130832052346600926424167009902"
      "829803553087005800387704758687923428053612864451456596148162238935900033"
      "917094683141205188616000211702577553792389670853917118547527592495253773"
      "028135298405566315903922235989614934474805789300370437580494193066066314"
      "056627605207631392651010580925826419831250810981343093764403877594495896"
      "516881097415880926429607388979497471571321217205535961262051641426436441"
      "668989765107456413733909427384182109285933511623871034309722437967253289"
      "084018145083721513211807496392673952789642893241520398827805325610653506"
      "029060153153064455898648607959013571280930834475689835845791849456112104"
      "462337569019001580859906425911782967213265389744605395555069797947978230"
      "708108432086217134763779632408473684293543722127232658767439906910370146"
      "716836295909075482355827087389127370874842532825987593970846704144140471"
      "956027276735614286138656432085771988513977140957180090146798065497158947"
      "229765733489703157617307078835099906185890777007500964162371428641176460"
      "739074789794941408428328217107759915202650066155868439585510978709442590"
      "231934194956788626761834746430104077432547436359522462253411168467463134"
      "24896.000000");
*/
  /*
    written = LIBC_NAMESPACE::sprintf(buff, "%La", 0.1L);
  #if defined(LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80)
    ASSERT_STREQ_LEN(written, buff, "0xc.ccccccccccccccdp-7");
  #elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT64)
    ASSERT_STREQ_LEN(written, buff, "0x1.999999999999ap-4");
  #elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT128)
    ASSERT_STREQ_LEN(written, buff, "0x1.999999999999999999999999999ap-4");
  #endif

    written = LIBC_NAMESPACE::sprintf(buff, "%La", 1.0e1000L);
  #if defined(LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80)
    ASSERT_STREQ_LEN(written, buff, "0xf.38db1f9dd3dac05p+3318");
  #elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT64)
    ASSERT_STREQ_LEN(written, buff, "inf");
  #elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT128)
    ASSERT_STREQ_LEN(written, buff, "0x1.e71b63f3ba7b580af1a52d2a7379p+3321");
  #endif

    written = LIBC_NAMESPACE::sprintf(buff, "%La", 1.0e-1000L);
  #if defined(LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80)
    ASSERT_STREQ_LEN(written, buff, "0x8.68a9188a89e1467p-3325");
  #elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT64)
    ASSERT_STREQ_LEN(written, buff, "0x0p+0");
  #elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT128)
    ASSERT_STREQ_LEN(written, buff, "0x1.0d152311513c28ce202627c06ec2p-3322");
  #endif
  */
}

#endif // LIBC_COPT_PRINTF_DISABLE_FLOAT

#if defined(LIBC_COMPILER_HAS_FIXED_POINT) &&                                  \
    !defined(LIBC_COPT_PRINTF_DISABLE_FIXED_POINT)
TEST_F(LlvmLibcSPrintfTest, FixedConv) {

  // These numeric tests are potentially a little weak, but the fuzz test is
  // more thorough than my handwritten tests tend to be.

  // TODO: Replace hex literals with their appropriate fixed point literals.

  written = LIBC_NAMESPACE::sprintf(buff, "%k", 0x0); // 0.0
  ASSERT_STREQ_LEN(written, buff, "0.000000");

  written = LIBC_NAMESPACE::sprintf(buff, "%k", 0x80000000); // -0.0
  ASSERT_STREQ_LEN(written, buff, "-0.000000");

  written = LIBC_NAMESPACE::sprintf(buff, "%r", 0xffff); // -fract max
  ASSERT_STREQ_LEN(written, buff, "-0.999969");

  written = LIBC_NAMESPACE::sprintf(buff, "%R", 0xffff); // unsigned fract max
  ASSERT_STREQ_LEN(written, buff, "0.999985");

  written = LIBC_NAMESPACE::sprintf(buff, "%k", 0xffffffff); // -accum max
  ASSERT_STREQ_LEN(written, buff, "-65535.999969");

  written =
      LIBC_NAMESPACE::sprintf(buff, "%K", 0xffffffff); // unsigned accum max
  ASSERT_STREQ_LEN(written, buff, "65535.999985");

  written = LIBC_NAMESPACE::sprintf(buff, "%r", 0x7fff); // fract max
  ASSERT_STREQ_LEN(written, buff, "0.999969");

  written = LIBC_NAMESPACE::sprintf(buff, "%k", 0x7fffffff); // accum max
  ASSERT_STREQ_LEN(written, buff, "65535.999969");

  // Length Modifier Tests.

  written = LIBC_NAMESPACE::sprintf(buff, "%hk", 0x0); // 0.0
  ASSERT_STREQ_LEN(written, buff, "0.000000");

  written = LIBC_NAMESPACE::sprintf(buff, "%hk", 0xffff); // -short accum max
  ASSERT_STREQ_LEN(written, buff, "-255.992188");

  written = LIBC_NAMESPACE::sprintf(buff, "%hr", 0x0); // 0.0
  ASSERT_STREQ_LEN(written, buff, "0.000000");

  written = LIBC_NAMESPACE::sprintf(buff, "%hr", 0xff); // -short fract max
  ASSERT_STREQ_LEN(written, buff, "-0.992188");

  written = LIBC_NAMESPACE::sprintf(buff, "%hK", 0x0); // 0.0
  ASSERT_STREQ_LEN(written, buff, "0.000000");

  written =
      LIBC_NAMESPACE::sprintf(buff, "%hK", 0xffff); // unsigned short accum max
  ASSERT_STREQ_LEN(written, buff, "255.996094");

  written = LIBC_NAMESPACE::sprintf(buff, "%hR", 0x0); // 0.0
  ASSERT_STREQ_LEN(written, buff, "0.000000");

  written =
      LIBC_NAMESPACE::sprintf(buff, "%hR", 0xff); // unsigned short fract max
  ASSERT_STREQ_LEN(written, buff, "0.996094");

  written = LIBC_NAMESPACE::sprintf(buff, "%lk", 0x0ll); // 0.0
  ASSERT_STREQ_LEN(written, buff, "0.000000");

  written = LIBC_NAMESPACE::sprintf(buff, "%lk",
                                    0xffffffffffffffff); //-long accum max
  ASSERT_STREQ_LEN(written, buff, "-4294967296.000000");

  written = LIBC_NAMESPACE::sprintf(buff, "%lr", 0x0); // 0.0
  ASSERT_STREQ_LEN(written, buff, "0.000000");

  written = LIBC_NAMESPACE::sprintf(buff, "%lr",
                                    0xffffffff); //-long fract max
  ASSERT_STREQ_LEN(written, buff, "-1.000000");

  written = LIBC_NAMESPACE::sprintf(buff, "%lK", 0x0ll); // 0.0
  ASSERT_STREQ_LEN(written, buff, "0.000000");

  written =
      LIBC_NAMESPACE::sprintf(buff, "%lK",
                              0xffffffffffffffff); // unsigned long accum max
  ASSERT_STREQ_LEN(written, buff, "4294967296.000000");

  written = LIBC_NAMESPACE::sprintf(buff, "%lR", 0x0); // 0.0
  ASSERT_STREQ_LEN(written, buff, "0.000000");

  written = LIBC_NAMESPACE::sprintf(buff, "%lR",
                                    0xffffffff); // unsigned long fract max
  ASSERT_STREQ_LEN(written, buff, "1.000000");

  // Min Width Tests.

  written = LIBC_NAMESPACE::sprintf(buff, "%10k", 0x0000a000); // 1.25
  ASSERT_STREQ_LEN(written, buff, "  1.250000");

  written = LIBC_NAMESPACE::sprintf(buff, "%10k", 0x8000a000); //-1.25
  ASSERT_STREQ_LEN(written, buff, " -1.250000");

  written = LIBC_NAMESPACE::sprintf(buff, "%8k", 0x0000a000); // 1.25
  ASSERT_STREQ_LEN(written, buff, "1.250000");

  written = LIBC_NAMESPACE::sprintf(buff, "%9k", 0x8000a000); //-1.25
  ASSERT_STREQ_LEN(written, buff, "-1.250000");

  written = LIBC_NAMESPACE::sprintf(buff, "%4k", 0x0000a000); // 1.25
  ASSERT_STREQ_LEN(written, buff, "1.250000");

  written = LIBC_NAMESPACE::sprintf(buff, "%4k", 0x8000a000); //-1.25
  ASSERT_STREQ_LEN(written, buff, "-1.250000");

  // Precision Tests.

  written =
      LIBC_NAMESPACE::sprintf(buff, "%.16K", 0xFFFFFFFF); // unsigned accum max
  ASSERT_STREQ_LEN(written, buff, "65535.9999847412109375");

  written = LIBC_NAMESPACE::sprintf(
      buff, "%.32lK", 0xFFFFFFFFFFFFFFFF); // unsigned long accum max
  ASSERT_STREQ_LEN(written, buff,
                   "4294967295.99999999976716935634613037109375");

  written =
      LIBC_NAMESPACE::sprintf(buff, "%.0K", 0xFFFFFFFF); // unsigned accum max
  ASSERT_STREQ_LEN(written, buff, "65536");

  written = LIBC_NAMESPACE::sprintf(buff, "%.0R", 0xFFFF); // unsigned fract max
  ASSERT_STREQ_LEN(written, buff, "1");

  // Flag Tests.

  written = LIBC_NAMESPACE::sprintf(buff, "%+k", 0x0000a000); // 1.25
  ASSERT_STREQ_LEN(written, buff, "+1.250000");

  written = LIBC_NAMESPACE::sprintf(buff, "%+k", 0x8000a000); //-1.25
  ASSERT_STREQ_LEN(written, buff, "-1.250000");

  written = LIBC_NAMESPACE::sprintf(buff, "% k", 0x0000a000); // 1.25
  ASSERT_STREQ_LEN(written, buff, " 1.250000");

  written = LIBC_NAMESPACE::sprintf(buff, "% k", 0x8000a000); //-1.25
  ASSERT_STREQ_LEN(written, buff, "-1.250000");

  // unsigned variants ignore sign flags.
  written = LIBC_NAMESPACE::sprintf(buff, "%+K", 0x00014000); // 1.25
  ASSERT_STREQ_LEN(written, buff, "1.250000");

  written = LIBC_NAMESPACE::sprintf(buff, "% K", 0x00014000); // 1.25
  ASSERT_STREQ_LEN(written, buff, "1.250000");

  written = LIBC_NAMESPACE::sprintf(buff, "%-10k", 0x0000c000); // 1.5
  ASSERT_STREQ_LEN(written, buff, "1.500000  ");

  written = LIBC_NAMESPACE::sprintf(buff, "%#.k", 0x00008000); // 1.0
  ASSERT_STREQ_LEN(written, buff, "1.");

  written = LIBC_NAMESPACE::sprintf(buff, "%#.0k", 0x0000c000); // 1.5
  ASSERT_STREQ_LEN(written, buff, "2.");

  written = LIBC_NAMESPACE::sprintf(buff, "%010k", 0x0000c000); // 1.5
  ASSERT_STREQ_LEN(written, buff, "001.500000");

  written = LIBC_NAMESPACE::sprintf(buff, "%010k", 0x8000c000); //-1.5
  ASSERT_STREQ_LEN(written, buff, "-01.500000");

  written = LIBC_NAMESPACE::sprintf(buff, "%+- #0k", 0); // 0.0
  ASSERT_STREQ_LEN(written, buff, "+0.000000");

  // Combined Tests.

  written = LIBC_NAMESPACE::sprintf(buff, "%10.2k", 0x0004feb8); // 9.99
  ASSERT_STREQ_LEN(written, buff, "      9.99");

  written = LIBC_NAMESPACE::sprintf(buff, "%5.1k", 0x0004feb8); // 9.99
  ASSERT_STREQ_LEN(written, buff, " 10.0");

  written = LIBC_NAMESPACE::sprintf(buff, "%-10.2k", 0x0004feb8); // 9.99
  ASSERT_STREQ_LEN(written, buff, "9.99      ");

  written = LIBC_NAMESPACE::sprintf(buff, "%-5.1k", 0x0004feb8); // 9.99
  ASSERT_STREQ_LEN(written, buff, "10.0 ");

  written = LIBC_NAMESPACE::sprintf(buff, "%-5.1k", 0x00000001); // accum min
  ASSERT_STREQ_LEN(written, buff, "0.0  ");

  written = LIBC_NAMESPACE::sprintf(buff, "%30k", 0x7fffffff); // accum max
  ASSERT_STREQ_LEN(written, buff, "                  65535.999969");

  written = LIBC_NAMESPACE::sprintf(buff, "%-30k", 0x7fffffff); // accum max
  ASSERT_STREQ_LEN(written, buff, "65535.999969                  ");

  written = LIBC_NAMESPACE::sprintf(buff, "%20.2lK",
                                    0x3b9ac9ffFD70A3D7); // 999999999.99
  ASSERT_STREQ_LEN(written, buff, "        999999999.99");

  written = LIBC_NAMESPACE::sprintf(buff, "%20.1lK",
                                    0x3b9ac9ffFD70A3D7); // 999999999.99
  ASSERT_STREQ_LEN(written, buff, "        1000000000.0");

  written = LIBC_NAMESPACE::sprintf(buff, "%12.3R %-12.3k", 0x1999,
                                    0x00800000); // 0.1, 256.0
  ASSERT_STREQ_LEN(written, buff, "       0.100 256.000     ");

  written =
      LIBC_NAMESPACE::sprintf(buff, "%+-#12.3lk % 012.3k", 0x000000001013a92all,
                              0x02740000); // 0.126, 1256.0
  ASSERT_STREQ_LEN(written, buff, "+0.126        0001256.000");
}
#endif // defined(LIBC_COMPILER_HAS_FIXED_POINT) &&
       // !defined(LIBC_COPT_PRINTF_DISABLE_FIXED_POINT)

#ifndef LIBC_COPT_PRINTF_DISABLE_STRERROR
TEST_F(LlvmLibcSPrintfTest, StrerrorConv) {
  LIBC_NAMESPACE::libc_errno = 0;
  written = LIBC_NAMESPACE::sprintf(buff, "%m");
  ASSERT_STREQ_LEN(written, buff, "Success");

  LIBC_NAMESPACE::libc_errno = ERANGE;
  written = LIBC_NAMESPACE::sprintf(buff, "%m");
  ASSERT_STREQ_LEN(written, buff, "Numerical result out of range");

  // Check that it correctly consumes no arguments.
  LIBC_NAMESPACE::libc_errno = 0;
  written = LIBC_NAMESPACE::sprintf(buff, "%m %d", 1);
  ASSERT_STREQ_LEN(written, buff, "Success 1");

  // Width Tests

  LIBC_NAMESPACE::libc_errno = 0;
  written = LIBC_NAMESPACE::sprintf(buff, "%10m");
  ASSERT_STREQ_LEN(written, buff, "   Success");

  LIBC_NAMESPACE::libc_errno = ERANGE;
  written = LIBC_NAMESPACE::sprintf(buff, "%10m");
  ASSERT_STREQ_LEN(written, buff, "Numerical result out of range");

  // Precision Tests

  LIBC_NAMESPACE::libc_errno = 0;
  written = LIBC_NAMESPACE::sprintf(buff, "%.10m");
  ASSERT_STREQ_LEN(written, buff, "Success");

  LIBC_NAMESPACE::libc_errno = ERANGE;
  written = LIBC_NAMESPACE::sprintf(buff, "%.10m");
  ASSERT_STREQ_LEN(written, buff, "Numerical ");

  // Flag Tests (Only '-' since the others only affect ints)

  LIBC_NAMESPACE::libc_errno = 0;
  written = LIBC_NAMESPACE::sprintf(buff, "%-10m");
  ASSERT_STREQ_LEN(written, buff, "Success   ");

  LIBC_NAMESPACE::libc_errno = ERANGE;
  written = LIBC_NAMESPACE::sprintf(buff, "%-10m");
  ASSERT_STREQ_LEN(written, buff, "Numerical result out of range");

  // Alt Mode Tests
  // Since alt mode here is effectively a completely separate conversion, it
  // gets separate tests.

  LIBC_NAMESPACE::libc_errno = 0;
  written = LIBC_NAMESPACE::sprintf(buff, "%#m");
  ASSERT_STREQ_LEN(written, buff, "0");

  LIBC_NAMESPACE::libc_errno = ERANGE;
  written = LIBC_NAMESPACE::sprintf(buff, "%#m");
  ASSERT_STREQ_LEN(written, buff, "ERANGE");

  LIBC_NAMESPACE::libc_errno = -9999;
  written = LIBC_NAMESPACE::sprintf(buff, "%#m");
  ASSERT_STREQ_LEN(written, buff, "-9999");

  // Alt Mode Width

  LIBC_NAMESPACE::libc_errno = 0;
  written = LIBC_NAMESPACE::sprintf(buff, "%#10m");
  ASSERT_STREQ_LEN(written, buff, "         0");

  LIBC_NAMESPACE::libc_errno = ERANGE;
  written = LIBC_NAMESPACE::sprintf(buff, "%#10m");
  ASSERT_STREQ_LEN(written, buff, "    ERANGE");

  LIBC_NAMESPACE::libc_errno = -9999;
  written = LIBC_NAMESPACE::sprintf(buff, "%#10m");
  ASSERT_STREQ_LEN(written, buff, "     -9999");

  LIBC_NAMESPACE::libc_errno = ERANGE;
  written = LIBC_NAMESPACE::sprintf(buff, "%#3m");
  ASSERT_STREQ_LEN(written, buff, "ERANGE");

  LIBC_NAMESPACE::libc_errno = -9999;
  written = LIBC_NAMESPACE::sprintf(buff, "%#3m");
  ASSERT_STREQ_LEN(written, buff, "-9999");

  // Alt Mode Precision

  LIBC_NAMESPACE::libc_errno = ERANGE;
  written = LIBC_NAMESPACE::sprintf(buff, "%#.10m");
  ASSERT_STREQ_LEN(written, buff, "ERANGE");

  LIBC_NAMESPACE::libc_errno = -9999;
  written = LIBC_NAMESPACE::sprintf(buff, "%#.10m");
  ASSERT_STREQ_LEN(written, buff, "-0000009999");

  LIBC_NAMESPACE::libc_errno = ERANGE;
  written = LIBC_NAMESPACE::sprintf(buff, "%#.3m");
  ASSERT_STREQ_LEN(written, buff, "ERA");

  LIBC_NAMESPACE::libc_errno = -9999;
  written = LIBC_NAMESPACE::sprintf(buff, "%#.3m");
  ASSERT_STREQ_LEN(written, buff, "-9999");

  // We don't test precision (or int flags) on errno = 0 because it behaves
  // weirdly, see the docs for more information.
  LIBC_NAMESPACE::libc_errno = 0;
  written = LIBC_NAMESPACE::sprintf(buff, "%#.1m");
  ASSERT_STREQ_LEN(written, buff, "0");

  // Alt Mode Flags

  // '-' flag
  LIBC_NAMESPACE::libc_errno = 0;
  written = LIBC_NAMESPACE::sprintf(buff, "%#-10m");
  ASSERT_STREQ_LEN(written, buff, "0         ");

  LIBC_NAMESPACE::libc_errno = ERANGE;
  written = LIBC_NAMESPACE::sprintf(buff, "%#-10m");
  ASSERT_STREQ_LEN(written, buff, "ERANGE    ");

  LIBC_NAMESPACE::libc_errno = -9999;
  written = LIBC_NAMESPACE::sprintf(buff, "%#-10m");
  ASSERT_STREQ_LEN(written, buff, "-9999     ");

  LIBC_NAMESPACE::libc_errno = ERANGE;
  written = LIBC_NAMESPACE::sprintf(buff, "%#-3m");
  ASSERT_STREQ_LEN(written, buff, "ERANGE");

  LIBC_NAMESPACE::libc_errno = -9999;
  written = LIBC_NAMESPACE::sprintf(buff, "%#-3m");
  ASSERT_STREQ_LEN(written, buff, "-9999");

  // '+' flag
  LIBC_NAMESPACE::libc_errno = ERANGE;
  written = LIBC_NAMESPACE::sprintf(buff, "%#+m");
  ASSERT_STREQ_LEN(written, buff, "ERANGE");

  LIBC_NAMESPACE::libc_errno = -9999;
  written = LIBC_NAMESPACE::sprintf(buff, "%#+m");
  ASSERT_STREQ_LEN(written, buff, "-9999");

  // Technically 9999 could be a valid error, since the standard just says errno
  // macros are "distinct positive values". In practice I don't expect this to
  // come up, but I've avoided it for the other %m tests for ease of
  // refactoring if necessary. Here it needs to be positive to test that the
  // flags that only affect positive signed integers are properly passed along.
  LIBC_NAMESPACE::libc_errno = 9999;
  written = LIBC_NAMESPACE::sprintf(buff, "%#+m");
  ASSERT_STREQ_LEN(written, buff, "+9999");

  // ' ' flag
  LIBC_NAMESPACE::libc_errno = ERANGE;
  written = LIBC_NAMESPACE::sprintf(buff, "%# m");
  ASSERT_STREQ_LEN(written, buff, "ERANGE");

  LIBC_NAMESPACE::libc_errno = -9999;
  written = LIBC_NAMESPACE::sprintf(buff, "%# m");
  ASSERT_STREQ_LEN(written, buff, "-9999");

  LIBC_NAMESPACE::libc_errno = 9999;
  written = LIBC_NAMESPACE::sprintf(buff, "%# m");
  ASSERT_STREQ_LEN(written, buff, " 9999");

  // '0' flag

  LIBC_NAMESPACE::libc_errno = ERANGE;
  written = LIBC_NAMESPACE::sprintf(buff, "%#010m");
  ASSERT_STREQ_LEN(written, buff, "    ERANGE");

  LIBC_NAMESPACE::libc_errno = -9999;
  written = LIBC_NAMESPACE::sprintf(buff, "%#010m");
  ASSERT_STREQ_LEN(written, buff, "-000009999");

  LIBC_NAMESPACE::libc_errno = ERANGE;
  written = LIBC_NAMESPACE::sprintf(buff, "%#03m");
  ASSERT_STREQ_LEN(written, buff, "ERANGE");

  LIBC_NAMESPACE::libc_errno = -9999;
  written = LIBC_NAMESPACE::sprintf(buff, "%#03m");
  ASSERT_STREQ_LEN(written, buff, "-9999");
}
#endif // LIBC_COPT_PRINTF_DISABLE_STRERROR

#ifndef LIBC_COPT_PRINTF_DISABLE_WRITE_INT
TEST(LlvmLibcSPrintfTest, WriteIntConv) {
  char buff[64];
  int written;
  int test_val = -1;

  test_val = -1;
  written = LIBC_NAMESPACE::sprintf(buff, "12345%n67890", &test_val);
  EXPECT_EQ(written, 10);
  EXPECT_EQ(test_val, 5);
  ASSERT_STREQ(buff, "1234567890");

  test_val = -1;
  written = LIBC_NAMESPACE::sprintf(buff, "%n", &test_val);
  EXPECT_EQ(written, 0);
  EXPECT_EQ(test_val, 0);
  ASSERT_STREQ(buff, "");

  test_val = 0x100;
  written = LIBC_NAMESPACE::sprintf(buff, "ABC%hhnDEF", &test_val);
  EXPECT_EQ(written, 6);
  EXPECT_EQ(test_val, 0x103);
  ASSERT_STREQ(buff, "ABCDEF");

  test_val = -1;
  written = LIBC_NAMESPACE::sprintf(buff, "%s%n", "87654321", &test_val);
  EXPECT_EQ(written, 8);
  EXPECT_EQ(test_val, 8);
  ASSERT_STREQ(buff, "87654321");

#ifndef LIBC_COPT_PRINTF_NO_NULLPTR_CHECKS
  written = LIBC_NAMESPACE::sprintf(buff, "abc123%n", nullptr);
  EXPECT_LT(written, 0);
#endif // LIBC_COPT_PRINTF_NO_NULLPTR_CHECKS
}
#endif // LIBC_COPT_PRINTF_DISABLE_WRITE_INT

#ifndef LIBC_COPT_PRINTF_DISABLE_INDEX_MODE
TEST(LlvmLibcSPrintfTest, IndexModeParsing) {
  char buff[64];
  int written;

  written = LIBC_NAMESPACE::sprintf(buff, "%1$s", "abcDEF123");
  EXPECT_EQ(written, 9);
  ASSERT_STREQ(buff, "abcDEF123");

  written = LIBC_NAMESPACE::sprintf(buff, "%1$s %%", "abcDEF123");
  EXPECT_EQ(written, 11);
  ASSERT_STREQ(buff, "abcDEF123 %");

  written =
      LIBC_NAMESPACE::sprintf(buff, "%3$s %1$s %2$s", "is", "hard", "ordering");
  EXPECT_EQ(written, 16);
  ASSERT_STREQ(buff, "ordering is hard");

  written = LIBC_NAMESPACE::sprintf(
      buff, "%10$s %9$s %8$c %7$s %6$s, %6$s %5$s %4$-*1$s %3$.*11$s %2$s. %%",
      6, "pain", "alphabetical", "such", "is", "this", "do", 'u', "would",
      "why", 1);
  EXPECT_EQ(written, 45);
  ASSERT_STREQ(buff, "why would u do this, this is such   a pain. %");
}
#endif // LIBC_COPT_PRINTF_DISABLE_INDEX_MODE

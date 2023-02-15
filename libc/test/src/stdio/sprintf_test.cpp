//===-- Unittests for sprintf ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/sprintf.h"

#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/PlatformDefs.h"
#include "test/UnitTest/Test.h"
#include "utils/testutils/RoundingModeUtils.h"

// #include <stdio.h>
// namespace __llvm_libc {
// using ::sprintf;
// }

class LlvmLibcSPrintfTest : public __llvm_libc::testing::Test {
protected:
  char buff[1000];
  int written;
};

// Subtract 1 from sizeof(expected_str) to account for the null byte.
#define ASSERT_STREQ_LEN(actual_written, actual_str, expected_str)             \
  EXPECT_EQ(actual_written, static_cast<int>(sizeof(expected_str) - 1));       \
  EXPECT_STREQ(actual_str, expected_str);

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

  written = __llvm_libc::sprintf(buff, "%hhu", 257); // 0x101
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

TEST(LlvmLibcSPrintfTest, HexConv) {
  char buff[64];
  int written;

  // Basic Tests.

  written = __llvm_libc::sprintf(buff, "%x", 0x123a);
  EXPECT_EQ(written, 4);
  ASSERT_STREQ(buff, "123a");

  written = __llvm_libc::sprintf(buff, "%X", 0x456b);
  EXPECT_EQ(written, 4);
  ASSERT_STREQ(buff, "456B");

  // Length Modifier Tests.

  written = __llvm_libc::sprintf(buff, "%hhx", 0x10001);
  EXPECT_EQ(written, 1);
  ASSERT_STREQ(buff, "1");

  written = __llvm_libc::sprintf(buff, "%llx", 0xffffffffffffffffull);
  EXPECT_EQ(written, 16);
  ASSERT_STREQ(buff, "ffffffffffffffff"); // ull max

  written = __llvm_libc::sprintf(buff, "%tX", ~ptrdiff_t(0));
  if (sizeof(ptrdiff_t) == 8) {
    EXPECT_EQ(written, 16);
    ASSERT_STREQ(buff, "FFFFFFFFFFFFFFFF");
  } else if (sizeof(ptrdiff_t) == 4) {
    EXPECT_EQ(written, 8);
    ASSERT_STREQ(buff, "FFFFFFFF");
  }

  // Min Width Tests.

  written = __llvm_libc::sprintf(buff, "%4x", 0x789);
  EXPECT_EQ(written, 4);
  ASSERT_STREQ(buff, " 789");

  written = __llvm_libc::sprintf(buff, "%2X", 0x987);
  EXPECT_EQ(written, 3);
  ASSERT_STREQ(buff, "987");

  // Precision Tests.

  written = __llvm_libc::sprintf(buff, "%x", 0);
  EXPECT_EQ(written, 1);
  ASSERT_STREQ(buff, "0");

  written = __llvm_libc::sprintf(buff, "%.0x", 0);
  EXPECT_EQ(written, 0);
  ASSERT_STREQ(buff, "");

  written = __llvm_libc::sprintf(buff, "%.5x", 0x1F3);
  EXPECT_EQ(written, 5);
  ASSERT_STREQ(buff, "001f3");

  written = __llvm_libc::sprintf(buff, "%.2x", 0x135);
  EXPECT_EQ(written, 3);
  ASSERT_STREQ(buff, "135");

  // Flag Tests.

  written = __llvm_libc::sprintf(buff, "%-5x", 0x246);
  EXPECT_EQ(written, 5);
  ASSERT_STREQ(buff, "246  ");

  written = __llvm_libc::sprintf(buff, "%#x", 0xd3f);
  EXPECT_EQ(written, 5);
  ASSERT_STREQ(buff, "0xd3f");

  written = __llvm_libc::sprintf(buff, "%#X", 0xE40);
  EXPECT_EQ(written, 5);
  ASSERT_STREQ(buff, "0XE40");

  written = __llvm_libc::sprintf(buff, "%05x", 0x470);
  EXPECT_EQ(written, 5);
  ASSERT_STREQ(buff, "00470");

  written = __llvm_libc::sprintf(buff, "%0#6x", 0x8c3);
  EXPECT_EQ(written, 6);
  ASSERT_STREQ(buff, "0x08c3");

  written = __llvm_libc::sprintf(buff, "%-#6x", 0x5f0);
  EXPECT_EQ(written, 6);
  ASSERT_STREQ(buff, "0x5f0 ");

  // Combined Tests.

  written = __llvm_libc::sprintf(buff, "%#-07x", 0x703);
  EXPECT_EQ(written, 7);
  ASSERT_STREQ(buff, "0x703  ");

  written = __llvm_libc::sprintf(buff, "%7.5x", 0x814);
  EXPECT_EQ(written, 7);
  ASSERT_STREQ(buff, "  00814");

  written = __llvm_libc::sprintf(buff, "%#9.5X", 0x9d4);
  EXPECT_EQ(written, 9);
  ASSERT_STREQ(buff, "  0X009D4");

  written = __llvm_libc::sprintf(buff, "%-7.5x", 0x260);
  EXPECT_EQ(written, 7);
  ASSERT_STREQ(buff, "00260  ");

  written = __llvm_libc::sprintf(buff, "%5.4x", 0x10000);
  EXPECT_EQ(written, 5);
  ASSERT_STREQ(buff, "10000");

  // Multiple Conversion Tests.

  written = __llvm_libc::sprintf(buff, "%10X %-#10x", 0x45b, 0x789);
  EXPECT_EQ(written, 21);
  ASSERT_STREQ(buff, "       45B 0x789     ");

  written = __llvm_libc::sprintf(buff, "%-5.4x%#.4x", 0x75, 0x25);
  EXPECT_EQ(written, 11);
  ASSERT_STREQ(buff, "0075 0x0025");

  written = __llvm_libc::sprintf(buff, "%04hhX %#.5llx %-6.3zX", 256 + 0x7f,
                                 0x1000000000ll, size_t(2));
  EXPECT_EQ(written, 24);
  ASSERT_STREQ(buff, "007F 0x1000000000 002   ");
}

TEST(LlvmLibcSPrintfTest, PointerConv) {
  char buff[64];
  int written;

  written = __llvm_libc::sprintf(buff, "%p", nullptr);
  EXPECT_EQ(written, 9);
  ASSERT_STREQ(buff, "(nullptr)");

  written = __llvm_libc::sprintf(buff, "%p", 0x1a2b3c4d);
  EXPECT_EQ(written, 10);
  ASSERT_STREQ(buff, "0x1a2b3c4d");

  written = __llvm_libc::sprintf(buff, "%p", buff);
  EXPECT_GT(written, 0);
}

TEST(LlvmLibcSPrintfTest, OctConv) {
  char buff[64];
  int written;

  // Basic Tests.

  written = __llvm_libc::sprintf(buff, "%o", 01234);
  EXPECT_EQ(written, 4);
  ASSERT_STREQ(buff, "1234");

  written = __llvm_libc::sprintf(buff, "%o", 04567);
  EXPECT_EQ(written, 4);
  ASSERT_STREQ(buff, "4567");

  // Length Modifier Tests.

  written = __llvm_libc::sprintf(buff, "%hho", 0401);
  EXPECT_EQ(written, 1);
  ASSERT_STREQ(buff, "1");

  written = __llvm_libc::sprintf(buff, "%llo", 01777777777777777777777ull);
  EXPECT_EQ(written, 22);
  ASSERT_STREQ(buff, "1777777777777777777777"); // ull max

  written = __llvm_libc::sprintf(buff, "%to", ~ptrdiff_t(0));
  if (sizeof(ptrdiff_t) == 8) {
    EXPECT_EQ(written, 22);
    ASSERT_STREQ(buff, "1777777777777777777777");
  } else if (sizeof(ptrdiff_t) == 4) {
    EXPECT_EQ(written, 11);
    ASSERT_STREQ(buff, "37777777777");
  }

  // Min Width Tests.

  written = __llvm_libc::sprintf(buff, "%4o", 0701);
  EXPECT_EQ(written, 4);
  ASSERT_STREQ(buff, " 701");

  written = __llvm_libc::sprintf(buff, "%2o", 0107);
  EXPECT_EQ(written, 3);
  ASSERT_STREQ(buff, "107");

  // Precision Tests.

  written = __llvm_libc::sprintf(buff, "%o", 0);
  EXPECT_EQ(written, 1);
  ASSERT_STREQ(buff, "0");

  written = __llvm_libc::sprintf(buff, "%.0o", 0);
  EXPECT_EQ(written, 0);
  ASSERT_STREQ(buff, "");

  written = __llvm_libc::sprintf(buff, "%.5o", 0153);
  EXPECT_EQ(written, 5);
  ASSERT_STREQ(buff, "00153");

  written = __llvm_libc::sprintf(buff, "%.2o", 0135);
  EXPECT_EQ(written, 3);
  ASSERT_STREQ(buff, "135");

  // Flag Tests.

  written = __llvm_libc::sprintf(buff, "%-5o", 0246);
  EXPECT_EQ(written, 5);
  ASSERT_STREQ(buff, "246  ");

  written = __llvm_libc::sprintf(buff, "%#o", 0234);
  EXPECT_EQ(written, 4);
  ASSERT_STREQ(buff, "0234");

  written = __llvm_libc::sprintf(buff, "%05o", 0470);
  EXPECT_EQ(written, 5);
  ASSERT_STREQ(buff, "00470");

  written = __llvm_libc::sprintf(buff, "%0#6o", 0753);
  EXPECT_EQ(written, 6);
  ASSERT_STREQ(buff, "000753");

  written = __llvm_libc::sprintf(buff, "%-#6o", 0642);
  EXPECT_EQ(written, 6);
  ASSERT_STREQ(buff, "0642  ");

  // Combined Tests.

  written = __llvm_libc::sprintf(buff, "%#-07o", 0703);
  EXPECT_EQ(written, 7);
  ASSERT_STREQ(buff, "0703   ");

  written = __llvm_libc::sprintf(buff, "%7.5o", 0314);
  EXPECT_EQ(written, 7);
  ASSERT_STREQ(buff, "  00314");

  written = __llvm_libc::sprintf(buff, "%#9.5o", 0234);
  EXPECT_EQ(written, 9);
  ASSERT_STREQ(buff, "    00234");

  written = __llvm_libc::sprintf(buff, "%-7.5o", 0260);
  EXPECT_EQ(written, 7);
  ASSERT_STREQ(buff, "00260  ");

  written = __llvm_libc::sprintf(buff, "%5.4o", 010000);
  EXPECT_EQ(written, 5);
  ASSERT_STREQ(buff, "10000");

  // Multiple Conversion Tests.

  written = __llvm_libc::sprintf(buff, "%10o %-#10o", 0456, 0123);
  EXPECT_EQ(written, 21);
  ASSERT_STREQ(buff, "       456 0123      ");

  written = __llvm_libc::sprintf(buff, "%-5.4o%#.4o", 075, 025);
  EXPECT_EQ(written, 9);
  ASSERT_STREQ(buff, "0075 0025");

  written = __llvm_libc::sprintf(buff, "%04hho %#.5llo %-6.3zo", 256 + 077,
                                 01000000000000ll, size_t(2));
  EXPECT_EQ(written, 26);
  ASSERT_STREQ(buff, "0077 01000000000000 002   ");
}

#ifndef LIBC_COPT_PRINTF_DISABLE_FLOAT

TEST_F(LlvmLibcSPrintfTest, FloatHexExpConv) {
  __llvm_libc::testutils::ForceRoundingMode r(
      __llvm_libc::testutils::RoundingMode::Nearest);
  double inf = __llvm_libc::fputil::FPBits<double>::inf().get_val();
  double nan = __llvm_libc::fputil::FPBits<double>::build_nan(1);

  written = __llvm_libc::sprintf(buff, "%a", 1.0);
  ASSERT_STREQ_LEN(written, buff, "0x1p+0");

  written = __llvm_libc::sprintf(buff, "%A", -1.0);
  ASSERT_STREQ_LEN(written, buff, "-0X1P+0");

  written = __llvm_libc::sprintf(buff, "%a", -0x1.abcdef12345p0);
  ASSERT_STREQ_LEN(written, buff, "-0x1.abcdef12345p+0");

  written = __llvm_libc::sprintf(buff, "%A", 0x1.abcdef12345p0);
  ASSERT_STREQ_LEN(written, buff, "0X1.ABCDEF12345P+0");

  written = __llvm_libc::sprintf(buff, "%a", 0.0);
  ASSERT_STREQ_LEN(written, buff, "0x0p+0");

  written = __llvm_libc::sprintf(buff, "%a", 1.0e100);
  ASSERT_STREQ_LEN(written, buff, "0x1.249ad2594c37dp+332");

  written = __llvm_libc::sprintf(buff, "%a", 0.1);
  ASSERT_STREQ_LEN(written, buff, "0x1.999999999999ap-4");

  // Subnormal Tests.

  written = __llvm_libc::sprintf(buff, "%a", 0x1.0p-1027);
  ASSERT_STREQ_LEN(written, buff, "0x0.08p-1022");

  written = __llvm_libc::sprintf(buff, "%a", 0x1.0p-1025);
  ASSERT_STREQ_LEN(written, buff, "0x0.2p-1022");

  written = __llvm_libc::sprintf(buff, "%a", 0x1.0p-1023);
  ASSERT_STREQ_LEN(written, buff, "0x0.8p-1022");

  written = __llvm_libc::sprintf(buff, "%a", 0x1.0p-1022);
  ASSERT_STREQ_LEN(written, buff, "0x1p-1022");

  written = __llvm_libc::sprintf(buff, "%a", 0x1.0p-1074);
  ASSERT_STREQ_LEN(written, buff, "0x0.0000000000001p-1022");

  // Inf/Nan Tests.

  written = __llvm_libc::sprintf(buff, "%a", inf);
  ASSERT_STREQ_LEN(written, buff, "inf");

  written = __llvm_libc::sprintf(buff, "%A", -inf);
  ASSERT_STREQ_LEN(written, buff, "-INF");

  written = __llvm_libc::sprintf(buff, "%a", nan);
  ASSERT_STREQ_LEN(written, buff, "nan");

  written = __llvm_libc::sprintf(buff, "%A", -nan);
  ASSERT_STREQ_LEN(written, buff, "-NAN");

  // Length Modifier Tests.

  written = __llvm_libc::sprintf(buff, "%La", 0.1L);
#if defined(SPECIAL_X86_LONG_DOUBLE)
  ASSERT_STREQ_LEN(written, buff, "0xc.ccccccccccccccdp-7");
#elif defined(LONG_DOUBLE_IS_DOUBLE)
  ASSERT_STREQ_LEN(written, buff, "0x1.999999999999ap-4");
#else // 128 bit long double
  ASSERT_STREQ_LEN(written, buff, "0x1.999999999999999999999999999ap-4");
#endif

  written = __llvm_libc::sprintf(buff, "%La", 1.0e1000L);
#if defined(SPECIAL_X86_LONG_DOUBLE)
  ASSERT_STREQ_LEN(written, buff, "0xf.38db1f9dd3dac05p+3318");
#elif defined(LONG_DOUBLE_IS_DOUBLE)
  ASSERT_STREQ_LEN(written, buff, "inf");
#else // 128 bit long double
  ASSERT_STREQ_LEN(written, buff, "0x1.e71b63f3ba7b580af1a52d2a7379p+3321");
#endif

  written = __llvm_libc::sprintf(buff, "%La", 1.0e-1000L);
#if defined(SPECIAL_X86_LONG_DOUBLE)
  ASSERT_STREQ_LEN(written, buff, "0x8.68a9188a89e1467p-3325");
#elif defined(LONG_DOUBLE_IS_DOUBLE)
  ASSERT_STREQ_LEN(written, buff, "0x0p+0");
#else // 128 bit long double
  ASSERT_STREQ_LEN(written, buff, "0x1.0d152311513c28ce202627c06ec2p-3322");
#endif

  // Min Width Tests.

  written = __llvm_libc::sprintf(buff, "%15a", 1.0);
  ASSERT_STREQ_LEN(written, buff, "         0x1p+0");

  written = __llvm_libc::sprintf(buff, "%15a", -1.0);
  ASSERT_STREQ_LEN(written, buff, "        -0x1p+0");

  written = __llvm_libc::sprintf(buff, "%15a", 1.0e10);
  ASSERT_STREQ_LEN(written, buff, " 0x1.2a05f2p+33");

  written = __llvm_libc::sprintf(buff, "%15a", -1.0e10);
  ASSERT_STREQ_LEN(written, buff, "-0x1.2a05f2p+33");

  written = __llvm_libc::sprintf(buff, "%10a", 1.0e10);
  ASSERT_STREQ_LEN(written, buff, "0x1.2a05f2p+33");

  written = __llvm_libc::sprintf(buff, "%5a", inf);
  ASSERT_STREQ_LEN(written, buff, "  inf");

  written = __llvm_libc::sprintf(buff, "%5a", -nan);
  ASSERT_STREQ_LEN(written, buff, " -nan");

  // Precision Tests.

  written = __llvm_libc::sprintf(buff, "%.1a", 1.0);
  ASSERT_STREQ_LEN(written, buff, "0x1.0p+0");

  written = __llvm_libc::sprintf(buff, "%.1a", 0.0);
  ASSERT_STREQ_LEN(written, buff, "0x0.0p+0");

  written = __llvm_libc::sprintf(buff, "%.1a", 0.1);
  ASSERT_STREQ_LEN(written, buff, "0x1.ap-4");

  written = __llvm_libc::sprintf(buff, "%.1a", 0x1.0fp0);
  ASSERT_STREQ_LEN(written, buff, "0x1.1p+0");

  written = __llvm_libc::sprintf(buff, "%.1a", 0x1.07p0);
  ASSERT_STREQ_LEN(written, buff, "0x1.0p+0");

  written = __llvm_libc::sprintf(buff, "%.1a", 0x1.08p0);
  ASSERT_STREQ_LEN(written, buff, "0x1.0p+0");

  written = __llvm_libc::sprintf(buff, "%.1a", 0x1.18p0);
  ASSERT_STREQ_LEN(written, buff, "0x1.2p+0");

  written = __llvm_libc::sprintf(buff, "%.1a", 0x1.ffp0);
  ASSERT_STREQ_LEN(written, buff, "0x2.0p+0");

  written = __llvm_libc::sprintf(buff, "%.5a", 1.25);
  ASSERT_STREQ_LEN(written, buff, "0x1.40000p+0");

  written = __llvm_libc::sprintf(buff, "%.0a", 1.25);
  ASSERT_STREQ_LEN(written, buff, "0x1p+0");

  written = __llvm_libc::sprintf(buff, "%.0a", 1.75);
  ASSERT_STREQ_LEN(written, buff, "0x2p+0");

  written = __llvm_libc::sprintf(buff, "%.1a", 0x1.0p-1023);
  ASSERT_STREQ_LEN(written, buff, "0x0.8p-1022");

  written = __llvm_libc::sprintf(buff, "%.1a", 0x1.8p-1023);
  ASSERT_STREQ_LEN(written, buff, "0x0.cp-1022");

  written = __llvm_libc::sprintf(buff, "%.1a", 0x1.0p-1024);
  ASSERT_STREQ_LEN(written, buff, "0x0.4p-1022");

  written = __llvm_libc::sprintf(buff, "%.0a", 0x1.0p-1023);
  ASSERT_STREQ_LEN(written, buff, "0x0p-1022");

  written = __llvm_libc::sprintf(buff, "%.0a", 0x1.8p-1023);
  ASSERT_STREQ_LEN(written, buff, "0x1p-1022");

  written = __llvm_libc::sprintf(buff, "%.0a", 0x1.0p-1024);
  ASSERT_STREQ_LEN(written, buff, "0x0p-1022");

  written = __llvm_libc::sprintf(buff, "%.2a", 0x1.0p-1027);
  ASSERT_STREQ_LEN(written, buff, "0x0.08p-1022");

  written = __llvm_libc::sprintf(buff, "%.1a", 0x1.0p-1027);
  ASSERT_STREQ_LEN(written, buff, "0x0.0p-1022");

  written = __llvm_libc::sprintf(buff, "%.5a", 0.0);
  ASSERT_STREQ_LEN(written, buff, "0x0.00000p+0");

  written = __llvm_libc::sprintf(buff, "%.5a", 0x1.008p0);
  ASSERT_STREQ_LEN(written, buff, "0x1.00800p+0");

  written = __llvm_libc::sprintf(buff, "%.5a", 0x1.008p10);
  ASSERT_STREQ_LEN(written, buff, "0x1.00800p+10");

  written = __llvm_libc::sprintf(buff, "%.5a", nan);
  ASSERT_STREQ_LEN(written, buff, "nan");

  written = __llvm_libc::sprintf(buff, "%.1La", 0.1L);
#if defined(SPECIAL_X86_LONG_DOUBLE)
  ASSERT_STREQ_LEN(written, buff, "0xc.dp-7");
#elif defined(LONG_DOUBLE_IS_DOUBLE)
  ASSERT_STREQ_LEN(written, buff, "0x1.ap-4");
#else // 128 bit long double
  ASSERT_STREQ_LEN(written, buff, "0x1.ap-4");
#endif

  written = __llvm_libc::sprintf(buff, "%.1La", 0xf.fffffffffffffffp16380L);
#if defined(SPECIAL_X86_LONG_DOUBLE)
  ASSERT_STREQ_LEN(written, buff, "0x1.0p+16384");
#elif defined(LONG_DOUBLE_IS_DOUBLE)
  ASSERT_STREQ_LEN(written, buff, "inf");
#else // 128 bit long double
  ASSERT_STREQ_LEN(written, buff, "0x2.0p+16383");
#endif

  // Rounding Mode Tests.

  {
    __llvm_libc::testutils::ForceRoundingMode r(
        __llvm_libc::testutils::RoundingMode::Nearest);

    written = __llvm_libc::sprintf(buff, "%.1a", 0x1.08p0);
    ASSERT_STREQ_LEN(written, buff, "0x1.0p+0");

    written = __llvm_libc::sprintf(buff, "%.1a", 0x1.18p0);
    ASSERT_STREQ_LEN(written, buff, "0x1.2p+0");

    written = __llvm_libc::sprintf(buff, "%.1a", 0x1.04p0);
    ASSERT_STREQ_LEN(written, buff, "0x1.0p+0");

    written = __llvm_libc::sprintf(buff, "%.1a", 0x1.14p0);
    ASSERT_STREQ_LEN(written, buff, "0x1.1p+0");

    written = __llvm_libc::sprintf(buff, "%.1a", -0x1.08p0);
    ASSERT_STREQ_LEN(written, buff, "-0x1.0p+0");

    written = __llvm_libc::sprintf(buff, "%.1a", -0x1.18p0);
    ASSERT_STREQ_LEN(written, buff, "-0x1.2p+0");

    written = __llvm_libc::sprintf(buff, "%.1a", -0x1.04p0);
    ASSERT_STREQ_LEN(written, buff, "-0x1.0p+0");

    written = __llvm_libc::sprintf(buff, "%.1a", -0x1.14p0);
    ASSERT_STREQ_LEN(written, buff, "-0x1.1p+0");
  }

  {
    __llvm_libc::testutils::ForceRoundingMode r(
        __llvm_libc::testutils::RoundingMode::Upward);

    written = __llvm_libc::sprintf(buff, "%.1a", 0x1.08p0);
    ASSERT_STREQ_LEN(written, buff, "0x1.1p+0");

    written = __llvm_libc::sprintf(buff, "%.1a", 0x1.18p0);
    ASSERT_STREQ_LEN(written, buff, "0x1.2p+0");

    written = __llvm_libc::sprintf(buff, "%.1a", 0x1.04p0);
    ASSERT_STREQ_LEN(written, buff, "0x1.1p+0");

    written = __llvm_libc::sprintf(buff, "%.1a", 0x1.14p0);
    ASSERT_STREQ_LEN(written, buff, "0x1.2p+0");

    written = __llvm_libc::sprintf(buff, "%.1a", -0x1.08p0);
    ASSERT_STREQ_LEN(written, buff, "-0x1.0p+0");

    written = __llvm_libc::sprintf(buff, "%.1a", -0x1.18p0);
    ASSERT_STREQ_LEN(written, buff, "-0x1.1p+0");

    written = __llvm_libc::sprintf(buff, "%.1a", -0x1.04p0);
    ASSERT_STREQ_LEN(written, buff, "-0x1.0p+0");

    written = __llvm_libc::sprintf(buff, "%.1a", -0x1.14p0);
    ASSERT_STREQ_LEN(written, buff, "-0x1.1p+0");
  }

  {
    __llvm_libc::testutils::ForceRoundingMode r(
        __llvm_libc::testutils::RoundingMode::Downward);

    written = __llvm_libc::sprintf(buff, "%.1a", 0x1.08p0);
    ASSERT_STREQ_LEN(written, buff, "0x1.0p+0");

    written = __llvm_libc::sprintf(buff, "%.1a", 0x1.18p0);
    ASSERT_STREQ_LEN(written, buff, "0x1.1p+0");

    written = __llvm_libc::sprintf(buff, "%.1a", 0x1.04p0);
    ASSERT_STREQ_LEN(written, buff, "0x1.0p+0");

    written = __llvm_libc::sprintf(buff, "%.1a", 0x1.14p0);
    ASSERT_STREQ_LEN(written, buff, "0x1.1p+0");

    written = __llvm_libc::sprintf(buff, "%.1a", -0x1.08p0);
    ASSERT_STREQ_LEN(written, buff, "-0x1.1p+0");

    written = __llvm_libc::sprintf(buff, "%.1a", -0x1.18p0);
    ASSERT_STREQ_LEN(written, buff, "-0x1.2p+0");

    written = __llvm_libc::sprintf(buff, "%.1a", -0x1.04p0);
    ASSERT_STREQ_LEN(written, buff, "-0x1.1p+0");

    written = __llvm_libc::sprintf(buff, "%.1a", -0x1.14p0);
    ASSERT_STREQ_LEN(written, buff, "-0x1.2p+0");
  }

  {
    __llvm_libc::testutils::ForceRoundingMode r(
        __llvm_libc::testutils::RoundingMode::TowardZero);

    written = __llvm_libc::sprintf(buff, "%.1a", 0x1.08p0);
    ASSERT_STREQ_LEN(written, buff, "0x1.0p+0");

    written = __llvm_libc::sprintf(buff, "%.1a", 0x1.18p0);
    ASSERT_STREQ_LEN(written, buff, "0x1.1p+0");

    written = __llvm_libc::sprintf(buff, "%.1a", 0x1.04p0);
    ASSERT_STREQ_LEN(written, buff, "0x1.0p+0");

    written = __llvm_libc::sprintf(buff, "%.1a", 0x1.14p0);
    ASSERT_STREQ_LEN(written, buff, "0x1.1p+0");

    written = __llvm_libc::sprintf(buff, "%.1a", -0x1.08p0);
    ASSERT_STREQ_LEN(written, buff, "-0x1.0p+0");

    written = __llvm_libc::sprintf(buff, "%.1a", -0x1.18p0);
    ASSERT_STREQ_LEN(written, buff, "-0x1.1p+0");

    written = __llvm_libc::sprintf(buff, "%.1a", -0x1.04p0);
    ASSERT_STREQ_LEN(written, buff, "-0x1.0p+0");

    written = __llvm_libc::sprintf(buff, "%.1a", -0x1.14p0);
    ASSERT_STREQ_LEN(written, buff, "-0x1.1p+0");
  }

  // Flag Tests.

  written = __llvm_libc::sprintf(buff, "%+a", nan);
  ASSERT_STREQ_LEN(written, buff, "+nan");

  written = __llvm_libc::sprintf(buff, "% A", inf);
  ASSERT_STREQ_LEN(written, buff, " INF");

  written = __llvm_libc::sprintf(buff, "%-5a", inf);
  ASSERT_STREQ_LEN(written, buff, "inf  ");

  written = __llvm_libc::sprintf(buff, "%+-5A", nan);
  ASSERT_STREQ_LEN(written, buff, "+NAN ");

  written = __llvm_libc::sprintf(buff, "%+a", 1.0);
  ASSERT_STREQ_LEN(written, buff, "+0x1p+0");

  written = __llvm_libc::sprintf(buff, "% a", 0.0);
  ASSERT_STREQ_LEN(written, buff, " 0x0p+0");

  written = __llvm_libc::sprintf(buff, "%-10a", 1.5);
  ASSERT_STREQ_LEN(written, buff, "0x1.8p+0  ");

  written = __llvm_libc::sprintf(buff, "%#a", 1.0);
  ASSERT_STREQ_LEN(written, buff, "0x1.p+0");

  written = __llvm_libc::sprintf(buff, "%#.0a", 1.5);
  ASSERT_STREQ_LEN(written, buff, "0x2.p+0");

  written = __llvm_libc::sprintf(buff, "%010a", 1.5);
  ASSERT_STREQ_LEN(written, buff, "0x001.8p+0");

  written = __llvm_libc::sprintf(buff, "%+- #0a", 0.0);
  ASSERT_STREQ_LEN(written, buff, "+0x0.p+0");

  // Combined Tests.

  written = __llvm_libc::sprintf(buff, "%12.3a %-12.3A", 0.1, 256.0);
  ASSERT_STREQ_LEN(written, buff, "  0x1.99ap-4 0X1.000P+8  ");

  written = __llvm_libc::sprintf(buff, "%+-#12.3a % 012.3a", 0.1256, 1256.0);
  ASSERT_STREQ_LEN(written, buff, "+0x1.014p-3   0x1.3a0p+10");
}

TEST_F(LlvmLibcSPrintfTest, FloatDecimalConv) {
  __llvm_libc::testutils::ForceRoundingMode r(
      __llvm_libc::testutils::RoundingMode::Nearest);
  double inf = __llvm_libc::fputil::FPBits<double>::inf().get_val();
  double nan = __llvm_libc::fputil::FPBits<double>::build_nan(1);

  written = __llvm_libc::sprintf(buff, "%f", 1.0);
  ASSERT_STREQ_LEN(written, buff, "1.000000");

  written = __llvm_libc::sprintf(buff, "%F", -1.0);
  ASSERT_STREQ_LEN(written, buff, "-1.000000");

  written = __llvm_libc::sprintf(buff, "%f", -1.234567);
  ASSERT_STREQ_LEN(written, buff, "-1.234567");

  written = __llvm_libc::sprintf(buff, "%f", 0.0);
  ASSERT_STREQ_LEN(written, buff, "0.000000");

  written = __llvm_libc::sprintf(buff, "%f", 1.5);
  ASSERT_STREQ_LEN(written, buff, "1.500000");

  written = __llvm_libc::sprintf(buff, "%f", 1e300);
  ASSERT_STREQ_LEN(
      written, buff,
      "100000000000000005250476025520442024870446858110815915491585411551180245"
      "798890819578637137508044786404370444383288387817694252323536043057564479"
      "218478670698284838720092657580373783023379478809005936895323497079994508"
      "111903896764088007465274278014249457925878882005684283811566947219638686"
      "5459400540160.000000");

  written = __llvm_libc::sprintf(buff, "%f", 0.1);
  ASSERT_STREQ_LEN(written, buff, "0.100000");

  written = __llvm_libc::sprintf(buff, "%f", 1234567890123456789.0);
  ASSERT_STREQ_LEN(written, buff, "1234567890123456768.000000");

  written = __llvm_libc::sprintf(buff, "%f", 9999999999999.99);
  ASSERT_STREQ_LEN(written, buff, "9999999999999.990234");

  // Simple Subnormal Tests.

  written = __llvm_libc::sprintf(buff, "%f", 0x1.0p-1027);
  ASSERT_STREQ_LEN(written, buff, "0.000000");

  written = __llvm_libc::sprintf(buff, "%f", 0x1.0p-1074);
  ASSERT_STREQ_LEN(written, buff, "0.000000");

  // Inf/Nan Tests.

  written = __llvm_libc::sprintf(buff, "%f", inf);
  ASSERT_STREQ_LEN(written, buff, "inf");

  written = __llvm_libc::sprintf(buff, "%F", -inf);
  ASSERT_STREQ_LEN(written, buff, "-INF");

  written = __llvm_libc::sprintf(buff, "%f", nan);
  ASSERT_STREQ_LEN(written, buff, "nan");

  written = __llvm_libc::sprintf(buff, "%F", -nan);
  ASSERT_STREQ_LEN(written, buff, "-NAN");

  // Length Modifier Tests.

  // TODO: Fix long doubles (needs bigger table or alternate algorithm.)
  // Currently the table values are generated, which is very slow.
  /*
  written = __llvm_libc::sprintf(buff, "%Lf", 1e100L);
  ASSERT_STREQ_LEN(written, buff,
                   "99999999999999999996693535322073426194986990198284960792713"
                   "91541752018669482644324418977840117055488.000000");

  written = __llvm_libc::sprintf(buff, "%Lf", 1.0L);
  ASSERT_STREQ_LEN(written, buff, "1.000000");

  char big_buff[10000];
  written = __llvm_libc::sprintf(big_buff, "%Lf", 1e1000L);
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

  written = __llvm_libc::sprintf(big_buff, "%Lf", 1e4900L);
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
    written = __llvm_libc::sprintf(buff, "%La", 0.1L);
  #if defined(SPECIAL_X86_LONG_DOUBLE)
    ASSERT_STREQ_LEN(written, buff, "0xc.ccccccccccccccdp-7");
  #elif defined(LONG_DOUBLE_IS_DOUBLE)
    ASSERT_STREQ_LEN(written, buff, "0x1.999999999999ap-4");
  #else // 128 bit long double
    ASSERT_STREQ_LEN(written, buff, "0x1.999999999999999999999999999ap-4");
  #endif

    written = __llvm_libc::sprintf(buff, "%La", 1.0e1000L);
  #if defined(SPECIAL_X86_LONG_DOUBLE)
    ASSERT_STREQ_LEN(written, buff, "0xf.38db1f9dd3dac05p+3318");
  #elif defined(LONG_DOUBLE_IS_DOUBLE)
    ASSERT_STREQ_LEN(written, buff, "inf");
  #else // 128 bit long double
    ASSERT_STREQ_LEN(written, buff, "0x1.e71b63f3ba7b580af1a52d2a7379p+3321");
  #endif

    written = __llvm_libc::sprintf(buff, "%La", 1.0e-1000L);
  #if defined(SPECIAL_X86_LONG_DOUBLE)
    ASSERT_STREQ_LEN(written, buff, "0x8.68a9188a89e1467p-3325");
  #elif defined(LONG_DOUBLE_IS_DOUBLE)
    ASSERT_STREQ_LEN(written, buff, "0x0p+0");
  #else // 128 bit long double
    ASSERT_STREQ_LEN(written, buff, "0x1.0d152311513c28ce202627c06ec2p-3322");
  #endif
  */

  // Min Width Tests.

  written = __llvm_libc::sprintf(buff, "%15f", 1.0);
  ASSERT_STREQ_LEN(written, buff, "       1.000000");

  written = __llvm_libc::sprintf(buff, "%15f", -1.0);
  ASSERT_STREQ_LEN(written, buff, "      -1.000000");

  written = __llvm_libc::sprintf(buff, "%15f", 1.0e5);
  ASSERT_STREQ_LEN(written, buff, "  100000.000000");

  written = __llvm_libc::sprintf(buff, "%15f", -1.0e5);
  ASSERT_STREQ_LEN(written, buff, " -100000.000000");

  written = __llvm_libc::sprintf(buff, "%10f", 1.0e5);
  ASSERT_STREQ_LEN(written, buff, "100000.000000");

  // Precision Tests.

  written = __llvm_libc::sprintf(buff, "%.1f", 1.0);
  ASSERT_STREQ_LEN(written, buff, "1.0");

  written = __llvm_libc::sprintf(buff, "%.1f", 0.0);
  ASSERT_STREQ_LEN(written, buff, "0.0");

  written = __llvm_libc::sprintf(buff, "%.0f", 0.0);
  ASSERT_STREQ_LEN(written, buff, "0");

  written = __llvm_libc::sprintf(buff, "%.1f", 0.1);
  ASSERT_STREQ_LEN(written, buff, "0.1");

  written = __llvm_libc::sprintf(buff, "%.1f", 1.09);
  ASSERT_STREQ_LEN(written, buff, "1.1");

  written = __llvm_libc::sprintf(buff, "%.1f", 1.04);
  ASSERT_STREQ_LEN(written, buff, "1.0");

  written = __llvm_libc::sprintf(buff, "%.1f", 1.19);
  ASSERT_STREQ_LEN(written, buff, "1.2");

  written = __llvm_libc::sprintf(buff, "%.1f", 1.99);
  ASSERT_STREQ_LEN(written, buff, "2.0");

  written = __llvm_libc::sprintf(buff, "%.1f", 9.99);
  ASSERT_STREQ_LEN(written, buff, "10.0");

  written = __llvm_libc::sprintf(buff, "%.2f", 9999999999999.99);
  ASSERT_STREQ_LEN(written, buff, "9999999999999.99");

  written = __llvm_libc::sprintf(buff, "%.1f", 9999999999999.99);
  ASSERT_STREQ_LEN(written, buff, "10000000000000.0");

  written = __llvm_libc::sprintf(buff, "%.5f", 1.25);
  ASSERT_STREQ_LEN(written, buff, "1.25000");

  written = __llvm_libc::sprintf(buff, "%.0f", 1.25);
  ASSERT_STREQ_LEN(written, buff, "1");

  written = __llvm_libc::sprintf(buff, "%.0f", 1.75);
  ASSERT_STREQ_LEN(written, buff, "2");

  written = __llvm_libc::sprintf(buff, "%.20f", 1.234e-10);
  ASSERT_STREQ_LEN(written, buff, "0.00000000012340000000");

  written = __llvm_libc::sprintf(buff, "%.2f", -9.99);
  ASSERT_STREQ_LEN(written, buff, "-9.99");

  written = __llvm_libc::sprintf(buff, "%.1f", -9.99);
  ASSERT_STREQ_LEN(written, buff, "-10.0");

  written = __llvm_libc::sprintf(buff, "%.5f", 0.0);
  ASSERT_STREQ_LEN(written, buff, "0.00000");

  written = __llvm_libc::sprintf(buff, "%.5f", 1.008);
  ASSERT_STREQ_LEN(written, buff, "1.00800");

  written = __llvm_libc::sprintf(buff, "%.5f", 1.008e3);
  ASSERT_STREQ_LEN(written, buff, "1008.00000");

  // Subnormal Precision Tests

  written = __llvm_libc::sprintf(buff, "%.310f", 0x1.0p-1022);
  ASSERT_STREQ_LEN(
      written, buff,
      "0."
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "0000000000000000000223");

  written = __llvm_libc::sprintf(buff, "%.310f", 0x1.0p-1023);
  ASSERT_STREQ_LEN(
      written, buff,
      "0."
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "0000000000000000000111");

  written = __llvm_libc::sprintf(buff, "%.315f", 9.99999e-310);
  ASSERT_STREQ_LEN(
      written, buff,
      "0."
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000999999");

  written = __llvm_libc::sprintf(buff, "%.314f", 9.99999e-310);
  ASSERT_STREQ_LEN(
      written, buff,
      "0."
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "00000000000000000000100000");

  written = __llvm_libc::sprintf(buff, "%.330f", 0x1.0p-1074);
  ASSERT_STREQ_LEN(
      written, buff,
      "0."
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000004940656");

  /*
    written = __llvm_libc::sprintf(buff, "%.1La", 0.1L);
  #if defined(SPECIAL_X86_LONG_DOUBLE)
    ASSERT_STREQ_LEN(written, buff, "0xc.dp-7");
  #elif defined(LONG_DOUBLE_IS_DOUBLE)
    ASSERT_STREQ_LEN(written, buff, "0x1.ap-4");
  #else // 128 bit long double
    ASSERT_STREQ_LEN(written, buff, "0x1.ap-4");
  #endif

    written = __llvm_libc::sprintf(buff, "%.1La", 0xf.fffffffffffffffp16380L);
  #if defined(SPECIAL_X86_LONG_DOUBLE)
    ASSERT_STREQ_LEN(written, buff, "0x1.0p+16384");
  #elif defined(LONG_DOUBLE_IS_DOUBLE)
    ASSERT_STREQ_LEN(written, buff, "inf");
  #else // 128 bit long double
    ASSERT_STREQ_LEN(written, buff, "0x2.0p+16383");
  #endif
  */

  // Rounding Mode Tests.

  {
    __llvm_libc::testutils::ForceRoundingMode r(
        __llvm_libc::testutils::RoundingMode::Nearest);

    written = __llvm_libc::sprintf(buff, "%.1f", 1.75);
    ASSERT_STREQ_LEN(written, buff, "1.8");

    written = __llvm_libc::sprintf(buff, "%.1f", 1.25);
    ASSERT_STREQ_LEN(written, buff, "1.2");

    written = __llvm_libc::sprintf(buff, "%.1f", 1.125);
    ASSERT_STREQ_LEN(written, buff, "1.1");

    written = __llvm_libc::sprintf(buff, "%.1f", 1.625);
    ASSERT_STREQ_LEN(written, buff, "1.6");

    written = __llvm_libc::sprintf(buff, "%.1f", 1.375);
    ASSERT_STREQ_LEN(written, buff, "1.4");

    written = __llvm_libc::sprintf(buff, "%.1f", 1.875);
    ASSERT_STREQ_LEN(written, buff, "1.9");

    written = __llvm_libc::sprintf(buff, "%.1f", -1.75);
    ASSERT_STREQ_LEN(written, buff, "-1.8");

    written = __llvm_libc::sprintf(buff, "%.1f", -1.25);
    ASSERT_STREQ_LEN(written, buff, "-1.2");

    written = __llvm_libc::sprintf(buff, "%.1f", -1.125);
    ASSERT_STREQ_LEN(written, buff, "-1.1");

    written = __llvm_libc::sprintf(buff, "%.1f", -1.625);
    ASSERT_STREQ_LEN(written, buff, "-1.6");

    written = __llvm_libc::sprintf(buff, "%.1f", -1.375);
    ASSERT_STREQ_LEN(written, buff, "-1.4");

    written = __llvm_libc::sprintf(buff, "%.1f", -1.875);
    ASSERT_STREQ_LEN(written, buff, "-1.9");
  }

  {
    __llvm_libc::testutils::ForceRoundingMode r(
        __llvm_libc::testutils::RoundingMode::Upward);

    written = __llvm_libc::sprintf(buff, "%.1f", 1.75);
    ASSERT_STREQ_LEN(written, buff, "1.8");

    written = __llvm_libc::sprintf(buff, "%.1f", 1.25);
    ASSERT_STREQ_LEN(written, buff, "1.3");

    written = __llvm_libc::sprintf(buff, "%.1f", 1.125);
    ASSERT_STREQ_LEN(written, buff, "1.2");

    written = __llvm_libc::sprintf(buff, "%.1f", 1.625);
    ASSERT_STREQ_LEN(written, buff, "1.7");

    written = __llvm_libc::sprintf(buff, "%.1f", 1.375);
    ASSERT_STREQ_LEN(written, buff, "1.4");

    written = __llvm_libc::sprintf(buff, "%.1f", 1.875);
    ASSERT_STREQ_LEN(written, buff, "1.9");

    written = __llvm_libc::sprintf(buff, "%.1f", -1.75);
    ASSERT_STREQ_LEN(written, buff, "-1.7");

    written = __llvm_libc::sprintf(buff, "%.1f", -1.25);
    ASSERT_STREQ_LEN(written, buff, "-1.2");

    written = __llvm_libc::sprintf(buff, "%.1f", -1.125);
    ASSERT_STREQ_LEN(written, buff, "-1.1");

    written = __llvm_libc::sprintf(buff, "%.1f", -1.625);
    ASSERT_STREQ_LEN(written, buff, "-1.6");

    written = __llvm_libc::sprintf(buff, "%.1f", -1.375);
    ASSERT_STREQ_LEN(written, buff, "-1.3");

    written = __llvm_libc::sprintf(buff, "%.1f", -1.875);
    ASSERT_STREQ_LEN(written, buff, "-1.8");
  }

  {
    __llvm_libc::testutils::ForceRoundingMode r(
        __llvm_libc::testutils::RoundingMode::Downward);

    written = __llvm_libc::sprintf(buff, "%.1f", 1.75);
    ASSERT_STREQ_LEN(written, buff, "1.7");

    written = __llvm_libc::sprintf(buff, "%.1f", 1.25);
    ASSERT_STREQ_LEN(written, buff, "1.2");

    written = __llvm_libc::sprintf(buff, "%.1f", 1.125);
    ASSERT_STREQ_LEN(written, buff, "1.1");

    written = __llvm_libc::sprintf(buff, "%.1f", 1.625);
    ASSERT_STREQ_LEN(written, buff, "1.6");

    written = __llvm_libc::sprintf(buff, "%.1f", 1.375);
    ASSERT_STREQ_LEN(written, buff, "1.3");

    written = __llvm_libc::sprintf(buff, "%.1f", 1.875);
    ASSERT_STREQ_LEN(written, buff, "1.8");

    written = __llvm_libc::sprintf(buff, "%.1f", -1.75);
    ASSERT_STREQ_LEN(written, buff, "-1.8");

    written = __llvm_libc::sprintf(buff, "%.1f", -1.25);
    ASSERT_STREQ_LEN(written, buff, "-1.3");

    written = __llvm_libc::sprintf(buff, "%.1f", -1.125);
    ASSERT_STREQ_LEN(written, buff, "-1.2");

    written = __llvm_libc::sprintf(buff, "%.1f", -1.625);
    ASSERT_STREQ_LEN(written, buff, "-1.7");

    written = __llvm_libc::sprintf(buff, "%.1f", -1.375);
    ASSERT_STREQ_LEN(written, buff, "-1.4");

    written = __llvm_libc::sprintf(buff, "%.1f", -1.875);
    ASSERT_STREQ_LEN(written, buff, "-1.9");
  }

  {
    __llvm_libc::testutils::ForceRoundingMode r(
        __llvm_libc::testutils::RoundingMode::TowardZero);

    written = __llvm_libc::sprintf(buff, "%.1f", 1.75);
    ASSERT_STREQ_LEN(written, buff, "1.7");

    written = __llvm_libc::sprintf(buff, "%.1f", 1.25);
    ASSERT_STREQ_LEN(written, buff, "1.2");

    written = __llvm_libc::sprintf(buff, "%.1f", 1.125);
    ASSERT_STREQ_LEN(written, buff, "1.1");

    written = __llvm_libc::sprintf(buff, "%.1f", 1.625);
    ASSERT_STREQ_LEN(written, buff, "1.6");

    written = __llvm_libc::sprintf(buff, "%.1f", 1.375);
    ASSERT_STREQ_LEN(written, buff, "1.3");

    written = __llvm_libc::sprintf(buff, "%.1f", 1.875);
    ASSERT_STREQ_LEN(written, buff, "1.8");

    written = __llvm_libc::sprintf(buff, "%.1f", -1.75);
    ASSERT_STREQ_LEN(written, buff, "-1.7");

    written = __llvm_libc::sprintf(buff, "%.1f", -1.25);
    ASSERT_STREQ_LEN(written, buff, "-1.2");

    written = __llvm_libc::sprintf(buff, "%.1f", -1.125);
    ASSERT_STREQ_LEN(written, buff, "-1.1");

    written = __llvm_libc::sprintf(buff, "%.1f", -1.625);
    ASSERT_STREQ_LEN(written, buff, "-1.6");

    written = __llvm_libc::sprintf(buff, "%.1f", -1.375);
    ASSERT_STREQ_LEN(written, buff, "-1.3");

    written = __llvm_libc::sprintf(buff, "%.1f", -1.875);
    ASSERT_STREQ_LEN(written, buff, "-1.8");
  }

  // Flag Tests.
  written = __llvm_libc::sprintf(buff, "%+f", 1.0);
  ASSERT_STREQ_LEN(written, buff, "+1.000000");

  written = __llvm_libc::sprintf(buff, "%+f", -1.0);
  ASSERT_STREQ_LEN(written, buff, "-1.000000");

  written = __llvm_libc::sprintf(buff, "% f", 1.0);
  ASSERT_STREQ_LEN(written, buff, " 1.000000");

  written = __llvm_libc::sprintf(buff, "% f", -1.0);
  ASSERT_STREQ_LEN(written, buff, "-1.000000");

  written = __llvm_libc::sprintf(buff, "%-10f", 1.5);
  ASSERT_STREQ_LEN(written, buff, "1.500000  ");

  written = __llvm_libc::sprintf(buff, "%#.f", 1.0);
  ASSERT_STREQ_LEN(written, buff, "1.");

  written = __llvm_libc::sprintf(buff, "%#.0f", 1.5);
  ASSERT_STREQ_LEN(written, buff, "2.");

  written = __llvm_libc::sprintf(buff, "%010f", 1.5);
  ASSERT_STREQ_LEN(written, buff, "001.500000");

  written = __llvm_libc::sprintf(buff, "%010f", -1.5);
  ASSERT_STREQ_LEN(written, buff, "-01.500000");

  written = __llvm_libc::sprintf(buff, "%+- #0f", 0.0);
  ASSERT_STREQ_LEN(written, buff, "+0.000000");

  // Combined Tests.

  written = __llvm_libc::sprintf(buff, "%10.2f", 9.99);
  ASSERT_STREQ_LEN(written, buff, "      9.99");

  written = __llvm_libc::sprintf(buff, "%5.1f", 9.99);
  ASSERT_STREQ_LEN(written, buff, " 10.0");

  written = __llvm_libc::sprintf(buff, "%-10.2f", 9.99);
  ASSERT_STREQ_LEN(written, buff, "9.99      ");

  written = __llvm_libc::sprintf(buff, "%-5.1f", 9.99);
  ASSERT_STREQ_LEN(written, buff, "10.0 ");

  written = __llvm_libc::sprintf(buff, "%-5.1f", 1.0e-50);
  ASSERT_STREQ_LEN(written, buff, "0.0  ");

  written = __llvm_libc::sprintf(buff, "%30f", 1234567890123456789.0);
  ASSERT_STREQ_LEN(written, buff, "    1234567890123456768.000000");

  written = __llvm_libc::sprintf(buff, "%-30f", 1234567890123456789.0);
  ASSERT_STREQ_LEN(written, buff, "1234567890123456768.000000    ");

  written = __llvm_libc::sprintf(buff, "%20.2f", 9999999999999.99);
  ASSERT_STREQ_LEN(written, buff, "    9999999999999.99");

  written = __llvm_libc::sprintf(buff, "%20.1f", 9999999999999.99);
  ASSERT_STREQ_LEN(written, buff, "    10000000000000.0");

  written = __llvm_libc::sprintf(buff, "%12.3f %-12.3f", 0.1, 256.0);
  ASSERT_STREQ_LEN(written, buff, "       0.100 256.000     ");

  written = __llvm_libc::sprintf(buff, "%+-#12.3f % 012.3f", 0.1256, 1256.0);
  ASSERT_STREQ_LEN(written, buff, "+0.126        0001256.000");
}

TEST_F(LlvmLibcSPrintfTest, FloatExponentConv) {
  __llvm_libc::testutils::ForceRoundingMode r(
      __llvm_libc::testutils::RoundingMode::Nearest);
  double inf = __llvm_libc::fputil::FPBits<double>::inf().get_val();
  double nan = __llvm_libc::fputil::FPBits<double>::build_nan(1);

  written = __llvm_libc::sprintf(buff, "%e", 1.0);
  ASSERT_STREQ_LEN(written, buff, "1.000000e+00");

  written = __llvm_libc::sprintf(buff, "%E", -1.0);
  ASSERT_STREQ_LEN(written, buff, "-1.000000E+00");

  written = __llvm_libc::sprintf(buff, "%e", -1.234567);
  ASSERT_STREQ_LEN(written, buff, "-1.234567e+00");

  written = __llvm_libc::sprintf(buff, "%e", 0.0);
  ASSERT_STREQ_LEN(written, buff, "0.000000e+00");

  written = __llvm_libc::sprintf(buff, "%e", 1.5);
  ASSERT_STREQ_LEN(written, buff, "1.500000e+00");

  written = __llvm_libc::sprintf(buff, "%e", 1e300);
  ASSERT_STREQ_LEN(written, buff, "1.000000e+300");

  written = __llvm_libc::sprintf(buff, "%e", 0.1);
  ASSERT_STREQ_LEN(written, buff, "1.000000e-01");

  written = __llvm_libc::sprintf(buff, "%e", 0.001);
  ASSERT_STREQ_LEN(written, buff, "1.000000e-03");

  written = __llvm_libc::sprintf(buff, "%e", 0.00001);
  ASSERT_STREQ_LEN(written, buff, "1.000000e-05");

  written = __llvm_libc::sprintf(buff, "%e", 0.0000001);
  ASSERT_STREQ_LEN(written, buff, "1.000000e-07");

  written = __llvm_libc::sprintf(buff, "%e", 0.000000001);
  ASSERT_STREQ_LEN(written, buff, "1.000000e-09");

  written = __llvm_libc::sprintf(buff, "%e", 1.0e-20);
  ASSERT_STREQ_LEN(written, buff, "1.000000e-20");

  written = __llvm_libc::sprintf(buff, "%e", 1234567890123456789.0);
  ASSERT_STREQ_LEN(written, buff, "1.234568e+18");

  written = __llvm_libc::sprintf(buff, "%e", 9999999000000.00);
  ASSERT_STREQ_LEN(written, buff, "9.999999e+12");

  // Simple Subnormal Tests.

  written = __llvm_libc::sprintf(buff, "%e", 0x1.0p-1027);
  ASSERT_STREQ_LEN(written, buff, "6.953356e-310");

  written = __llvm_libc::sprintf(buff, "%e", 0x1.0p-1074);
  ASSERT_STREQ_LEN(written, buff, "4.940656e-324");

  // Inf/Nan Tests.

  written = __llvm_libc::sprintf(buff, "%e", inf);
  ASSERT_STREQ_LEN(written, buff, "inf");

  written = __llvm_libc::sprintf(buff, "%E", -inf);
  ASSERT_STREQ_LEN(written, buff, "-INF");

  written = __llvm_libc::sprintf(buff, "%e", nan);
  ASSERT_STREQ_LEN(written, buff, "nan");

  written = __llvm_libc::sprintf(buff, "%E", -nan);
  ASSERT_STREQ_LEN(written, buff, "-NAN");

  // Length Modifier Tests.

  // TODO: Fix long doubles (needs bigger table or alternate algorithm.)
  // Currently the table values are generated, which is very slow.
  /*
  written = __llvm_libc::sprintf(buff, "%Lf", 1e100L);
  ASSERT_STREQ_LEN(written, buff,
                   "99999999999999999996693535322073426194986990198284960792713"
                   "91541752018669482644324418977840117055488.000000");

  written = __llvm_libc::sprintf(buff, "%Lf", 1.0L);
  ASSERT_STREQ_LEN(written, buff, "1.000000");

  char big_buff[10000];
  written = __llvm_libc::sprintf(big_buff, "%Lf", 1e1000L);
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

  written = __llvm_libc::sprintf(big_buff, "%Lf", 1e4900L);
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
    written = __llvm_libc::sprintf(buff, "%La", 0.1L);
  #if defined(SPECIAL_X86_LONG_DOUBLE)
    ASSERT_STREQ_LEN(written, buff, "0xc.ccccccccccccccdp-7");
  #elif defined(LONG_DOUBLE_IS_DOUBLE)
    ASSERT_STREQ_LEN(written, buff, "0x1.999999999999ap-4");
  #else // 128 bit long double
    ASSERT_STREQ_LEN(written, buff, "0x1.999999999999999999999999999ap-4");
  #endif

    written = __llvm_libc::sprintf(buff, "%La", 1.0e1000L);
  #if defined(SPECIAL_X86_LONG_DOUBLE)
    ASSERT_STREQ_LEN(written, buff, "0xf.38db1f9dd3dac05p+3318");
  #elif defined(LONG_DOUBLE_IS_DOUBLE)
    ASSERT_STREQ_LEN(written, buff, "inf");
  #else // 128 bit long double
    ASSERT_STREQ_LEN(written, buff, "0x1.e71b63f3ba7b580af1a52d2a7379p+3321");
  #endif

    written = __llvm_libc::sprintf(buff, "%La", 1.0e-1000L);
  #if defined(SPECIAL_X86_LONG_DOUBLE)
    ASSERT_STREQ_LEN(written, buff, "0x8.68a9188a89e1467p-3325");
  #elif defined(LONG_DOUBLE_IS_DOUBLE)
    ASSERT_STREQ_LEN(written, buff, "0x0p+0");
  #else // 128 bit long double
    ASSERT_STREQ_LEN(written, buff, "0x1.0d152311513c28ce202627c06ec2p-3322");
  #endif
  */

  // Min Width Tests.

  written = __llvm_libc::sprintf(buff, "%15e", 1.0);
  ASSERT_STREQ_LEN(written, buff, "   1.000000e+00");

  written = __llvm_libc::sprintf(buff, "%15e", -1.0);
  ASSERT_STREQ_LEN(written, buff, "  -1.000000e+00");

  written = __llvm_libc::sprintf(buff, "%15e", 1.0e5);
  ASSERT_STREQ_LEN(written, buff, "   1.000000e+05");

  written = __llvm_libc::sprintf(buff, "%15e", -1.0e5);
  ASSERT_STREQ_LEN(written, buff, "  -1.000000e+05");

  written = __llvm_libc::sprintf(buff, "%10e", 1.0e-5);
  ASSERT_STREQ_LEN(written, buff, "1.000000e-05");

  // Precision Tests.

  written = __llvm_libc::sprintf(buff, "%.1e", 1.0);
  ASSERT_STREQ_LEN(written, buff, "1.0e+00");

  written = __llvm_libc::sprintf(buff, "%.1e", 0.0);
  ASSERT_STREQ_LEN(written, buff, "0.0e+00");

  written = __llvm_libc::sprintf(buff, "%.0e", 0.0);
  ASSERT_STREQ_LEN(written, buff, "0e+00");

  written = __llvm_libc::sprintf(buff, "%.1e", 0.1);
  ASSERT_STREQ_LEN(written, buff, "1.0e-01");

  written = __llvm_libc::sprintf(buff, "%.1e", 1.09);
  ASSERT_STREQ_LEN(written, buff, "1.1e+00");

  written = __llvm_libc::sprintf(buff, "%.1e", 1.04);
  ASSERT_STREQ_LEN(written, buff, "1.0e+00");

  written = __llvm_libc::sprintf(buff, "%.1e", 1.19);
  ASSERT_STREQ_LEN(written, buff, "1.2e+00");

  written = __llvm_libc::sprintf(buff, "%.1e", 1.99);
  ASSERT_STREQ_LEN(written, buff, "2.0e+00");

  written = __llvm_libc::sprintf(buff, "%.1e", 9.99);
  ASSERT_STREQ_LEN(written, buff, "1.0e+01");

  written = __llvm_libc::sprintf(buff, "%.2e", 99.9);
  ASSERT_STREQ_LEN(written, buff, "9.99e+01");

  written = __llvm_libc::sprintf(buff, "%.1e", 99.9);
  ASSERT_STREQ_LEN(written, buff, "1.0e+02");

  written = __llvm_libc::sprintf(buff, "%.5e", 1.25);
  ASSERT_STREQ_LEN(written, buff, "1.25000e+00");

  written = __llvm_libc::sprintf(buff, "%.0e", 1.25);
  ASSERT_STREQ_LEN(written, buff, "1e+00");

  written = __llvm_libc::sprintf(buff, "%.0e", 1.75);
  ASSERT_STREQ_LEN(written, buff, "2e+00");

  written = __llvm_libc::sprintf(buff, "%.20e", 1.234e-10);
  ASSERT_STREQ_LEN(written, buff, "1.23400000000000008140e-10");

  written = __llvm_libc::sprintf(buff, "%.2e", -9.99);
  ASSERT_STREQ_LEN(written, buff, "-9.99e+00");

  written = __llvm_libc::sprintf(buff, "%.1e", -9.99);
  ASSERT_STREQ_LEN(written, buff, "-1.0e+01");

  written = __llvm_libc::sprintf(buff, "%.5e", 0.0);
  ASSERT_STREQ_LEN(written, buff, "0.00000e+00");

  written = __llvm_libc::sprintf(buff, "%.5e", 1.008);
  ASSERT_STREQ_LEN(written, buff, "1.00800e+00");

  written = __llvm_libc::sprintf(buff, "%.5e", 1.008e3);
  ASSERT_STREQ_LEN(written, buff, "1.00800e+03");

  // Subnormal Precision Tests

  written = __llvm_libc::sprintf(buff, "%.310e", 0x1.0p-1022);
  ASSERT_STREQ_LEN(
      written, buff,
      "2."
      "225073858507201383090232717332404064219215980462331830553327416887204434"
      "813918195854283159012511020564067339731035811005152434161553460108856012"
      "385377718821130777993532002330479610147442583636071921565046942503734208"
      "375250806650616658158948720491179968591639648500635908770118304874799780"
      "8877537499494515804516e-308");

  written = __llvm_libc::sprintf(buff, "%.30e", 0x1.0p-1022);
  ASSERT_STREQ_LEN(written, buff, "2.225073858507201383090232717332e-308");

  written = __llvm_libc::sprintf(buff, "%.310e", 0x1.0p-1023);
  ASSERT_STREQ_LEN(
      written, buff,
      "1."
      "112536929253600691545116358666202032109607990231165915276663708443602217"
      "406959097927141579506255510282033669865517905502576217080776730054428006"
      "192688859410565388996766001165239805073721291818035960782523471251867104"
      "187625403325308329079474360245589984295819824250317954385059152437399890"
      "4438768749747257902258e-308");

  written = __llvm_libc::sprintf(buff, "%.6e", 9.99999e-310);
  ASSERT_STREQ_LEN(written, buff, "9.999990e-310");

  written = __llvm_libc::sprintf(buff, "%.5e", 9.99999e-310);
  ASSERT_STREQ_LEN(written, buff, "9.99999e-310");

  written = __llvm_libc::sprintf(buff, "%.4e", 9.99999e-310);
  ASSERT_STREQ_LEN(written, buff, "1.0000e-309");

  written = __llvm_libc::sprintf(buff, "%.3e", 9.99999e-310);
  ASSERT_STREQ_LEN(written, buff, "1.000e-309");

  written = __llvm_libc::sprintf(buff, "%.2e", 9.99999e-310);
  ASSERT_STREQ_LEN(written, buff, "1.00e-309");

  written = __llvm_libc::sprintf(buff, "%.1e", 9.99999e-310);
  ASSERT_STREQ_LEN(written, buff, "1.0e-309");

  written = __llvm_libc::sprintf(buff, "%.0e", 9.99999e-310);
  ASSERT_STREQ_LEN(written, buff, "1e-309");

  written = __llvm_libc::sprintf(buff, "%.10e", 0x1.0p-1074);
  ASSERT_STREQ_LEN(written, buff, "4.9406564584e-324");

  /*
    written = __llvm_libc::sprintf(buff, "%.1La", 0.1L);
  #if defined(SPECIAL_X86_LONG_DOUBLE)
    ASSERT_STREQ_LEN(written, buff, "0xc.dp-7");
  #elif defined(LONG_DOUBLE_IS_DOUBLE)
    ASSERT_STREQ_LEN(written, buff, "0x1.ap-4");
  #else // 128 bit long double
    ASSERT_STREQ_LEN(written, buff, "0x1.ap-4");
  #endif

    written = __llvm_libc::sprintf(buff, "%.1La", 0xf.fffffffffffffffp16380L);
  #if defined(SPECIAL_X86_LONG_DOUBLE)
    ASSERT_STREQ_LEN(written, buff, "0x1.0p+16384");
  #elif defined(LONG_DOUBLE_IS_DOUBLE)
    ASSERT_STREQ_LEN(written, buff, "inf");
  #else // 128 bit long double
    ASSERT_STREQ_LEN(written, buff, "0x2.0p+16383");
  #endif
  */

  // Rounding Mode Tests.

  {
    __llvm_libc::testutils::ForceRoundingMode r(
        __llvm_libc::testutils::RoundingMode::Nearest);

    written = __llvm_libc::sprintf(buff, "%.1e", 1.75);
    ASSERT_STREQ_LEN(written, buff, "1.8e+00");

    written = __llvm_libc::sprintf(buff, "%.1e", 1.25);
    ASSERT_STREQ_LEN(written, buff, "1.2e+00");

    written = __llvm_libc::sprintf(buff, "%.1e", 1.125);
    ASSERT_STREQ_LEN(written, buff, "1.1e+00");

    written = __llvm_libc::sprintf(buff, "%.1e", 1.625);
    ASSERT_STREQ_LEN(written, buff, "1.6e+00");

    written = __llvm_libc::sprintf(buff, "%.1e", 1.375);
    ASSERT_STREQ_LEN(written, buff, "1.4e+00");

    written = __llvm_libc::sprintf(buff, "%.1e", 1.875);
    ASSERT_STREQ_LEN(written, buff, "1.9e+00");

    written = __llvm_libc::sprintf(buff, "%.1e", -1.75);
    ASSERT_STREQ_LEN(written, buff, "-1.8e+00");

    written = __llvm_libc::sprintf(buff, "%.1e", -1.25);
    ASSERT_STREQ_LEN(written, buff, "-1.2e+00");

    written = __llvm_libc::sprintf(buff, "%.1e", -1.125);
    ASSERT_STREQ_LEN(written, buff, "-1.1e+00");

    written = __llvm_libc::sprintf(buff, "%.1e", -1.625);
    ASSERT_STREQ_LEN(written, buff, "-1.6e+00");

    written = __llvm_libc::sprintf(buff, "%.1e", -1.375);
    ASSERT_STREQ_LEN(written, buff, "-1.4e+00");

    written = __llvm_libc::sprintf(buff, "%.1e", -1.875);
    ASSERT_STREQ_LEN(written, buff, "-1.9e+00");
  }

  {
    __llvm_libc::testutils::ForceRoundingMode r(
        __llvm_libc::testutils::RoundingMode::Upward);

    written = __llvm_libc::sprintf(buff, "%.1e", 1.75);
    ASSERT_STREQ_LEN(written, buff, "1.8e+00");

    written = __llvm_libc::sprintf(buff, "%.1e", 1.25);
    ASSERT_STREQ_LEN(written, buff, "1.3e+00");

    written = __llvm_libc::sprintf(buff, "%.1e", 1.125);
    ASSERT_STREQ_LEN(written, buff, "1.2e+00");

    written = __llvm_libc::sprintf(buff, "%.1e", 1.625);
    ASSERT_STREQ_LEN(written, buff, "1.7e+00");

    written = __llvm_libc::sprintf(buff, "%.1e", 1.375);
    ASSERT_STREQ_LEN(written, buff, "1.4e+00");

    written = __llvm_libc::sprintf(buff, "%.1e", 1.875);
    ASSERT_STREQ_LEN(written, buff, "1.9e+00");

    written = __llvm_libc::sprintf(buff, "%.1e", -1.75);
    ASSERT_STREQ_LEN(written, buff, "-1.7e+00");

    written = __llvm_libc::sprintf(buff, "%.1e", -1.25);
    ASSERT_STREQ_LEN(written, buff, "-1.2e+00");

    written = __llvm_libc::sprintf(buff, "%.1e", -1.125);
    ASSERT_STREQ_LEN(written, buff, "-1.1e+00");

    written = __llvm_libc::sprintf(buff, "%.1e", -1.625);
    ASSERT_STREQ_LEN(written, buff, "-1.6e+00");

    written = __llvm_libc::sprintf(buff, "%.1e", -1.375);
    ASSERT_STREQ_LEN(written, buff, "-1.3e+00");

    written = __llvm_libc::sprintf(buff, "%.1e", -1.875);
    ASSERT_STREQ_LEN(written, buff, "-1.8e+00");
  }

  {
    __llvm_libc::testutils::ForceRoundingMode r(
        __llvm_libc::testutils::RoundingMode::Downward);

    written = __llvm_libc::sprintf(buff, "%.1e", 1.75);
    ASSERT_STREQ_LEN(written, buff, "1.7e+00");

    written = __llvm_libc::sprintf(buff, "%.1e", 1.25);
    ASSERT_STREQ_LEN(written, buff, "1.2e+00");

    written = __llvm_libc::sprintf(buff, "%.1e", 1.125);
    ASSERT_STREQ_LEN(written, buff, "1.1e+00");

    written = __llvm_libc::sprintf(buff, "%.1e", 1.625);
    ASSERT_STREQ_LEN(written, buff, "1.6e+00");

    written = __llvm_libc::sprintf(buff, "%.1e", 1.375);
    ASSERT_STREQ_LEN(written, buff, "1.3e+00");

    written = __llvm_libc::sprintf(buff, "%.1e", 1.875);
    ASSERT_STREQ_LEN(written, buff, "1.8e+00");

    written = __llvm_libc::sprintf(buff, "%.1e", -1.75);
    ASSERT_STREQ_LEN(written, buff, "-1.8e+00");

    written = __llvm_libc::sprintf(buff, "%.1e", -1.25);
    ASSERT_STREQ_LEN(written, buff, "-1.3e+00");

    written = __llvm_libc::sprintf(buff, "%.1e", -1.125);
    ASSERT_STREQ_LEN(written, buff, "-1.2e+00");

    written = __llvm_libc::sprintf(buff, "%.1e", -1.625);
    ASSERT_STREQ_LEN(written, buff, "-1.7e+00");

    written = __llvm_libc::sprintf(buff, "%.1e", -1.375);
    ASSERT_STREQ_LEN(written, buff, "-1.4e+00");

    written = __llvm_libc::sprintf(buff, "%.1e", -1.875);
    ASSERT_STREQ_LEN(written, buff, "-1.9e+00");
  }

  {
    __llvm_libc::testutils::ForceRoundingMode r(
        __llvm_libc::testutils::RoundingMode::TowardZero);

    written = __llvm_libc::sprintf(buff, "%.1e", 1.75);
    ASSERT_STREQ_LEN(written, buff, "1.7e+00");

    written = __llvm_libc::sprintf(buff, "%.1e", 1.25);
    ASSERT_STREQ_LEN(written, buff, "1.2e+00");

    written = __llvm_libc::sprintf(buff, "%.1e", 1.125);
    ASSERT_STREQ_LEN(written, buff, "1.1e+00");

    written = __llvm_libc::sprintf(buff, "%.1e", 1.625);
    ASSERT_STREQ_LEN(written, buff, "1.6e+00");

    written = __llvm_libc::sprintf(buff, "%.1e", 1.375);
    ASSERT_STREQ_LEN(written, buff, "1.3e+00");

    written = __llvm_libc::sprintf(buff, "%.1e", 1.875);
    ASSERT_STREQ_LEN(written, buff, "1.8e+00");

    written = __llvm_libc::sprintf(buff, "%.1e", -1.75);
    ASSERT_STREQ_LEN(written, buff, "-1.7e+00");

    written = __llvm_libc::sprintf(buff, "%.1e", -1.25);
    ASSERT_STREQ_LEN(written, buff, "-1.2e+00");

    written = __llvm_libc::sprintf(buff, "%.1e", -1.125);
    ASSERT_STREQ_LEN(written, buff, "-1.1e+00");

    written = __llvm_libc::sprintf(buff, "%.1e", -1.625);
    ASSERT_STREQ_LEN(written, buff, "-1.6e+00");

    written = __llvm_libc::sprintf(buff, "%.1e", -1.375);
    ASSERT_STREQ_LEN(written, buff, "-1.3e+00");

    written = __llvm_libc::sprintf(buff, "%.1e", -1.875);
    ASSERT_STREQ_LEN(written, buff, "-1.8e+00");
  }

  // Flag Tests.
  written = __llvm_libc::sprintf(buff, "%+e", 1.0);
  ASSERT_STREQ_LEN(written, buff, "+1.000000e+00");

  written = __llvm_libc::sprintf(buff, "%+e", -1.0);
  ASSERT_STREQ_LEN(written, buff, "-1.000000e+00");

  written = __llvm_libc::sprintf(buff, "% e", 1.0);
  ASSERT_STREQ_LEN(written, buff, " 1.000000e+00");

  written = __llvm_libc::sprintf(buff, "% e", -1.0);
  ASSERT_STREQ_LEN(written, buff, "-1.000000e+00");

  written = __llvm_libc::sprintf(buff, "%-15e", 1.5);
  ASSERT_STREQ_LEN(written, buff, "1.500000e+00   ");

  written = __llvm_libc::sprintf(buff, "%#.e", 1.0);
  ASSERT_STREQ_LEN(written, buff, "1.e+00");

  written = __llvm_libc::sprintf(buff, "%#.0e", 1.5);
  ASSERT_STREQ_LEN(written, buff, "2.e+00");

  written = __llvm_libc::sprintf(buff, "%015e", 1.5);
  ASSERT_STREQ_LEN(written, buff, "0001.500000e+00");

  written = __llvm_libc::sprintf(buff, "%015e", -1.5);
  ASSERT_STREQ_LEN(written, buff, "-001.500000e+00");

  written = __llvm_libc::sprintf(buff, "%+- #0e", 0.0);
  ASSERT_STREQ_LEN(written, buff, "+0.000000e+00");

  // Combined Tests.

  written = __llvm_libc::sprintf(buff, "%10.2e", 9.99);
  ASSERT_STREQ_LEN(written, buff, "  9.99e+00");

  written = __llvm_libc::sprintf(buff, "%10.1e", 9.99);
  ASSERT_STREQ_LEN(written, buff, "   1.0e+01");

  written = __llvm_libc::sprintf(buff, "%10.0e", 9.99);
  ASSERT_STREQ_LEN(written, buff, "     1e+01");

  written = __llvm_libc::sprintf(buff, "%10.0e", 0.0999);
  ASSERT_STREQ_LEN(written, buff, "     1e-01");

  written = __llvm_libc::sprintf(buff, "%-10.2e", 9.99);
  ASSERT_STREQ_LEN(written, buff, "9.99e+00  ");

  written = __llvm_libc::sprintf(buff, "%-10.1e", 9.99);
  ASSERT_STREQ_LEN(written, buff, "1.0e+01   ");

  written = __llvm_libc::sprintf(buff, "%-10.1e", 1.0e-50);
  ASSERT_STREQ_LEN(written, buff, "1.0e-50   ");

  written = __llvm_libc::sprintf(buff, "%30e", 1234567890123456789.0);
  ASSERT_STREQ_LEN(written, buff, "                  1.234568e+18");

  written = __llvm_libc::sprintf(buff, "%-30e", 1234567890123456789.0);
  ASSERT_STREQ_LEN(written, buff, "1.234568e+18                  ");

  written = __llvm_libc::sprintf(buff, "%25.14e", 9999999999999.99);
  ASSERT_STREQ_LEN(written, buff, "     9.99999999999999e+12");

  written = __llvm_libc::sprintf(buff, "%25.13e", 9999999999999.99);
  ASSERT_STREQ_LEN(written, buff, "      1.0000000000000e+13");

  written = __llvm_libc::sprintf(buff, "%25.12e", 9999999999999.99);
  ASSERT_STREQ_LEN(written, buff, "       1.000000000000e+13");

  written = __llvm_libc::sprintf(buff, "%12.3e %-12.3e", 0.1, 256.0);
  ASSERT_STREQ_LEN(written, buff, "   1.000e-01 2.560e+02   ");

  written = __llvm_libc::sprintf(buff, "%+-#12.3e % 012.3e", 0.1256, 1256.0);
  ASSERT_STREQ_LEN(written, buff, "+1.256e-01    001.256e+03");
}

TEST_F(LlvmLibcSPrintfTest, FloatAutoConv) {
  __llvm_libc::testutils::ForceRoundingMode r(
      __llvm_libc::testutils::RoundingMode::Nearest);
  double inf = __llvm_libc::fputil::FPBits<double>::inf().get_val();
  double nan = __llvm_libc::fputil::FPBits<double>::build_nan(1);

  written = __llvm_libc::sprintf(buff, "%g", 1.0);
  ASSERT_STREQ_LEN(written, buff, "1");

  written = __llvm_libc::sprintf(buff, "%G", -1.0);
  ASSERT_STREQ_LEN(written, buff, "-1");

  written = __llvm_libc::sprintf(buff, "%g", -1.234567);
  ASSERT_STREQ_LEN(written, buff, "-1.23457");

  written = __llvm_libc::sprintf(buff, "%g", 0.0);
  ASSERT_STREQ_LEN(written, buff, "0");

  written = __llvm_libc::sprintf(buff, "%g", -0.0);
  ASSERT_STREQ_LEN(written, buff, "-0");

  written = __llvm_libc::sprintf(buff, "%g", 1.5);
  ASSERT_STREQ_LEN(written, buff, "1.5");

  written = __llvm_libc::sprintf(buff, "%g", 1e300);
  ASSERT_STREQ_LEN(written, buff, "1e+300");

  written = __llvm_libc::sprintf(buff, "%g", 0.1);
  ASSERT_STREQ_LEN(written, buff, "0.1");

  written = __llvm_libc::sprintf(buff, "%g", 0.001);
  ASSERT_STREQ_LEN(written, buff, "0.001");

  written = __llvm_libc::sprintf(buff, "%g", 0.00001);
  ASSERT_STREQ_LEN(written, buff, "1e-05");

  written = __llvm_libc::sprintf(buff, "%g", 0.0000001);
  ASSERT_STREQ_LEN(written, buff, "1e-07");

  written = __llvm_libc::sprintf(buff, "%g", 0.000000001);
  ASSERT_STREQ_LEN(written, buff, "1e-09");

  written = __llvm_libc::sprintf(buff, "%g", 1.0e-20);
  ASSERT_STREQ_LEN(written, buff, "1e-20");

  written = __llvm_libc::sprintf(buff, "%g", 1234567890123456789.0);
  ASSERT_STREQ_LEN(written, buff, "1.23457e+18");

  written = __llvm_libc::sprintf(buff, "%g", 9999990000000.00);
  ASSERT_STREQ_LEN(written, buff, "9.99999e+12");

  written = __llvm_libc::sprintf(buff, "%g", 9999999000000.00);
  ASSERT_STREQ_LEN(written, buff, "1e+13");

  // Simple Subnormal Tests.

  written = __llvm_libc::sprintf(buff, "%g", 0x1.0p-1027);
  ASSERT_STREQ_LEN(written, buff, "6.95336e-310");

  written = __llvm_libc::sprintf(buff, "%g", 0x1.0p-1074);
  ASSERT_STREQ_LEN(written, buff, "4.94066e-324");

  // Inf/Nan Tests.

  written = __llvm_libc::sprintf(buff, "%g", inf);
  ASSERT_STREQ_LEN(written, buff, "inf");

  written = __llvm_libc::sprintf(buff, "%G", -inf);
  ASSERT_STREQ_LEN(written, buff, "-INF");

  written = __llvm_libc::sprintf(buff, "%g", nan);
  ASSERT_STREQ_LEN(written, buff, "nan");

  written = __llvm_libc::sprintf(buff, "%G", -nan);
  ASSERT_STREQ_LEN(written, buff, "-NAN");

  // Length Modifier Tests.

  // TODO: Uncomment the below tests after long double support is added
  /*
  written = __llvm_libc::sprintf(buff, "%Lf", 1e100L);
  ASSERT_STREQ_LEN(written, buff,
                   "99999999999999999996693535322073426194986990198284960792713"
                   "91541752018669482644324418977840117055488.000000");

  written = __llvm_libc::sprintf(buff, "%Lf", 1.0L);
  ASSERT_STREQ_LEN(written, buff, "1.000000");

  char big_buff[10000];
  written = __llvm_libc::sprintf(big_buff, "%Lf", 1e1000L);
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

  written = __llvm_libc::sprintf(big_buff, "%Lf", 1e4900L);
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
    written = __llvm_libc::sprintf(buff, "%La", 0.1L);
  #if defined(SPECIAL_X86_LONG_DOUBLE)
    ASSERT_STREQ_LEN(written, buff, "0xc.ccccccccccccccdp-7");
  #elif defined(LONG_DOUBLE_IS_DOUBLE)
    ASSERT_STREQ_LEN(written, buff, "0x1.999999999999ap-4");
  #else // 128 bit long double
    ASSERT_STREQ_LEN(written, buff, "0x1.999999999999999999999999999ap-4");
  #endif

    written = __llvm_libc::sprintf(buff, "%La", 1.0e1000L);
  #if defined(SPECIAL_X86_LONG_DOUBLE)
    ASSERT_STREQ_LEN(written, buff, "0xf.38db1f9dd3dac05p+3318");
  #elif defined(LONG_DOUBLE_IS_DOUBLE)
    ASSERT_STREQ_LEN(written, buff, "inf");
  #else // 128 bit long double
    ASSERT_STREQ_LEN(written, buff, "0x1.e71b63f3ba7b580af1a52d2a7379p+3321");
  #endif

    written = __llvm_libc::sprintf(buff, "%La", 1.0e-1000L);
  #if defined(SPECIAL_X86_LONG_DOUBLE)
    ASSERT_STREQ_LEN(written, buff, "0x8.68a9188a89e1467p-3325");
  #elif defined(LONG_DOUBLE_IS_DOUBLE)
    ASSERT_STREQ_LEN(written, buff, "0x0p+0");
  #else // 128 bit long double
    ASSERT_STREQ_LEN(written, buff, "0x1.0d152311513c28ce202627c06ec2p-3322");
  #endif
  */

  // Min Width Tests.

  written = __llvm_libc::sprintf(buff, "%15g", 1.0);
  ASSERT_STREQ_LEN(written, buff, "              1");

  written = __llvm_libc::sprintf(buff, "%15g", -1.0);
  ASSERT_STREQ_LEN(written, buff, "             -1");

  written = __llvm_libc::sprintf(buff, "%15g", 1.0e5);
  ASSERT_STREQ_LEN(written, buff, "         100000");

  written = __llvm_libc::sprintf(buff, "%15g", -1.0e5);
  ASSERT_STREQ_LEN(written, buff, "        -100000");

  written = __llvm_libc::sprintf(buff, "%10g", 1.0e-5);
  ASSERT_STREQ_LEN(written, buff, "     1e-05");

  // Precision Tests.

  written = __llvm_libc::sprintf(buff, "%.2g", 1.23456789);
  ASSERT_STREQ_LEN(written, buff, "1.2");

  // Trimming trailing zeroes causes the precision to be ignored here.
  written = __llvm_libc::sprintf(buff, "%.1g", 0.0);
  ASSERT_STREQ_LEN(written, buff, "0");

  written = __llvm_libc::sprintf(buff, "%.0g", 0.0);
  ASSERT_STREQ_LEN(written, buff, "0");

  written = __llvm_libc::sprintf(buff, "%.2g", 0.1);
  ASSERT_STREQ_LEN(written, buff, "0.1");

  written = __llvm_libc::sprintf(buff, "%.2g", 1.09);
  ASSERT_STREQ_LEN(written, buff, "1.1");

  written = __llvm_libc::sprintf(buff, "%.2g", 1.04);
  ASSERT_STREQ_LEN(written, buff, "1");

  written = __llvm_libc::sprintf(buff, "%.2g", 1.19);
  ASSERT_STREQ_LEN(written, buff, "1.2");

  written = __llvm_libc::sprintf(buff, "%.2g", 1.99);
  ASSERT_STREQ_LEN(written, buff, "2");

  written = __llvm_libc::sprintf(buff, "%.2g", 9.99);
  ASSERT_STREQ_LEN(written, buff, "10");

  written = __llvm_libc::sprintf(buff, "%.3g", 99.9);
  ASSERT_STREQ_LEN(written, buff, "99.9");

  written = __llvm_libc::sprintf(buff, "%.2g", 99.9);
  ASSERT_STREQ_LEN(written, buff, "1e+02");

  written = __llvm_libc::sprintf(buff, "%.1g", 99.9);
  ASSERT_STREQ_LEN(written, buff, "1e+02");

  written = __llvm_libc::sprintf(buff, "%.5g", 1.25);
  ASSERT_STREQ_LEN(written, buff, "1.25");

  written = __llvm_libc::sprintf(buff, "%.0g", 1.25);
  ASSERT_STREQ_LEN(written, buff, "1");

  written = __llvm_libc::sprintf(buff, "%.0g", 1.75);
  ASSERT_STREQ_LEN(written, buff, "2");

  written = __llvm_libc::sprintf(buff, "%.20g", 1.234e-10);
  ASSERT_STREQ_LEN(written, buff, "1.2340000000000000814e-10");

  written = __llvm_libc::sprintf(buff, "%.3g", -9.99);
  ASSERT_STREQ_LEN(written, buff, "-9.99");

  written = __llvm_libc::sprintf(buff, "%.2g", -9.99);
  ASSERT_STREQ_LEN(written, buff, "-10");

  written = __llvm_libc::sprintf(buff, "%.1g", -9.99);
  ASSERT_STREQ_LEN(written, buff, "-1e+01");

  written = __llvm_libc::sprintf(buff, "%.5g", 1.008);
  ASSERT_STREQ_LEN(written, buff, "1.008");

  written = __llvm_libc::sprintf(buff, "%.5g", 1.008e3);
  ASSERT_STREQ_LEN(written, buff, "1008");

  written = __llvm_libc::sprintf(buff, "%.4g", 9999.0);
  ASSERT_STREQ_LEN(written, buff, "9999");

  written = __llvm_libc::sprintf(buff, "%.3g", 9999.0);
  ASSERT_STREQ_LEN(written, buff, "1e+04");

  written = __llvm_libc::sprintf(buff, "%.3g", 1256.0);
  ASSERT_STREQ_LEN(written, buff, "1.26e+03");

  // Subnormal Precision Tests

  written = __llvm_libc::sprintf(buff, "%.310g", 0x1.0p-1022);
  ASSERT_STREQ_LEN(
      written, buff,
      "2."
      "225073858507201383090232717332404064219215980462331830553327416887204434"
      "813918195854283159012511020564067339731035811005152434161553460108856012"
      "385377718821130777993532002330479610147442583636071921565046942503734208"
      "375250806650616658158948720491179968591639648500635908770118304874799780"
      "887753749949451580452e-308");

  written = __llvm_libc::sprintf(buff, "%.30g", 0x1.0p-1022);
  ASSERT_STREQ_LEN(written, buff, "2.22507385850720138309023271733e-308");

  written = __llvm_libc::sprintf(buff, "%.310g", 0x1.0p-1023);
  ASSERT_STREQ_LEN(
      written, buff,
      "1."
      "112536929253600691545116358666202032109607990231165915276663708443602217"
      "406959097927141579506255510282033669865517905502576217080776730054428006"
      "192688859410565388996766001165239805073721291818035960782523471251867104"
      "187625403325308329079474360245589984295819824250317954385059152437399890"
      "443876874974725790226e-308");

  written = __llvm_libc::sprintf(buff, "%.7g", 9.99999e-310);
  ASSERT_STREQ_LEN(written, buff, "9.99999e-310");

  written = __llvm_libc::sprintf(buff, "%.6g", 9.99999e-310);
  ASSERT_STREQ_LEN(written, buff, "9.99999e-310");

  written = __llvm_libc::sprintf(buff, "%.5g", 9.99999e-310);
  ASSERT_STREQ_LEN(written, buff, "1e-309");

  written = __llvm_libc::sprintf(buff, "%.4g", 9.99999e-310);
  ASSERT_STREQ_LEN(written, buff, "1e-309");

  written = __llvm_libc::sprintf(buff, "%.3g", 9.99999e-310);
  ASSERT_STREQ_LEN(written, buff, "1e-309");

  written = __llvm_libc::sprintf(buff, "%.2g", 9.99999e-310);
  ASSERT_STREQ_LEN(written, buff, "1e-309");

  written = __llvm_libc::sprintf(buff, "%.1g", 9.99999e-310);
  ASSERT_STREQ_LEN(written, buff, "1e-309");

  written = __llvm_libc::sprintf(buff, "%.0g", 9.99999e-310);
  ASSERT_STREQ_LEN(written, buff, "1e-309");

  written = __llvm_libc::sprintf(buff, "%.10g", 0x1.0p-1074);
  ASSERT_STREQ_LEN(written, buff, "4.940656458e-324");

  // Long double precision tests.
  // These are currently commented out because they require long double support
  // that isn't ready yet.
  /*
    written = __llvm_libc::sprintf(buff, "%.1La", 0.1L);
  #if defined(SPECIAL_X86_LONG_DOUBLE)
    ASSERT_STREQ_LEN(written, buff, "0xc.dp-7");
  #elif defined(LONG_DOUBLE_IS_DOUBLE)
    ASSERT_STREQ_LEN(written, buff, "0x1.ap-4");
  #else // 128 bit long double
    ASSERT_STREQ_LEN(written, buff, "0x1.ap-4");
  #endif

    written = __llvm_libc::sprintf(buff, "%.1La", 0xf.fffffffffffffffp16380L);
  #if defined(SPECIAL_X86_LONG_DOUBLE)
    ASSERT_STREQ_LEN(written, buff, "0x1.0p+16384");
  #elif defined(LONG_DOUBLE_IS_DOUBLE)
    ASSERT_STREQ_LEN(written, buff, "inf");
  #else // 128 bit long double
    ASSERT_STREQ_LEN(written, buff, "0x2.0p+16383");
  #endif
  */

  // Rounding Mode Tests.

  {
    __llvm_libc::testutils::ForceRoundingMode r(
        __llvm_libc::testutils::RoundingMode::Nearest);

    written = __llvm_libc::sprintf(buff, "%.2g", 1.75);
    ASSERT_STREQ_LEN(written, buff, "1.8");

    written = __llvm_libc::sprintf(buff, "%.2g", 1.25);
    ASSERT_STREQ_LEN(written, buff, "1.2");

    written = __llvm_libc::sprintf(buff, "%.2g", 1.125);
    ASSERT_STREQ_LEN(written, buff, "1.1");

    written = __llvm_libc::sprintf(buff, "%.2g", 1.625);
    ASSERT_STREQ_LEN(written, buff, "1.6");

    written = __llvm_libc::sprintf(buff, "%.2g", 1.375);
    ASSERT_STREQ_LEN(written, buff, "1.4");

    written = __llvm_libc::sprintf(buff, "%.2g", 1.875);
    ASSERT_STREQ_LEN(written, buff, "1.9");

    written = __llvm_libc::sprintf(buff, "%.2g", -1.75);
    ASSERT_STREQ_LEN(written, buff, "-1.8");

    written = __llvm_libc::sprintf(buff, "%.2g", -1.25);
    ASSERT_STREQ_LEN(written, buff, "-1.2");

    written = __llvm_libc::sprintf(buff, "%.2g", -1.125);
    ASSERT_STREQ_LEN(written, buff, "-1.1");

    written = __llvm_libc::sprintf(buff, "%.2g", -1.625);
    ASSERT_STREQ_LEN(written, buff, "-1.6");

    written = __llvm_libc::sprintf(buff, "%.2g", -1.375);
    ASSERT_STREQ_LEN(written, buff, "-1.4");

    written = __llvm_libc::sprintf(buff, "%.2g", -1.875);
    ASSERT_STREQ_LEN(written, buff, "-1.9");
  }

  {
    __llvm_libc::testutils::ForceRoundingMode r(
        __llvm_libc::testutils::RoundingMode::Upward);

    written = __llvm_libc::sprintf(buff, "%.2g", 1.75);
    ASSERT_STREQ_LEN(written, buff, "1.8");

    written = __llvm_libc::sprintf(buff, "%.2g", 1.25);
    ASSERT_STREQ_LEN(written, buff, "1.3");

    written = __llvm_libc::sprintf(buff, "%.2g", 1.125);
    ASSERT_STREQ_LEN(written, buff, "1.2");

    written = __llvm_libc::sprintf(buff, "%.2g", 1.625);
    ASSERT_STREQ_LEN(written, buff, "1.7");

    written = __llvm_libc::sprintf(buff, "%.2g", 1.375);
    ASSERT_STREQ_LEN(written, buff, "1.4");

    written = __llvm_libc::sprintf(buff, "%.2g", 1.875);
    ASSERT_STREQ_LEN(written, buff, "1.9");

    written = __llvm_libc::sprintf(buff, "%.2g", -1.75);
    ASSERT_STREQ_LEN(written, buff, "-1.7");

    written = __llvm_libc::sprintf(buff, "%.2g", -1.25);
    ASSERT_STREQ_LEN(written, buff, "-1.2");

    written = __llvm_libc::sprintf(buff, "%.2g", -1.125);
    ASSERT_STREQ_LEN(written, buff, "-1.1");

    written = __llvm_libc::sprintf(buff, "%.2g", -1.625);
    ASSERT_STREQ_LEN(written, buff, "-1.6");

    written = __llvm_libc::sprintf(buff, "%.2g", -1.375);
    ASSERT_STREQ_LEN(written, buff, "-1.3");

    written = __llvm_libc::sprintf(buff, "%.2g", -1.875);
    ASSERT_STREQ_LEN(written, buff, "-1.8");
  }

  {
    __llvm_libc::testutils::ForceRoundingMode r(
        __llvm_libc::testutils::RoundingMode::Downward);

    written = __llvm_libc::sprintf(buff, "%.2g", 1.75);
    ASSERT_STREQ_LEN(written, buff, "1.7");

    written = __llvm_libc::sprintf(buff, "%.2g", 1.25);
    ASSERT_STREQ_LEN(written, buff, "1.2");

    written = __llvm_libc::sprintf(buff, "%.2g", 1.125);
    ASSERT_STREQ_LEN(written, buff, "1.1");

    written = __llvm_libc::sprintf(buff, "%.2g", 1.625);
    ASSERT_STREQ_LEN(written, buff, "1.6");

    written = __llvm_libc::sprintf(buff, "%.2g", 1.375);
    ASSERT_STREQ_LEN(written, buff, "1.3");

    written = __llvm_libc::sprintf(buff, "%.2g", 1.875);
    ASSERT_STREQ_LEN(written, buff, "1.8");

    written = __llvm_libc::sprintf(buff, "%.2g", -1.75);
    ASSERT_STREQ_LEN(written, buff, "-1.8");

    written = __llvm_libc::sprintf(buff, "%.2g", -1.25);
    ASSERT_STREQ_LEN(written, buff, "-1.3");

    written = __llvm_libc::sprintf(buff, "%.2g", -1.125);
    ASSERT_STREQ_LEN(written, buff, "-1.2");

    written = __llvm_libc::sprintf(buff, "%.2g", -1.625);
    ASSERT_STREQ_LEN(written, buff, "-1.7");

    written = __llvm_libc::sprintf(buff, "%.2g", -1.375);
    ASSERT_STREQ_LEN(written, buff, "-1.4");

    written = __llvm_libc::sprintf(buff, "%.2g", -1.875);
    ASSERT_STREQ_LEN(written, buff, "-1.9");
  }

  {
    __llvm_libc::testutils::ForceRoundingMode r(
        __llvm_libc::testutils::RoundingMode::TowardZero);

    written = __llvm_libc::sprintf(buff, "%.2g", 1.75);
    ASSERT_STREQ_LEN(written, buff, "1.7");

    written = __llvm_libc::sprintf(buff, "%.2g", 1.25);
    ASSERT_STREQ_LEN(written, buff, "1.2");

    written = __llvm_libc::sprintf(buff, "%.2g", 1.125);
    ASSERT_STREQ_LEN(written, buff, "1.1");

    written = __llvm_libc::sprintf(buff, "%.2g", 1.625);
    ASSERT_STREQ_LEN(written, buff, "1.6");

    written = __llvm_libc::sprintf(buff, "%.2g", 1.375);
    ASSERT_STREQ_LEN(written, buff, "1.3");

    written = __llvm_libc::sprintf(buff, "%.2g", 1.875);
    ASSERT_STREQ_LEN(written, buff, "1.8");

    written = __llvm_libc::sprintf(buff, "%.2g", -1.75);
    ASSERT_STREQ_LEN(written, buff, "-1.7");

    written = __llvm_libc::sprintf(buff, "%.2g", -1.25);
    ASSERT_STREQ_LEN(written, buff, "-1.2");

    written = __llvm_libc::sprintf(buff, "%.2g", -1.125);
    ASSERT_STREQ_LEN(written, buff, "-1.1");

    written = __llvm_libc::sprintf(buff, "%.2g", -1.625);
    ASSERT_STREQ_LEN(written, buff, "-1.6");

    written = __llvm_libc::sprintf(buff, "%.2g", -1.375);
    ASSERT_STREQ_LEN(written, buff, "-1.3");

    written = __llvm_libc::sprintf(buff, "%.2g", -1.875);
    ASSERT_STREQ_LEN(written, buff, "-1.8");
  }

  // Flag Tests.
  written = __llvm_libc::sprintf(buff, "%+g", 1.0);
  ASSERT_STREQ_LEN(written, buff, "+1");

  written = __llvm_libc::sprintf(buff, "%+g", -1.0);
  ASSERT_STREQ_LEN(written, buff, "-1");

  written = __llvm_libc::sprintf(buff, "% g", 1.0);
  ASSERT_STREQ_LEN(written, buff, " 1");

  written = __llvm_libc::sprintf(buff, "% g", -1.0);
  ASSERT_STREQ_LEN(written, buff, "-1");

  written = __llvm_libc::sprintf(buff, "%-15g", 1.5);
  ASSERT_STREQ_LEN(written, buff, "1.5            ");

  written = __llvm_libc::sprintf(buff, "%#.g", 1.0);
  ASSERT_STREQ_LEN(written, buff, "1.");

  written = __llvm_libc::sprintf(buff, "%#g", 1.0);
  ASSERT_STREQ_LEN(written, buff, "1.00000");

  written = __llvm_libc::sprintf(buff, "%#.0g", 1.5);
  ASSERT_STREQ_LEN(written, buff, "2.");

  written = __llvm_libc::sprintf(buff, "%015g", 1.5);
  ASSERT_STREQ_LEN(written, buff, "0000000000001.5");

  written = __llvm_libc::sprintf(buff, "%015g", -1.5);
  ASSERT_STREQ_LEN(written, buff, "-000000000001.5");

  written = __llvm_libc::sprintf(buff, "%+- #0g", 0.0);
  ASSERT_STREQ_LEN(written, buff, "+0.00000");

  // Combined Tests.

  written = __llvm_libc::sprintf(buff, "%10.3g", 9.99);
  ASSERT_STREQ_LEN(written, buff, "      9.99");

  written = __llvm_libc::sprintf(buff, "%10.2g", 9.99);
  ASSERT_STREQ_LEN(written, buff, "        10");

  written = __llvm_libc::sprintf(buff, "%10.1g", 9.99);
  ASSERT_STREQ_LEN(written, buff, "     1e+01");

  written = __llvm_libc::sprintf(buff, "%-10.3g", 9.99);
  ASSERT_STREQ_LEN(written, buff, "9.99      ");

  written = __llvm_libc::sprintf(buff, "%-10.2g", 9.99);
  ASSERT_STREQ_LEN(written, buff, "10        ");

  written = __llvm_libc::sprintf(buff, "%-10.1g", 9.99);
  ASSERT_STREQ_LEN(written, buff, "1e+01     ");

  written = __llvm_libc::sprintf(buff, "%-10.1g", 1.0e-50);
  ASSERT_STREQ_LEN(written, buff, "1e-50     ");

  written = __llvm_libc::sprintf(buff, "%30g", 1234567890123456789.0);
  ASSERT_STREQ_LEN(written, buff, "                   1.23457e+18");

  written = __llvm_libc::sprintf(buff, "%-30g", 1234567890123456789.0);
  ASSERT_STREQ_LEN(written, buff, "1.23457e+18                   ");

  written = __llvm_libc::sprintf(buff, "%25.15g", 9999999999999.99);
  ASSERT_STREQ_LEN(written, buff, "         9999999999999.99");

  written = __llvm_libc::sprintf(buff, "%25.14g", 9999999999999.99);
  ASSERT_STREQ_LEN(written, buff, "           10000000000000");

  written = __llvm_libc::sprintf(buff, "%25.13g", 9999999999999.99);
  ASSERT_STREQ_LEN(written, buff, "                    1e+13");

  written = __llvm_libc::sprintf(buff, "%#12.3g %-12.3g", 0.1, 256.0);
  ASSERT_STREQ_LEN(written, buff, "       0.100 256         ");

  written = __llvm_libc::sprintf(buff, "%+-#12.3g % 012.3g", 0.1256, 1256.0);
  ASSERT_STREQ_LEN(written, buff, "+0.126        0001.26e+03");
}

#endif // LIBC_COPT_PRINTF_DISABLE_FLOAT

#ifndef LIBC_COPT_PRINTF_DISABLE_WRITE_INT
TEST(LlvmLibcSPrintfTest, WriteIntConv) {
  char buff[64];
  int written;
  int test_val = -1;

  test_val = -1;
  written = __llvm_libc::sprintf(buff, "12345%n67890", &test_val);
  EXPECT_EQ(written, 10);
  EXPECT_EQ(test_val, 5);
  ASSERT_STREQ(buff, "1234567890");

  test_val = -1;
  written = __llvm_libc::sprintf(buff, "%n", &test_val);
  EXPECT_EQ(written, 0);
  EXPECT_EQ(test_val, 0);
  ASSERT_STREQ(buff, "");

  test_val = 0x100;
  written = __llvm_libc::sprintf(buff, "ABC%hhnDEF", &test_val);
  EXPECT_EQ(written, 6);
  EXPECT_EQ(test_val, 0x103);
  ASSERT_STREQ(buff, "ABCDEF");

  test_val = -1;
  written = __llvm_libc::sprintf(buff, "%s%n", "87654321", &test_val);
  EXPECT_EQ(written, 8);
  EXPECT_EQ(test_val, 8);
  ASSERT_STREQ(buff, "87654321");

  written = __llvm_libc::sprintf(buff, "abc123%n", nullptr);
  EXPECT_LT(written, 0);
}
#endif // LIBC_COPT_PRINTF_DISABLE_WRITE_INT

#ifndef LIBC_COPT_PRINTF_DISABLE_INDEX_MODE
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
#endif // LIBC_COPT_PRINTF_DISABLE_INDEX_MODE

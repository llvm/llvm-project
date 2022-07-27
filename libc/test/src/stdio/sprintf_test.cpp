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
#include "utils/UnitTest/Test.h"
#include "utils/testutils/RoundingModeUtils.h"

class LlvmLibcSPrintfTest : public __llvm_libc::testing::Test {
protected:
  char buff[64];
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

#ifndef LLVM_LIBC_PRINTF_DISABLE_FLOAT

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
#endif // LLVM_LIBC_PRINTF_DISABLE_FLOAT

#ifndef LLVM_LIBC_PRINTF_DISABLE_WRITE_INT
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
#endif // LLVM_LIBC_PRINTF_DISABLE_WRITE_INT

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

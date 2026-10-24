//===-- Unittests for swprintf --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for swprintf.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"
#include "hdr/types/wint_t.h"
#include "hdr/wchar_macros.h"
#include "src/__support/CPP/limits.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/macros/config.h"
#include "src/__support/printf_core/printf_config.h"
#include "src/wchar/swprintf.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/RoundingModeUtils.h"
#include "test/UnitTest/Test.h"

#include <inttypes.h>

using LIBC_NAMESPACE::fputil::testing::ForceRoundingMode;
using LIBC_NAMESPACE::fputil::testing::RoundingMode;

using LlvmLibcSwprintfTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

#define ASSERT_STREQ_LEN(actual_written, actual_str, expected_str)             \
  EXPECT_EQ(actual_written,                                                    \
            static_cast<int>(sizeof(expected_str) / sizeof(wchar_t) - 1));     \
  EXPECT_STREQ(actual_str, expected_str);

TEST_F(LlvmLibcSwprintfTest, SimpleNoConv) {
  constexpr size_t N = 64;
  wchar_t buff[N];
  int written;

  written = LIBC_NAMESPACE::swprintf(
      buff, N, L"A simple string with no conversions 😀.");
  ASSERT_STREQ_LEN(written, buff, L"A simple string with no conversions 😀.");
}

TEST_F(LlvmLibcSwprintfTest, PercentConv) {
  constexpr size_t N = 64;
  wchar_t buff[N];
  int written;

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%%");
  ASSERT_STREQ_LEN(written, buff, L"%");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"abc %% def");
  ASSERT_STREQ_LEN(written, buff, L"abc % def");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%%%%%%");
  ASSERT_STREQ_LEN(written, buff, L"%%%");
}

TEST_F(LlvmLibcSwprintfTest, CharConv) {
  constexpr size_t N = 64;
  wchar_t buff[N];
  int written;

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%c", 'a');
  ASSERT_STREQ_LEN(written, buff, L"a");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%3c %-3c", '1', '2');
  ASSERT_STREQ_LEN(written, buff, L"  1 2  ");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%*c", 2, '3');
  ASSERT_STREQ_LEN(written, buff, L" 3");
}

TEST_F(LlvmLibcSwprintfTest, StringConv) {
  constexpr size_t N = 64;
  wchar_t buff[N];
  int written;

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%s", "abcDEF123");
  ASSERT_STREQ_LEN(written, buff, L"abcDEF123");

  written =
      LIBC_NAMESPACE::swprintf(buff, N, L"%10s %-10s", "centered", "title");
  ASSERT_STREQ_LEN(written, buff, L"  centered title     ");

  written = LIBC_NAMESPACE::swprintf(
      buff, N, L"%-5.4s%-4.4s", "words can describe", "soups most delicious");
  ASSERT_STREQ_LEN(written, buff, L"word soup");

  written =
      LIBC_NAMESPACE::swprintf(buff, N, L"%*s %.*s %*.*s", 10, "beginning", 2,
                               "isn't", 12, 10, "important. Ever.");
  ASSERT_STREQ_LEN(written, buff, L" beginning is   important.");

#ifndef LIBC_COPT_PRINTF_NO_NULLPTR_CHECKS
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%s", nullptr);
  ASSERT_STREQ_LEN(written, buff, L"(null)");
#endif // LIBC_COPT_PRINTF_NO_NULLPTR_CHECKS
}

TEST_F(LlvmLibcSwprintfTest, IntConv) {
  constexpr size_t N = 64;
  wchar_t buff[N];
  int written;

  // Basic Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%d", 123);
  ASSERT_STREQ_LEN(written, buff, L"123");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%i", -456);
  ASSERT_STREQ_LEN(written, buff, L"-456");

  // Length Modifier Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%hhu", 257); // 0x101
  ASSERT_STREQ_LEN(written, buff, L"1");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%llu", 18446744073709551615ull);
  ASSERT_STREQ_LEN(written, buff, L"18446744073709551615"); // ull max

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%u", ~0);
  if (sizeof(int) == 4) {
    ASSERT_STREQ_LEN(written, buff, L"4294967295");
  }

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%tu", ~ptrdiff_t(0));
  if (sizeof(ptrdiff_t) == 8) {
    ASSERT_STREQ_LEN(written, buff, L"18446744073709551615");
  } else if (sizeof(ptrdiff_t) == 4) {
    ASSERT_STREQ_LEN(written, buff, L"4294967295");
  }

  written =
      LIBC_NAMESPACE::swprintf(buff, N, L"%lld", -9223372036854775807ll - 1ll);
  ASSERT_STREQ_LEN(written, buff, L"-9223372036854775808"); // ll min

#ifndef LIBC_COPT_PRINTF_DISABLE_BITINT
  // Bit int width tests
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%w3d", 5807);
  ASSERT_STREQ_LEN(written, buff, L"7");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%w3d", 1);
  ASSERT_STREQ_LEN(written, buff, L"1");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%w64d", 9223372036854775807ll);
  ASSERT_STREQ_LEN(written, buff, L"9223372036854775807");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%w-1d", 5807);
  ASSERT_STREQ_LEN(written, buff, L"%w-1d");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%w0d", 5807);
  ASSERT_STREQ_LEN(written, buff, L"%w0d");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%w999d", 9223372036854775807ll);
  ASSERT_STREQ_LEN(written, buff, L"9223372036854775807");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%winvalid%w1d", 5807, 5807);
  ASSERT_STREQ_LEN(written, buff, L"%winvalid1");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%w-1d%w1d", 5807, 5807);
  ASSERT_STREQ_LEN(written, buff, L"%w-1d1");

  wchar_t format[64];
  wchar_t uintmax[128];
  LIBC_NAMESPACE::swprintf(format, sizeof(format) / sizeof(wchar_t), L"%%w%du",
                           sizeof(uintmax_t) * CHAR_BIT);
  const int uintmax_len =
      LIBC_NAMESPACE::swprintf(uintmax, sizeof(uintmax) / sizeof(wchar_t),
                               L"%ju", sizeof(uintmax_t) * CHAR_BIT);
  written =
      LIBC_NAMESPACE::swprintf(buff, N, format, sizeof(uintmax_t) * CHAR_BIT);
  EXPECT_EQ(written, uintmax_len);
  ASSERT_STREQ(buff, uintmax);

  written =
      LIBC_NAMESPACE::swprintf(buff, N, L"%w64u", 18446744073709551615ull);
  ASSERT_STREQ_LEN(written, buff, L"18446744073709551615"); // ull max

  written =
      LIBC_NAMESPACE::swprintf(buff, N, L"%w64d", -9223372036854775807ll - 1ll);
  ASSERT_STREQ_LEN(written, buff, L"-9223372036854775808"); // ll min

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%wf3d", 5807);
  ASSERT_STREQ_LEN(written, buff, L"7");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%wf3d", 1);
  ASSERT_STREQ_LEN(written, buff, L"1");

  written =
      LIBC_NAMESPACE::swprintf(buff, N, L"%wf64u", 18446744073709551615ull);
  ASSERT_STREQ_LEN(written, buff, L"18446744073709551615"); // ull max

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%wf64d",
                                     -9223372036854775807ll - 1ll);
  ASSERT_STREQ_LEN(written, buff, L"-9223372036854775808"); // ll min

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%wf0d", 5807);
  ASSERT_STREQ_LEN(written, buff, L"%wf0d");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%wf-1d", 5807);
  ASSERT_STREQ_LEN(written, buff, L"%wf-1d");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%wfinvalid%wf1d", 5807, 5807);
  ASSERT_STREQ_LEN(written, buff, L"%wfinvalid1");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%wf-1d%wf1d", 5807, 5807);
  ASSERT_STREQ_LEN(written, buff, L"%wf-1d1");

  written =
      LIBC_NAMESPACE::swprintf(buff, N, L"%wf999d", 9223372036854775807ll);
  ASSERT_STREQ_LEN(written, buff, L"9223372036854775807");
#endif // LIBC_COPT_PRINTF_DISABLE_BITINT

  // Min Width Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%4d", 789);
  ASSERT_STREQ_LEN(written, buff, L" 789");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%2d", 987);
  ASSERT_STREQ_LEN(written, buff, L"987");

  // Precision Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%d", 0);
  ASSERT_STREQ_LEN(written, buff, L"0");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.0d", 0);
  ASSERT_STREQ_LEN(written, buff, L"");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.5d", 654);
  ASSERT_STREQ_LEN(written, buff, L"00654");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.5d", -321);
  ASSERT_STREQ_LEN(written, buff, L"-00321");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2d", 135);
  ASSERT_STREQ_LEN(written, buff, L"135");

  // Flag Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.5d", -321);
  ASSERT_STREQ_LEN(written, buff, L"-00321");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%-5d", 246);
  ASSERT_STREQ_LEN(written, buff, L"246  ");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%-5d", -147);
  ASSERT_STREQ_LEN(written, buff, L"-147 ");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%+d", 258);
  ASSERT_STREQ_LEN(written, buff, L"+258");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"% d", 369);
  ASSERT_STREQ_LEN(written, buff, L" 369");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%05d", 470);
  ASSERT_STREQ_LEN(written, buff, L"00470");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%05d", -581);
  ASSERT_STREQ_LEN(written, buff, L"-0581");

  // Combined Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%+ u", 692);
  ASSERT_STREQ_LEN(written, buff, L"692");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%+ -05d", 703);
  ASSERT_STREQ_LEN(written, buff, L"+703 ");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%7.5d", 814);
  ASSERT_STREQ_LEN(written, buff, L"  00814");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%7.5d", -925);
  ASSERT_STREQ_LEN(written, buff, L" -00925");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%7.5d", 159);
  ASSERT_STREQ_LEN(written, buff, L"  00159");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"% -7.5d", 260);
  ASSERT_STREQ_LEN(written, buff, L" 00260 ");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%5.4d", 10000);
  ASSERT_STREQ_LEN(written, buff, L"10000");

  // Multiple Conversion Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%10d %-10d", 456, -789);
  ASSERT_STREQ_LEN(written, buff, L"       456 -789      ");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%-5.4d%+.4u", 75, 25);
  ASSERT_STREQ_LEN(written, buff, L"0075 0025");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"% 05hhi %+-0.5llu %-+ 06.3zd",
                                     256 + 127, 68719476736ll, size_t(2));
  ASSERT_STREQ_LEN(written, buff, L" 0127 68719476736 +002  ");
}

TEST_F(LlvmLibcSwprintfTest, HexConv) {
  constexpr size_t N = 64;
  wchar_t buff[N];
  int written;

  // Basic Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%x", 0x123a);
  ASSERT_STREQ_LEN(written, buff, L"123a");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%X", 0x456b);
  ASSERT_STREQ_LEN(written, buff, L"456B");

  // Length Modifier Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%hhx", 0x10001);
  ASSERT_STREQ_LEN(written, buff, L"1");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%llx", 0xffffffffffffffffull);
  ASSERT_STREQ_LEN(written, buff, L"ffffffffffffffff"); // ull max

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%tX", ~ptrdiff_t(0));
  if (sizeof(ptrdiff_t) == 8) {
    ASSERT_STREQ_LEN(written, buff, L"FFFFFFFFFFFFFFFF");
  } else if (sizeof(ptrdiff_t) == 4) {
    ASSERT_STREQ_LEN(written, buff, L"FFFFFFFF");
  }

  // Min Width Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%4x", 0x789);
  ASSERT_STREQ_LEN(written, buff, L" 789");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%2X", 0x987);
  ASSERT_STREQ_LEN(written, buff, L"987");

  // Precision Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%x", 0);
  ASSERT_STREQ_LEN(written, buff, L"0");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.0x", 0);
  ASSERT_STREQ_LEN(written, buff, L"");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.5x", 0x1F3);
  ASSERT_STREQ_LEN(written, buff, L"001f3");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2x", 0x135);
  ASSERT_STREQ_LEN(written, buff, L"135");

  // Flag Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%-5x", 0x246);
  ASSERT_STREQ_LEN(written, buff, L"246  ");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%#x", 0xd3f);
  ASSERT_STREQ_LEN(written, buff, L"0xd3f");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%#x", 0);
  ASSERT_STREQ_LEN(written, buff, L"0");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%#X", 0xE40);
  ASSERT_STREQ_LEN(written, buff, L"0XE40");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%05x", 0x470);
  ASSERT_STREQ_LEN(written, buff, L"00470");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%0#6x", 0x8c3);
  ASSERT_STREQ_LEN(written, buff, L"0x08c3");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%-#6x", 0x5f0);
  ASSERT_STREQ_LEN(written, buff, L"0x5f0 ");

  // Combined Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%#-07x", 0x703);
  ASSERT_STREQ_LEN(written, buff, L"0x703  ");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%7.5x", 0x814);
  ASSERT_STREQ_LEN(written, buff, L"  00814");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%#9.5X", 0x9d4);
  ASSERT_STREQ_LEN(written, buff, L"  0X009D4");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%#.x", 0);
  ASSERT_STREQ_LEN(written, buff, L"");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%-7.5x", 0x260);
  ASSERT_STREQ_LEN(written, buff, L"00260  ");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%5.4x", 0x10000);
  ASSERT_STREQ_LEN(written, buff, L"10000");

  // Multiple Conversion Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%10X %-#10x", 0x45b, 0x789);
  ASSERT_STREQ_LEN(written, buff, L"       45B 0x789     ");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%-5.4x%#.4x", 0x75, 0x25);
  ASSERT_STREQ_LEN(written, buff, L"0075 0x0025");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%04hhX %#.5llx %-6.3zX",
                                     256 + 0x7f, 0x1000000000ll, size_t(2));
  ASSERT_STREQ_LEN(written, buff, L"007F 0x1000000000 002   ");
}

TEST_F(LlvmLibcSwprintfTest, BinConv) {
  constexpr size_t N = 64;
  wchar_t buff[N];
  int written;

  // Basic Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%b", 42);
  ASSERT_STREQ_LEN(written, buff, L"101010");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%B", 12081991);
  ASSERT_STREQ_LEN(written, buff, L"101110000101101101000111");

  // Min Width Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%10b", 0b101010);
  ASSERT_STREQ_LEN(written, buff, L"    101010");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%2B", 0b101010);
  ASSERT_STREQ_LEN(written, buff, L"101010");

  // Precision Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%b", 0);
  ASSERT_STREQ_LEN(written, buff, L"0");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.0b", 0);
  ASSERT_STREQ_LEN(written, buff, L"");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.5b", 0b111);
  ASSERT_STREQ_LEN(written, buff, L"00111");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2b", 0b111);
  ASSERT_STREQ_LEN(written, buff, L"111");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%3b", 0b111);
  ASSERT_STREQ_LEN(written, buff, L"111");

  // Flag Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%-5b", 0b111);
  ASSERT_STREQ_LEN(written, buff, L"111  ");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%#b", 0b111);
  ASSERT_STREQ_LEN(written, buff, L"0b111");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%#b", 0);
  ASSERT_STREQ_LEN(written, buff, L"0");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%#B", 0b111);
  ASSERT_STREQ_LEN(written, buff, L"0B111");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%05b", 0b111);
  ASSERT_STREQ_LEN(written, buff, L"00111");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%0#6b", 0b111);
  ASSERT_STREQ_LEN(written, buff, L"0b0111");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%-#6b", 0b111);
  ASSERT_STREQ_LEN(written, buff, L"0b111 ");

  // Combined Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%#-07b", 0b111);
  ASSERT_STREQ_LEN(written, buff, L"0b111  ");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%7.5b", 0b111);
  ASSERT_STREQ_LEN(written, buff, L"  00111");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%#9.5B", 0b111);
  ASSERT_STREQ_LEN(written, buff, L"  0B00111");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%#.b", 0);
  ASSERT_STREQ_LEN(written, buff, L"");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%-7.5b", 0b111);
  ASSERT_STREQ_LEN(written, buff, L"00111  ");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%5.4b", 0b1111);
  ASSERT_STREQ_LEN(written, buff, L" 1111");

  // Multiple Conversion Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%10B %-#10b", 0b101, 0b110);
  ASSERT_STREQ_LEN(written, buff, L"       101 0b110     ");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%-5.4b%#.4b", 0b101, 0b110);
  ASSERT_STREQ_LEN(written, buff, L"0101 0b0110");
}

TEST_F(LlvmLibcSwprintfTest, PointerConv) {
  constexpr size_t N = 64;
  wchar_t buff[N];
  int written;

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%p", nullptr);
  ASSERT_STREQ_LEN(written, buff, L"(nullptr)");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%p",
                                     static_cast<uintptr_t>(0x1a2b3c4d));
  ASSERT_STREQ_LEN(written, buff, L"0x1a2b3c4d");

  if constexpr (sizeof(void *) > 4) {
    written = LIBC_NAMESPACE::swprintf(
        buff, N, L"%p", static_cast<uintptr_t>(0x1a2b3c4d5e6f7081));
    ASSERT_STREQ_LEN(written, buff, L"0x1a2b3c4d5e6f7081");
  }

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%p", &written);
  EXPECT_GT(written, 0);

  // Width tests:

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%20p", nullptr);
  ASSERT_STREQ_LEN(written, buff, L"           (nullptr)");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%20p",
                                     static_cast<uintptr_t>(0x1a2b3c4d));
  ASSERT_STREQ_LEN(written, buff, L"          0x1a2b3c4d");

  // Flag tests:

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%-20p", nullptr);
  ASSERT_STREQ_LEN(written, buff, L"(nullptr)           ");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%-20p",
                                     static_cast<uintptr_t>(0x1a2b3c4d));
  ASSERT_STREQ_LEN(written, buff, L"0x1a2b3c4d          ");

  // Using the 0 flag is technically undefined, but here we're following the
  // convention of matching the behavior of %#x.
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%020p",
                                     static_cast<uintptr_t>(0x1a2b3c4d));
  ASSERT_STREQ_LEN(written, buff, L"0x00000000001a2b3c4d");

  // Precision tests:
  // These are all undefined behavior. The precision option is undefined for %p.

  // Precision specifies the number of characters for a string conversion.
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.5p", nullptr);
  ASSERT_STREQ_LEN(written, buff, L"(null");

  // Precision specifies the number of digits to be written for %x conversions,
  // and the "0x" doesn't count as part of the digits.
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.20p",
                                     static_cast<uintptr_t>(0x1a2b3c4d));
  ASSERT_STREQ_LEN(written, buff, L"0x0000000000001a2b3c4d");
}

TEST_F(LlvmLibcSwprintfTest, OctConv) {
  constexpr size_t N = 64;
  wchar_t buff[N];
  int written;

  // Basic Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%o", 01234);
  ASSERT_STREQ_LEN(written, buff, L"1234");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%o", 04567);
  ASSERT_STREQ_LEN(written, buff, L"4567");

  // Length Modifier Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%hho", 0401);
  ASSERT_STREQ_LEN(written, buff, L"1");

  written =
      LIBC_NAMESPACE::swprintf(buff, N, L"%llo", 01777777777777777777777ull);
  ASSERT_STREQ_LEN(written, buff, L"1777777777777777777777"); // ull max

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%to", ~ptrdiff_t(0));
  if (sizeof(ptrdiff_t) == 8) {
    ASSERT_STREQ_LEN(written, buff, L"1777777777777777777777");
  } else if (sizeof(ptrdiff_t) == 4) {
    ASSERT_STREQ_LEN(written, buff, L"37777777777");
  }

  // Min Width Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%4o", 0701);
  ASSERT_STREQ_LEN(written, buff, L" 701");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%2o", 0107);
  ASSERT_STREQ_LEN(written, buff, L"107");

  // Precision Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%o", 0);
  ASSERT_STREQ_LEN(written, buff, L"0");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.0o", 0);
  ASSERT_STREQ_LEN(written, buff, L"");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.5o", 0153);
  ASSERT_STREQ_LEN(written, buff, L"00153");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2o", 0135);
  ASSERT_STREQ_LEN(written, buff, L"135");

  // Flag Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%-5o", 0246);
  ASSERT_STREQ_LEN(written, buff, L"246  ");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%#o", 0234);
  ASSERT_STREQ_LEN(written, buff, L"0234");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%#o", 0);
  ASSERT_STREQ_LEN(written, buff, L"0");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%05o", 0470);
  ASSERT_STREQ_LEN(written, buff, L"00470");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%0#6o", 0753);
  ASSERT_STREQ_LEN(written, buff, L"000753");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%-#6o", 0642);
  ASSERT_STREQ_LEN(written, buff, L"0642  ");

  // Combined Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%#-07o", 0703);
  ASSERT_STREQ_LEN(written, buff, L"0703   ");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%#.o", 0);
  ASSERT_STREQ_LEN(written, buff, L"0");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%7.5o", 0314);
  ASSERT_STREQ_LEN(written, buff, L"  00314");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%#9.5o", 0234);
  ASSERT_STREQ_LEN(written, buff, L"    00234");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%-7.5o", 0260);
  ASSERT_STREQ_LEN(written, buff, L"00260  ");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%5.4o", 010000);
  ASSERT_STREQ_LEN(written, buff, L"10000");

  // Multiple Conversion Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%10o %-#10o", 0456, 0123);
  ASSERT_STREQ_LEN(written, buff, L"       456 0123      ");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%-5.4o%#.4o", 075, 025);
  ASSERT_STREQ_LEN(written, buff, L"0075 0025");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%04hho %#.5llo %-6.3zo",
                                     256 + 077, 01000000000000ll, size_t(2));
  ASSERT_STREQ_LEN(written, buff, L"0077 01000000000000 002   ");
}

#ifndef LIBC_COPT_PRINTF_DISABLE_FLOAT

TEST_F(LlvmLibcSwprintfTest, FloatHexExpConv) {
  constexpr size_t N = 128;
  wchar_t buff[N];
  int written;

  ForceRoundingMode r(RoundingMode::Nearest);
  double inf = LIBC_NAMESPACE::fputil::FPBits<double>::inf().get_val();
  double nan = LIBC_NAMESPACE::fputil::FPBits<double>::quiet_nan().get_val();
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%a", 1.0);
  ASSERT_STREQ_LEN(written, buff, L"0x1p+0");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%A", -1.0);
  ASSERT_STREQ_LEN(written, buff, L"-0X1P+0");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%a", -0x1.abcdef12345p0);
  ASSERT_STREQ_LEN(written, buff, L"-0x1.abcdef12345p+0");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%A", 0x1.abcdef12345p0);
  ASSERT_STREQ_LEN(written, buff, L"0X1.ABCDEF12345P+0");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%a", 0.0);
  ASSERT_STREQ_LEN(written, buff, L"0x0p+0");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%a", 1.0e100);
  ASSERT_STREQ_LEN(written, buff, L"0x1.249ad2594c37dp+332");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%a", 0.1);
  ASSERT_STREQ_LEN(written, buff, L"0x1.999999999999ap-4");

  // Subnormal Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%a", 0x1.0p-1027);
  ASSERT_STREQ_LEN(written, buff, L"0x0.08p-1022");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%a", 0x1.0p-1025);
  ASSERT_STREQ_LEN(written, buff, L"0x0.2p-1022");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%a", 0x1.0p-1023);
  ASSERT_STREQ_LEN(written, buff, L"0x0.8p-1022");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%a", 0x1.0p-1022);
  ASSERT_STREQ_LEN(written, buff, L"0x1p-1022");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%a", 0x1.0p-1074);
  ASSERT_STREQ_LEN(written, buff, L"0x0.0000000000001p-1022");

  // Inf/Nan Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%a", inf);
  ASSERT_STREQ_LEN(written, buff, L"inf");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%A", -inf);
  ASSERT_STREQ_LEN(written, buff, L"-INF");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%a", nan);
  ASSERT_STREQ_LEN(written, buff, L"nan");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%A", -nan);
  ASSERT_STREQ_LEN(written, buff, L"-NAN");

  // Length Modifier Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%La", 0.1L);
#if defined(LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80)
  ASSERT_STREQ_LEN(written, buff, L"0xc.ccccccccccccccdp-7");
#elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT64)
  ASSERT_STREQ_LEN(written, buff, L"0x1.999999999999ap-4");
#elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT128)
  ASSERT_STREQ_LEN(written, buff, L"0x1.999999999999999999999999999ap-4");
#endif
#if defined(LIBC_INTERNAL_PRINTF_CONVERT_FLOAT128)
  written =
      LIBC_NAMESPACE::swprintf(buff, N, L"%Qa", float128(1) / float128(10));
  ASSERT_STREQ_LEN(written, buff, L"0x1.999999999999999999999999999ap-4");
#endif // LIBC_INTERNAL_PRINTF_CONVERT_FLOAT128

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%La", 1.0e1000L);
#if defined(LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80)
  ASSERT_STREQ_LEN(written, buff, L"0xf.38db1f9dd3dac05p+3318");
#elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT64)
  ASSERT_STREQ_LEN(written, buff, L"inf");
#elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT128)
  ASSERT_STREQ_LEN(written, buff, L"0x1.e71b63f3ba7b580af1a52d2a7379p+3321");
#endif
#if defined(LIBC_INTERNAL_PRINTF_CONVERT_FLOAT128) &&                          \
    defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT128)
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%Qa", 1.0e1000L);
  ASSERT_STREQ_LEN(written, buff, L"0x1.e71b63f3ba7b580af1a52d2a7379p+3321");
#endif // LIBC_INTERNAL_PRINTF_CONVERT_FLOAT128 &&
       // LIBC_TYPES_LONG_DOUBLE_IS_FLOAT128

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%La", 1.0e-1000L);
#if defined(LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80)
  ASSERT_STREQ_LEN(written, buff, L"0x8.68a9188a89e1467p-3325");
#elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT64)
  ASSERT_STREQ_LEN(written, buff, L"0x0p+0");
#elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT128)
  ASSERT_STREQ_LEN(written, buff, L"0x1.0d152311513c28ce202627c06ec2p-3322");
#endif
#if defined(LIBC_INTERNAL_PRINTF_CONVERT_FLOAT128) &&                          \
    defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT128)
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%Qa", 1.0e-1000L);
  ASSERT_STREQ_LEN(written, buff, L"0x1.0d152311513c28ce202627c06ec2p-3322");
#endif // LIBC_INTERNAL_PRINTF_CONVERT_FLOAT128 &&
       // LIBC_TYPES_LONG_DOUBLE_IS_FLOAT128

  // Min Width Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%15a", 1.0);
  ASSERT_STREQ_LEN(written, buff, L"         0x1p+0");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%15a", -1.0);
  ASSERT_STREQ_LEN(written, buff, L"        -0x1p+0");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%15a", 1.0e10);
  ASSERT_STREQ_LEN(written, buff, L" 0x1.2a05f2p+33");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%15a", -1.0e10);
  ASSERT_STREQ_LEN(written, buff, L"-0x1.2a05f2p+33");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%10a", 1.0e10);
  ASSERT_STREQ_LEN(written, buff, L"0x1.2a05f2p+33");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%5a", inf);
  ASSERT_STREQ_LEN(written, buff, L"  inf");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%5a", -nan);
  ASSERT_STREQ_LEN(written, buff, L" -nan");

  // Precision Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1a", 1.0);
  ASSERT_STREQ_LEN(written, buff, L"0x1.0p+0");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1a", 0.0);
  ASSERT_STREQ_LEN(written, buff, L"0x0.0p+0");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1a", 0.1);
  ASSERT_STREQ_LEN(written, buff, L"0x1.ap-4");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1a", 0x1.0fp0);
  ASSERT_STREQ_LEN(written, buff, L"0x1.1p+0");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1a", 0x1.07p0);
  ASSERT_STREQ_LEN(written, buff, L"0x1.0p+0");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1a", 0x1.08p0);
  ASSERT_STREQ_LEN(written, buff, L"0x1.0p+0");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1a", 0x1.18p0);
  ASSERT_STREQ_LEN(written, buff, L"0x1.2p+0");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1a", 0x1.ffp0);
  ASSERT_STREQ_LEN(written, buff, L"0x2.0p+0");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.5a", 1.25);
  ASSERT_STREQ_LEN(written, buff, L"0x1.40000p+0");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.0a", 1.25);
  ASSERT_STREQ_LEN(written, buff, L"0x1p+0");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.0a", 1.75);
  ASSERT_STREQ_LEN(written, buff, L"0x2p+0");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1a", 0x1.0p-1023);
  ASSERT_STREQ_LEN(written, buff, L"0x0.8p-1022");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1a", 0x1.8p-1023);
  ASSERT_STREQ_LEN(written, buff, L"0x0.cp-1022");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1a", 0x1.0p-1024);
  ASSERT_STREQ_LEN(written, buff, L"0x0.4p-1022");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.0a", 0x1.0p-1023);
  ASSERT_STREQ_LEN(written, buff, L"0x0p-1022");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.0a", 0x1.8p-1023);
  ASSERT_STREQ_LEN(written, buff, L"0x1p-1022");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.0a", 0x1.0p-1024);
  ASSERT_STREQ_LEN(written, buff, L"0x0p-1022");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2a", 0x1.0p-1027);
  ASSERT_STREQ_LEN(written, buff, L"0x0.08p-1022");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1a", 0x1.0p-1027);
  ASSERT_STREQ_LEN(written, buff, L"0x0.0p-1022");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.5a", 0.0);
  ASSERT_STREQ_LEN(written, buff, L"0x0.00000p+0");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.5a", 0x1.008p0);
  ASSERT_STREQ_LEN(written, buff, L"0x1.00800p+0");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.5a", 0x1.008p10);
  ASSERT_STREQ_LEN(written, buff, L"0x1.00800p+10");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.5a", nan);
  ASSERT_STREQ_LEN(written, buff, L"nan");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%La", 0.0L);
  ASSERT_STREQ_LEN(written, buff, L"0x0p+0");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1La", 0.1L);
#if defined(LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80)
  ASSERT_STREQ_LEN(written, buff, L"0xc.dp-7");
#elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT64)
  ASSERT_STREQ_LEN(written, buff, L"0x1.ap-4");
#elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT128)
  ASSERT_STREQ_LEN(written, buff, L"0x1.ap-4");
#endif
#if defined(LIBC_INTERNAL_PRINTF_CONVERT_FLOAT128)
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1Qa", float128(0.1));
  ASSERT_STREQ_LEN(written, buff, L"0x1.ap-4");
#endif // LIBC_INTERNAL_PRINTF_CONVERT_FLOAT128

  written =
      LIBC_NAMESPACE::swprintf(buff, N, L"%.1La", 0xf.fffffffffffffffp16380L);
#if defined(LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80)
  ASSERT_STREQ_LEN(written, buff, L"0x1.0p+16384");
#elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT64)
  ASSERT_STREQ_LEN(written, buff, L"inf");
#elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT128)
  ASSERT_STREQ_LEN(written, buff, L"0x2.0p+16383");
#endif
#if defined(LIBC_INTERNAL_PRINTF_CONVERT_FLOAT128)
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1Qa",
                                     float128(0xf.fffffffffffffffp16380L));
  ASSERT_STREQ_LEN(written, buff, L"0x2.0p+16383");
#endif // LIBC_INTERNAL_PRINTF_CONVERT_FLOAT128

  // Rounding Mode Tests.

  if (ForceRoundingMode r(RoundingMode::Nearest); r.success) {
    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1a", 0x1.08p0);
    ASSERT_STREQ_LEN(written, buff, L"0x1.0p+0");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1a", 0x1.18p0);
    ASSERT_STREQ_LEN(written, buff, L"0x1.2p+0");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1a", 0x1.04p0);
    ASSERT_STREQ_LEN(written, buff, L"0x1.0p+0");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1a", 0x1.14p0);
    ASSERT_STREQ_LEN(written, buff, L"0x1.1p+0");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1a", -0x1.08p0);
    ASSERT_STREQ_LEN(written, buff, L"-0x1.0p+0");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1a", -0x1.18p0);
    ASSERT_STREQ_LEN(written, buff, L"-0x1.2p+0");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1a", -0x1.04p0);
    ASSERT_STREQ_LEN(written, buff, L"-0x1.0p+0");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1a", -0x1.14p0);
    ASSERT_STREQ_LEN(written, buff, L"-0x1.1p+0");
  }

  if (ForceRoundingMode r(RoundingMode::Upward); r.success) {
    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1a", 0x1.08p0);
    ASSERT_STREQ_LEN(written, buff, L"0x1.1p+0");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1a", 0x1.18p0);
    ASSERT_STREQ_LEN(written, buff, L"0x1.2p+0");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1a", 0x1.04p0);
    ASSERT_STREQ_LEN(written, buff, L"0x1.1p+0");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1a", 0x1.14p0);
    ASSERT_STREQ_LEN(written, buff, L"0x1.2p+0");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1a", -0x1.08p0);
    ASSERT_STREQ_LEN(written, buff, L"-0x1.0p+0");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1a", -0x1.18p0);
    ASSERT_STREQ_LEN(written, buff, L"-0x1.1p+0");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1a", -0x1.04p0);
    ASSERT_STREQ_LEN(written, buff, L"-0x1.0p+0");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1a", -0x1.14p0);
    ASSERT_STREQ_LEN(written, buff, L"-0x1.1p+0");
  }

  if (ForceRoundingMode r(RoundingMode::Downward); r.success) {
    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1a", 0x1.08p0);
    ASSERT_STREQ_LEN(written, buff, L"0x1.0p+0");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1a", 0x1.18p0);
    ASSERT_STREQ_LEN(written, buff, L"0x1.1p+0");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1a", 0x1.04p0);
    ASSERT_STREQ_LEN(written, buff, L"0x1.0p+0");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1a", 0x1.14p0);
    ASSERT_STREQ_LEN(written, buff, L"0x1.1p+0");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1a", -0x1.08p0);
    ASSERT_STREQ_LEN(written, buff, L"-0x1.1p+0");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1a", -0x1.18p0);
    ASSERT_STREQ_LEN(written, buff, L"-0x1.2p+0");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1a", -0x1.04p0);
    ASSERT_STREQ_LEN(written, buff, L"-0x1.1p+0");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1a", -0x1.14p0);
    ASSERT_STREQ_LEN(written, buff, L"-0x1.2p+0");
  }

  if (ForceRoundingMode r(RoundingMode::TowardZero); r.success) {
    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1a", 0x1.08p0);
    ASSERT_STREQ_LEN(written, buff, L"0x1.0p+0");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1a", 0x1.18p0);
    ASSERT_STREQ_LEN(written, buff, L"0x1.1p+0");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1a", 0x1.04p0);
    ASSERT_STREQ_LEN(written, buff, L"0x1.0p+0");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1a", 0x1.14p0);
    ASSERT_STREQ_LEN(written, buff, L"0x1.1p+0");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1a", -0x1.08p0);
    ASSERT_STREQ_LEN(written, buff, L"-0x1.0p+0");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1a", -0x1.18p0);
    ASSERT_STREQ_LEN(written, buff, L"-0x1.1p+0");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1a", -0x1.04p0);
    ASSERT_STREQ_LEN(written, buff, L"-0x1.0p+0");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1a", -0x1.14p0);
    ASSERT_STREQ_LEN(written, buff, L"-0x1.1p+0");
  }

  // Flag Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%+a", nan);
  ASSERT_STREQ_LEN(written, buff, L"+nan");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"% A", inf);
  ASSERT_STREQ_LEN(written, buff, L" INF");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%-5a", inf);
  ASSERT_STREQ_LEN(written, buff, L"inf  ");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%+-5A", nan);
  ASSERT_STREQ_LEN(written, buff, L"+NAN ");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%+a", 1.0);
  ASSERT_STREQ_LEN(written, buff, L"+0x1p+0");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"% a", 0.0);
  ASSERT_STREQ_LEN(written, buff, L" 0x0p+0");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%-10a", 1.5);
  ASSERT_STREQ_LEN(written, buff, L"0x1.8p+0  ");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%#a", 1.0);
  ASSERT_STREQ_LEN(written, buff, L"0x1.p+0");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%#.0a", 1.5);
  ASSERT_STREQ_LEN(written, buff, L"0x2.p+0");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%010a", 1.5);
  ASSERT_STREQ_LEN(written, buff, L"0x001.8p+0");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%+- #0a", 0.0);
  ASSERT_STREQ_LEN(written, buff, L"+0x0.p+0");

  // Combined Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%12.3a %-12.3A", 0.1, 256.0);
  ASSERT_STREQ_LEN(written, buff, L"  0x1.99ap-4 0X1.000P+8  ");

  written =
      LIBC_NAMESPACE::swprintf(buff, N, L"%+-#12.3a % 012.3a", 0.1256, 1256.0);
  ASSERT_STREQ_LEN(written, buff, L"+0x1.014p-3   0x1.3a0p+10");

  // These tests check that the padding is properly calculated based on the
  // min_width field. Specifically, they check that the extra zeroes added by
  // the high precision are accounted for correctly.
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%50.50a", 0x1.0p0);
  ASSERT_STREQ_LEN(
      written, buff,
      L"0x1.00000000000000000000000000000000000000000000000000p+0");

  // The difference with this test is that the formatted number is exactly 57
  // characters, so padding to 58 adds a space.
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%58.50a", 0x1.0p0);
  ASSERT_STREQ_LEN(
      written, buff,
      L" 0x1.00000000000000000000000000000000000000000000000000p+0");
}

TEST_F(LlvmLibcSwprintfTest, FloatDecimalConv) {
  constexpr size_t N = 1500;
  wchar_t buff[N];
  int written;

  ForceRoundingMode r(RoundingMode::Nearest);
  double inf = LIBC_NAMESPACE::fputil::FPBits<double>::inf().get_val();
  double nan = LIBC_NAMESPACE::fputil::FPBits<double>::quiet_nan().get_val();
  long double ld_inf =
      LIBC_NAMESPACE::fputil::FPBits<long double>::inf().get_val();
  long double ld_nan =
      LIBC_NAMESPACE::fputil::FPBits<long double>::quiet_nan().get_val();
#if defined(LIBC_INTERNAL_PRINTF_CONVERT_FLOAT128)
  float128 f128_inf = LIBC_NAMESPACE::fputil::FPBits<float128>::inf().get_val();
  float128 f128_nan =
      LIBC_NAMESPACE::fputil::FPBits<float128>::quiet_nan().get_val();
#endif // LIBC_INTERNAL_PRINTF_CONVERT_FLOAT128

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%f", 1.0);
  ASSERT_STREQ_LEN(written, buff, L"1.000000");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%F", -1.0);
  ASSERT_STREQ_LEN(written, buff, L"-1.000000");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%f", -1.234567);
  ASSERT_STREQ_LEN(written, buff, L"-1.234567");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%f", 0.0);
  ASSERT_STREQ_LEN(written, buff, L"0.000000");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%f", 1.5);
  ASSERT_STREQ_LEN(written, buff, L"1.500000");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%f", 0.1);
  ASSERT_STREQ_LEN(written, buff, L"0.100000");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%f", 1234567890123456789.0);
  ASSERT_STREQ_LEN(written, buff, L"1234567890123456768.000000");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%f", 9999999999999.99);
  ASSERT_STREQ_LEN(written, buff, L"9999999999999.990234");

  // Simple Subnormal Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%f", 0x1.0p-1027);
  ASSERT_STREQ_LEN(written, buff, L"0.000000");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%f", 0x1.0p-1074);
  ASSERT_STREQ_LEN(written, buff, L"0.000000");

  // Inf/Nan Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%f", inf);
  ASSERT_STREQ_LEN(written, buff, L"inf");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%F", -inf);
  ASSERT_STREQ_LEN(written, buff, L"-INF");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%f", nan);
  ASSERT_STREQ_LEN(written, buff, L"nan");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%F", -nan);
  ASSERT_STREQ_LEN(written, buff, L"-NAN");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%Lf", ld_inf);
  ASSERT_STREQ_LEN(written, buff, L"inf");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%LF", -ld_inf);
  ASSERT_STREQ_LEN(written, buff, L"-INF");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%Lf", ld_nan);
  ASSERT_STREQ_LEN(written, buff, L"nan");

// Some float128 systems (specifically the ones used for aarch64 buildbots)
// don't respect signs for long double NaNs.
#if defined(LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80) ||                          \
    defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT64)
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%LF", -ld_nan);
  ASSERT_STREQ_LEN(written, buff, L"-NAN");
#endif

#if defined(LIBC_INTERNAL_PRINTF_CONVERT_FLOAT128)
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%Qf", f128_inf);
  ASSERT_STREQ_LEN(written, buff, L"inf");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%Qf", -f128_inf);
  ASSERT_STREQ_LEN(written, buff, L"-inf");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%QF", f128_inf);
  ASSERT_STREQ_LEN(written, buff, L"INF");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%QF", -f128_inf);
  ASSERT_STREQ_LEN(written, buff, L"-INF");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%Qf", f128_nan);
  ASSERT_STREQ_LEN(written, buff, L"nan");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%QF", f128_nan);
  ASSERT_STREQ_LEN(written, buff, L"NAN");
#endif // LIBC_INTERNAL_PRINTF_CONVERT_FLOAT128

  // Min Width Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%15f", 1.0);
  ASSERT_STREQ_LEN(written, buff, L"       1.000000");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%15f", -1.0);
  ASSERT_STREQ_LEN(written, buff, L"      -1.000000");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%15f", 1.0e5);
  ASSERT_STREQ_LEN(written, buff, L"  100000.000000");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%15f", -1.0e5);
  ASSERT_STREQ_LEN(written, buff, L" -100000.000000");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%10f", 1.0e5);
  ASSERT_STREQ_LEN(written, buff, L"100000.000000");

  // Precision Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", 1.0);
  ASSERT_STREQ_LEN(written, buff, L"1.0");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", 0.0);
  ASSERT_STREQ_LEN(written, buff, L"0.0");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.0f", 0.0);
  ASSERT_STREQ_LEN(written, buff, L"0");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", 0.1);
  ASSERT_STREQ_LEN(written, buff, L"0.1");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", 1.09);
  ASSERT_STREQ_LEN(written, buff, L"1.1");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", 1.04);
  ASSERT_STREQ_LEN(written, buff, L"1.0");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", 1.19);
  ASSERT_STREQ_LEN(written, buff, L"1.2");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", 1.99);
  ASSERT_STREQ_LEN(written, buff, L"2.0");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", 9.99);
  ASSERT_STREQ_LEN(written, buff, L"10.0");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2f", 9999999999999.99);
  ASSERT_STREQ_LEN(written, buff, L"9999999999999.99");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", 9999999999999.99);
  ASSERT_STREQ_LEN(written, buff, L"10000000000000.0");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.5f", 1.25);
  ASSERT_STREQ_LEN(written, buff, L"1.25000");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.0f", 1.25);
  ASSERT_STREQ_LEN(written, buff, L"1");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.0f", 1.75);
  ASSERT_STREQ_LEN(written, buff, L"2");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.20f", 1.234e-10);
  ASSERT_STREQ_LEN(written, buff, L"0.00000000012340000000");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2f", -9.99);
  ASSERT_STREQ_LEN(written, buff, L"-9.99");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", -9.99);
  ASSERT_STREQ_LEN(written, buff, L"-10.0");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.5f", 0.0);
  ASSERT_STREQ_LEN(written, buff, L"0.00000");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.5f", 1.008);
  ASSERT_STREQ_LEN(written, buff, L"1.00800");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.5f", 1.008e3);
  ASSERT_STREQ_LEN(written, buff, L"1008.00000");

  // Found with the help of Fred Tydeman's tbin2dec test.
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", 0x1.1000000000006p+3);
  ASSERT_STREQ_LEN(written, buff, L"8.5");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.0f", 0x1.1000000000006p+3);
  ASSERT_STREQ_LEN(written, buff, L"9");

  // If no precision is specified it defaults to 6 for %f.
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%f", 2325885.4901960781);
  ASSERT_STREQ_LEN(written, buff, L"2325885.490196");

  // Rounding Mode Tests.

  if (ForceRoundingMode r(RoundingMode::Nearest); r.success) {
    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", 1.75);
    ASSERT_STREQ_LEN(written, buff, L"1.8");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", 1.25);
    ASSERT_STREQ_LEN(written, buff, L"1.2");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", 1.125);
    ASSERT_STREQ_LEN(written, buff, L"1.1");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", 1.625);
    ASSERT_STREQ_LEN(written, buff, L"1.6");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", 1.375);
    ASSERT_STREQ_LEN(written, buff, L"1.4");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", 1.875);
    ASSERT_STREQ_LEN(written, buff, L"1.9");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", -1.75);
    ASSERT_STREQ_LEN(written, buff, L"-1.8");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", -1.25);
    ASSERT_STREQ_LEN(written, buff, L"-1.2");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", -1.125);
    ASSERT_STREQ_LEN(written, buff, L"-1.1");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", -1.625);
    ASSERT_STREQ_LEN(written, buff, L"-1.6");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", -1.375);
    ASSERT_STREQ_LEN(written, buff, L"-1.4");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", -1.875);
    ASSERT_STREQ_LEN(written, buff, L"-1.9");
  }

  if (ForceRoundingMode r(RoundingMode::Upward); r.success) {
    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", 1.75);
    ASSERT_STREQ_LEN(written, buff, L"1.8");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", 1.25);
    ASSERT_STREQ_LEN(written, buff, L"1.3");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", 1.125);
    ASSERT_STREQ_LEN(written, buff, L"1.2");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", 1.625);
    ASSERT_STREQ_LEN(written, buff, L"1.7");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", 1.375);
    ASSERT_STREQ_LEN(written, buff, L"1.4");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", 1.875);
    ASSERT_STREQ_LEN(written, buff, L"1.9");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", -1.75);
    ASSERT_STREQ_LEN(written, buff, L"-1.7");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", -1.25);
    ASSERT_STREQ_LEN(written, buff, L"-1.2");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", -1.125);
    ASSERT_STREQ_LEN(written, buff, L"-1.1");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", -1.625);
    ASSERT_STREQ_LEN(written, buff, L"-1.6");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", -1.375);
    ASSERT_STREQ_LEN(written, buff, L"-1.3");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", -1.875);
    ASSERT_STREQ_LEN(written, buff, L"-1.8");
  }

  if (ForceRoundingMode r(RoundingMode::Downward); r.success) {
    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", 1.75);
    ASSERT_STREQ_LEN(written, buff, L"1.7");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", 1.25);
    ASSERT_STREQ_LEN(written, buff, L"1.2");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", 1.125);
    ASSERT_STREQ_LEN(written, buff, L"1.1");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", 1.625);
    ASSERT_STREQ_LEN(written, buff, L"1.6");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", 1.375);
    ASSERT_STREQ_LEN(written, buff, L"1.3");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", 1.875);
    ASSERT_STREQ_LEN(written, buff, L"1.8");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", -1.75);
    ASSERT_STREQ_LEN(written, buff, L"-1.8");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", -1.25);
    ASSERT_STREQ_LEN(written, buff, L"-1.3");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", -1.125);
    ASSERT_STREQ_LEN(written, buff, L"-1.2");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", -1.625);
    ASSERT_STREQ_LEN(written, buff, L"-1.7");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", -1.375);
    ASSERT_STREQ_LEN(written, buff, L"-1.4");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", -1.875);
    ASSERT_STREQ_LEN(written, buff, L"-1.9");
  }

  if (ForceRoundingMode r(RoundingMode::TowardZero); r.success) {
    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", 1.75);
    ASSERT_STREQ_LEN(written, buff, L"1.7");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", 1.25);
    ASSERT_STREQ_LEN(written, buff, L"1.2");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", 1.125);
    ASSERT_STREQ_LEN(written, buff, L"1.1");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", 1.625);
    ASSERT_STREQ_LEN(written, buff, L"1.6");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", 1.375);
    ASSERT_STREQ_LEN(written, buff, L"1.3");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", 1.875);
    ASSERT_STREQ_LEN(written, buff, L"1.8");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", -1.75);
    ASSERT_STREQ_LEN(written, buff, L"-1.7");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", -1.25);
    ASSERT_STREQ_LEN(written, buff, L"-1.2");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", -1.125);
    ASSERT_STREQ_LEN(written, buff, L"-1.1");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", -1.625);
    ASSERT_STREQ_LEN(written, buff, L"-1.6");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", -1.375);
    ASSERT_STREQ_LEN(written, buff, L"-1.3");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1f", -1.875);
    ASSERT_STREQ_LEN(written, buff, L"-1.8");
  }

  // Flag Tests.
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%+f", 1.0);
  ASSERT_STREQ_LEN(written, buff, L"+1.000000");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%+f", -1.0);
  ASSERT_STREQ_LEN(written, buff, L"-1.000000");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"% f", 1.0);
  ASSERT_STREQ_LEN(written, buff, L" 1.000000");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"% f", -1.0);
  ASSERT_STREQ_LEN(written, buff, L"-1.000000");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%-10f", 1.5);
  ASSERT_STREQ_LEN(written, buff, L"1.500000  ");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%#.f", 1.0);
  ASSERT_STREQ_LEN(written, buff, L"1.");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%#.0f", 1.5);
  ASSERT_STREQ_LEN(written, buff, L"2.");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%010f", 1.5);
  ASSERT_STREQ_LEN(written, buff, L"001.500000");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%010f", -1.5);
  ASSERT_STREQ_LEN(written, buff, L"-01.500000");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%+- #0f", 0.0);
  ASSERT_STREQ_LEN(written, buff, L"+0.000000");

  // Combined Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%10.2f", 9.99);
  ASSERT_STREQ_LEN(written, buff, L"      9.99");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%5.1f", 9.99);
  ASSERT_STREQ_LEN(written, buff, L" 10.0");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%-10.2f", 9.99);
  ASSERT_STREQ_LEN(written, buff, L"9.99      ");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%-5.1f", 9.99);
  ASSERT_STREQ_LEN(written, buff, L"10.0 ");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%-5.1f", 1.0e-50);
  ASSERT_STREQ_LEN(written, buff, L"0.0  ");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%30f", 1234567890123456789.0);
  ASSERT_STREQ_LEN(written, buff, L"    1234567890123456768.000000");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%-30f", 1234567890123456789.0);
  ASSERT_STREQ_LEN(written, buff, L"1234567890123456768.000000    ");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%20.2f", 9999999999999.99);
  ASSERT_STREQ_LEN(written, buff, L"    9999999999999.99");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%20.1f", 9999999999999.99);
  ASSERT_STREQ_LEN(written, buff, L"    10000000000000.0");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%12.3f %-12.3f", 0.1, 256.0);
  ASSERT_STREQ_LEN(written, buff, L"       0.100 256.000     ");

  written =
      LIBC_NAMESPACE::swprintf(buff, N, L"%+-#12.3f % 012.3f", 0.1256, 1256.0);
  ASSERT_STREQ_LEN(written, buff, L"+0.126        0001256.000");
}

TEST_F(LlvmLibcSwprintfTest, FloatDecimalLongDoubleConv) {
  constexpr size_t N = 64;
  wchar_t buff[N];
  int written;

  ForceRoundingMode r(RoundingMode::Nearest);

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%Lf", 1.0L);
  ASSERT_STREQ_LEN(written, buff, L"1.000000");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.Lf", -2.5L);
  ASSERT_STREQ_LEN(written, buff, L"-2");
}

#if defined(LIBC_INTERNAL_PRINTF_CONVERT_FLOAT128)
TEST_F(LlvmLibcSwprintfTest, FloatDecimalFloat128Conv) {
  constexpr size_t N = 64;
  wchar_t buff[N];
  int written;

  ForceRoundingMode r(RoundingMode::Nearest);

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%Qf", float128(1.0L));
  ASSERT_STREQ_LEN(written, buff, L"1.000000");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.Qf", float128(-2.5L));
  ASSERT_STREQ_LEN(written, buff, L"-2");
}
#endif // LIBC_INTERNAL_PRINTF_CONVERT_FLOAT128

TEST_F(LlvmLibcSwprintfTest, FloatExponentConv) {
  constexpr size_t N = 1000;
  wchar_t buff[N];
  int written;

  ForceRoundingMode r(RoundingMode::Nearest);
  double inf = LIBC_NAMESPACE::fputil::FPBits<double>::inf().get_val();
  double nan = LIBC_NAMESPACE::fputil::FPBits<double>::quiet_nan().get_val();

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%e", 1.0);
  ASSERT_STREQ_LEN(written, buff, L"1.000000e+00");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%E", -1.0);
  ASSERT_STREQ_LEN(written, buff, L"-1.000000E+00");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%e", -1.234567);
  ASSERT_STREQ_LEN(written, buff, L"-1.234567e+00");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%e", 0.0);
  ASSERT_STREQ_LEN(written, buff, L"0.000000e+00");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%e", 1.5);
  ASSERT_STREQ_LEN(written, buff, L"1.500000e+00");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%e", 1e300);
  ASSERT_STREQ_LEN(written, buff, L"1.000000e+300");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%e", 0.1);
  ASSERT_STREQ_LEN(written, buff, L"1.000000e-01");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%e", 0.001);
  ASSERT_STREQ_LEN(written, buff, L"1.000000e-03");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%e", 0.00001);
  ASSERT_STREQ_LEN(written, buff, L"1.000000e-05");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%e", 0.0000001);
  ASSERT_STREQ_LEN(written, buff, L"1.000000e-07");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%e", 0.000000001);
  ASSERT_STREQ_LEN(written, buff, L"1.000000e-09");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%e", 1.0e-20);
  ASSERT_STREQ_LEN(written, buff, L"1.000000e-20");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%e", 1234567890123456789.0);
  ASSERT_STREQ_LEN(written, buff, L"1.234568e+18");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%e", 9999999000000.00);
  ASSERT_STREQ_LEN(written, buff, L"9.999999e+12");

  // Simple Subnormal Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%e", 0x1.0p-1027);
  ASSERT_STREQ_LEN(written, buff, L"6.953356e-310");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%e", 0x1.0p-1074);
  ASSERT_STREQ_LEN(written, buff, L"4.940656e-324");

  // Inf/Nan Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%e", inf);
  ASSERT_STREQ_LEN(written, buff, L"inf");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%E", -inf);
  ASSERT_STREQ_LEN(written, buff, L"-INF");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%e", nan);
  ASSERT_STREQ_LEN(written, buff, L"nan");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%E", -nan);
  ASSERT_STREQ_LEN(written, buff, L"-NAN");

  // Min Width Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%15e", 1.0);
  ASSERT_STREQ_LEN(written, buff, L"   1.000000e+00");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%15e", -1.0);
  ASSERT_STREQ_LEN(written, buff, L"  -1.000000e+00");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%15e", 1.0e5);
  ASSERT_STREQ_LEN(written, buff, L"   1.000000e+05");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%15e", -1.0e5);
  ASSERT_STREQ_LEN(written, buff, L"  -1.000000e+05");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%10e", 1.0e-5);
  ASSERT_STREQ_LEN(written, buff, L"1.000000e-05");

  // Precision Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", 1.0);
  ASSERT_STREQ_LEN(written, buff, L"1.0e+00");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", 0.0);
  ASSERT_STREQ_LEN(written, buff, L"0.0e+00");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.0e", 0.0);
  ASSERT_STREQ_LEN(written, buff, L"0e+00");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", 0.1);
  ASSERT_STREQ_LEN(written, buff, L"1.0e-01");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", 1.09);
  ASSERT_STREQ_LEN(written, buff, L"1.1e+00");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", 1.04);
  ASSERT_STREQ_LEN(written, buff, L"1.0e+00");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", 1.19);
  ASSERT_STREQ_LEN(written, buff, L"1.2e+00");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", 1.99);
  ASSERT_STREQ_LEN(written, buff, L"2.0e+00");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", 9.99);
  ASSERT_STREQ_LEN(written, buff, L"1.0e+01");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2e", 99.9);
  ASSERT_STREQ_LEN(written, buff, L"9.99e+01");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", 99.9);
  ASSERT_STREQ_LEN(written, buff, L"1.0e+02");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.5e", 1.25);
  ASSERT_STREQ_LEN(written, buff, L"1.25000e+00");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.0e", 1.25);
  ASSERT_STREQ_LEN(written, buff, L"1e+00");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.0e", 1.75);
  ASSERT_STREQ_LEN(written, buff, L"2e+00");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.20e", 1.234e-10);
  ASSERT_STREQ_LEN(written, buff, L"1.23400000000000008140e-10");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2e", -9.99);
  ASSERT_STREQ_LEN(written, buff, L"-9.99e+00");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", -9.99);
  ASSERT_STREQ_LEN(written, buff, L"-1.0e+01");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.5e", 0.0);
  ASSERT_STREQ_LEN(written, buff, L"0.00000e+00");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.5e", 1.008);
  ASSERT_STREQ_LEN(written, buff, L"1.00800e+00");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.5e", 1.008e3);
  ASSERT_STREQ_LEN(written, buff, L"1.00800e+03");

  // Rounding Mode Tests.

  if (ForceRoundingMode r(RoundingMode::Nearest); r.success) {
    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", 1.75);
    ASSERT_STREQ_LEN(written, buff, L"1.8e+00");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", 1.25);
    ASSERT_STREQ_LEN(written, buff, L"1.2e+00");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", 1.125);
    ASSERT_STREQ_LEN(written, buff, L"1.1e+00");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", 1.625);
    ASSERT_STREQ_LEN(written, buff, L"1.6e+00");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", 1.375);
    ASSERT_STREQ_LEN(written, buff, L"1.4e+00");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", 1.875);
    ASSERT_STREQ_LEN(written, buff, L"1.9e+00");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", -1.75);
    ASSERT_STREQ_LEN(written, buff, L"-1.8e+00");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", -1.25);
    ASSERT_STREQ_LEN(written, buff, L"-1.2e+00");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", -1.125);
    ASSERT_STREQ_LEN(written, buff, L"-1.1e+00");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", -1.625);
    ASSERT_STREQ_LEN(written, buff, L"-1.6e+00");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", -1.375);
    ASSERT_STREQ_LEN(written, buff, L"-1.4e+00");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", -1.875);
    ASSERT_STREQ_LEN(written, buff, L"-1.9e+00");
  }

  if (ForceRoundingMode r(RoundingMode::Upward); r.success) {
    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", 1.75);
    ASSERT_STREQ_LEN(written, buff, L"1.8e+00");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", 1.25);
    ASSERT_STREQ_LEN(written, buff, L"1.3e+00");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", 1.125);
    ASSERT_STREQ_LEN(written, buff, L"1.2e+00");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", 1.625);
    ASSERT_STREQ_LEN(written, buff, L"1.7e+00");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", 1.375);
    ASSERT_STREQ_LEN(written, buff, L"1.4e+00");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", 1.875);
    ASSERT_STREQ_LEN(written, buff, L"1.9e+00");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", -1.75);
    ASSERT_STREQ_LEN(written, buff, L"-1.7e+00");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", -1.25);
    ASSERT_STREQ_LEN(written, buff, L"-1.2e+00");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", -1.125);
    ASSERT_STREQ_LEN(written, buff, L"-1.1e+00");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", -1.625);
    ASSERT_STREQ_LEN(written, buff, L"-1.6e+00");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", -1.375);
    ASSERT_STREQ_LEN(written, buff, L"-1.3e+00");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", -1.875);
    ASSERT_STREQ_LEN(written, buff, L"-1.8e+00");
  }

  if (ForceRoundingMode r(RoundingMode::Downward); r.success) {
    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", 1.75);
    ASSERT_STREQ_LEN(written, buff, L"1.7e+00");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", 1.25);
    ASSERT_STREQ_LEN(written, buff, L"1.2e+00");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", 1.125);
    ASSERT_STREQ_LEN(written, buff, L"1.1e+00");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", 1.625);
    ASSERT_STREQ_LEN(written, buff, L"1.6e+00");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", 1.375);
    ASSERT_STREQ_LEN(written, buff, L"1.3e+00");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", 1.875);
    ASSERT_STREQ_LEN(written, buff, L"1.8e+00");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", -1.75);
    ASSERT_STREQ_LEN(written, buff, L"-1.8e+00");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", -1.25);
    ASSERT_STREQ_LEN(written, buff, L"-1.3e+00");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", -1.125);
    ASSERT_STREQ_LEN(written, buff, L"-1.2e+00");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", -1.625);
    ASSERT_STREQ_LEN(written, buff, L"-1.7e+00");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", -1.375);
    ASSERT_STREQ_LEN(written, buff, L"-1.4e+00");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", -1.875);
    ASSERT_STREQ_LEN(written, buff, L"-1.9e+00");
  }

  if (ForceRoundingMode r(RoundingMode::TowardZero); r.success) {
    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", 1.75);
    ASSERT_STREQ_LEN(written, buff, L"1.7e+00");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", 1.25);
    ASSERT_STREQ_LEN(written, buff, L"1.2e+00");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", 1.125);
    ASSERT_STREQ_LEN(written, buff, L"1.1e+00");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", 1.625);
    ASSERT_STREQ_LEN(written, buff, L"1.6e+00");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", 1.375);
    ASSERT_STREQ_LEN(written, buff, L"1.3e+00");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", 1.875);
    ASSERT_STREQ_LEN(written, buff, L"1.8e+00");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", -1.75);
    ASSERT_STREQ_LEN(written, buff, L"-1.7e+00");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", -1.25);
    ASSERT_STREQ_LEN(written, buff, L"-1.2e+00");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", -1.125);
    ASSERT_STREQ_LEN(written, buff, L"-1.1e+00");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", -1.625);
    ASSERT_STREQ_LEN(written, buff, L"-1.6e+00");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", -1.375);
    ASSERT_STREQ_LEN(written, buff, L"-1.3e+00");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1e", -1.875);
    ASSERT_STREQ_LEN(written, buff, L"-1.8e+00");
  }

  // Flag Tests.
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%+e", 1.0);
  ASSERT_STREQ_LEN(written, buff, L"+1.000000e+00");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%+e", -1.0);
  ASSERT_STREQ_LEN(written, buff, L"-1.000000e+00");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"% e", 1.0);
  ASSERT_STREQ_LEN(written, buff, L" 1.000000e+00");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"% e", -1.0);
  ASSERT_STREQ_LEN(written, buff, L"-1.000000e+00");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%-15e", 1.5);
  ASSERT_STREQ_LEN(written, buff, L"1.500000e+00   ");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%#.e", 1.0);
  ASSERT_STREQ_LEN(written, buff, L"1.e+00");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%#.0e", 1.5);
  ASSERT_STREQ_LEN(written, buff, L"2.e+00");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%015e", 1.5);
  ASSERT_STREQ_LEN(written, buff, L"0001.500000e+00");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%015e", -1.5);
  ASSERT_STREQ_LEN(written, buff, L"-001.500000e+00");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%+- #0e", 0.0);
  ASSERT_STREQ_LEN(written, buff, L"+0.000000e+00");

  // Combined Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%10.2e", 9.99);
  ASSERT_STREQ_LEN(written, buff, L"  9.99e+00");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%10.1e", 9.99);
  ASSERT_STREQ_LEN(written, buff, L"   1.0e+01");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%10.0e", 9.99);
  ASSERT_STREQ_LEN(written, buff, L"     1e+01");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%10.0e", 0.0999);
  ASSERT_STREQ_LEN(written, buff, L"     1e-01");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%-10.2e", 9.99);
  ASSERT_STREQ_LEN(written, buff, L"9.99e+00  ");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%-10.1e", 9.99);
  ASSERT_STREQ_LEN(written, buff, L"1.0e+01   ");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%-10.1e", 1.0e-50);
  ASSERT_STREQ_LEN(written, buff, L"1.0e-50   ");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%30e", 1234567890123456789.0);
  ASSERT_STREQ_LEN(written, buff, L"                  1.234568e+18");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%-30e", 1234567890123456789.0);
  ASSERT_STREQ_LEN(written, buff, L"1.234568e+18                  ");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%25.14e", 9999999999999.99);
  ASSERT_STREQ_LEN(written, buff, L"     9.99999999999999e+12");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%25.13e", 9999999999999.99);
  ASSERT_STREQ_LEN(written, buff, L"      1.0000000000000e+13");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%25.12e", 9999999999999.99);
  ASSERT_STREQ_LEN(written, buff, L"       1.000000000000e+13");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%12.3e %-12.3e", 0.1, 256.0);
  ASSERT_STREQ_LEN(written, buff, L"   1.000e-01 2.560e+02   ");

  written =
      LIBC_NAMESPACE::swprintf(buff, N, L"%+-#12.3e % 012.3e", 0.1256, 1256.0);
  ASSERT_STREQ_LEN(written, buff, L"+1.256e-01    001.256e+03");
}

TEST_F(LlvmLibcSwprintfTest, FloatExponentLongDoubleConv) {
  constexpr size_t N = 1000;
  wchar_t buff[N];
  int written;

  ForceRoundingMode r(RoundingMode::Nearest);
  // Length Modifier Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.9Le", 1000000000500000000.0L);
  ASSERT_STREQ_LEN(written, buff, L"1.000000000e+18");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%Le", 1e100L);
  ASSERT_STREQ_LEN(written, buff, L"1.000000e+100");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%Le", 1.0L);
  ASSERT_STREQ_LEN(written, buff, L"1.000000e+00");

#if !defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT64)
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.9Le", 1000000000500000000.1L);
  ASSERT_STREQ_LEN(written, buff, L"1.000000001e+18");

  written =
      LIBC_NAMESPACE::swprintf(buff, N, L"%Le", 0xf.fffffffffffffffp+16380L);
  ASSERT_STREQ_LEN(written, buff, L"1.189731e+4932");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%Le", 1e1000L);
  ASSERT_STREQ_LEN(written, buff, L"1.000000e+1000");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%Le", 1e4900L);
  ASSERT_STREQ_LEN(written, buff, L"1.000000e+4900");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%Le", 1.2345678e4900L);
  ASSERT_STREQ_LEN(written, buff, L"1.234568e+4900");
#endif
}

#if defined(LIBC_INTERNAL_PRINTF_CONVERT_FLOAT128)
TEST_F(LlvmLibcSwprintfTest, FloatExponentFloat128Conv) {
  constexpr size_t N = 1000;
  wchar_t buff[N];
  int written;

  ForceRoundingMode r(RoundingMode::Nearest);

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.9Qe",
                                     float128(1000000000500000000.1L));
  ASSERT_STREQ_LEN(written, buff, L"1.000000001e+18");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.9Qe",
                                     float128(1000000000500000000.0L));
  ASSERT_STREQ_LEN(written, buff, L"1.000000000e+18");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%Qe", float128(1e100L));
  ASSERT_STREQ_LEN(written, buff, L"1.000000e+100");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%Qe", float128(1.0L));
  ASSERT_STREQ_LEN(written, buff, L"1.000000e+00");

#if !defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT64)
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%Qe",
                                     float128(0xf.fffffffffffffffp+16380L));
  ASSERT_STREQ_LEN(written, buff, L"1.189731e+4932");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%Qe", float128(1e1000L));
  ASSERT_STREQ_LEN(written, buff, L"1.000000e+1000");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%Qe", float128(1e4900L));
  ASSERT_STREQ_LEN(written, buff, L"1.000000e+4900");

  written =
      LIBC_NAMESPACE::swprintf(buff, N, L"%Qe", float128(1.2345678e4900L));
  ASSERT_STREQ_LEN(written, buff, L"1.234568e+4900");

#endif // !LIBC_TYPES_LONG_DOUBLE_IS_FLOAT64
}
#endif // LIBC_INTERNAL_PRINTF_CONVERT_FLOAT128

TEST_F(LlvmLibcSwprintfTest, FloatAutoConv) {
  constexpr size_t N = 1000;
  wchar_t buff[N];
  int written;

  ForceRoundingMode r(RoundingMode::Nearest);
  double inf = LIBC_NAMESPACE::fputil::FPBits<double>::inf().get_val();
  double nan = LIBC_NAMESPACE::fputil::FPBits<double>::quiet_nan().get_val();

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%g", 1.0);
  ASSERT_STREQ_LEN(written, buff, L"1");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%G", -1.0);
  ASSERT_STREQ_LEN(written, buff, L"-1");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%g", -1.234567);
  ASSERT_STREQ_LEN(written, buff, L"-1.23457");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%g", 0.0);
  ASSERT_STREQ_LEN(written, buff, L"0");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%g", -0.0);
  ASSERT_STREQ_LEN(written, buff, L"-0");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%g", 1.5);
  ASSERT_STREQ_LEN(written, buff, L"1.5");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%g", 1e300);
  ASSERT_STREQ_LEN(written, buff, L"1e+300");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%g", 0.1);
  ASSERT_STREQ_LEN(written, buff, L"0.1");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%g", 0.001);
  ASSERT_STREQ_LEN(written, buff, L"0.001");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%g", 0.00001);
  ASSERT_STREQ_LEN(written, buff, L"1e-05");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%g", 0.0000001);
  ASSERT_STREQ_LEN(written, buff, L"1e-07");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%g", 0.000000001);
  ASSERT_STREQ_LEN(written, buff, L"1e-09");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%g", 1.0e-20);
  ASSERT_STREQ_LEN(written, buff, L"1e-20");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%g", 1234567890123456789.0);
  ASSERT_STREQ_LEN(written, buff, L"1.23457e+18");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%g", 9999990000000.00);
  ASSERT_STREQ_LEN(written, buff, L"9.99999e+12");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%g", 9999999000000.00);
  ASSERT_STREQ_LEN(written, buff, L"1e+13");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%g", 0xa.aaaaaaaaaaaaaabp-7);
  ASSERT_STREQ_LEN(written, buff, L"0.0833333");

  // Simple Subnormal Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%g", 0x1.0p-1027);
  ASSERT_STREQ_LEN(written, buff, L"6.95336e-310");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%g", 0x1.0p-1074);
  ASSERT_STREQ_LEN(written, buff, L"4.94066e-324");

  // Inf/Nan Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%g", inf);
  ASSERT_STREQ_LEN(written, buff, L"inf");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%G", -inf);
  ASSERT_STREQ_LEN(written, buff, L"-INF");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%g", nan);
  ASSERT_STREQ_LEN(written, buff, L"nan");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%G", -nan);
  ASSERT_STREQ_LEN(written, buff, L"-NAN");

  // Min Width Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%15g", 1.0);
  ASSERT_STREQ_LEN(written, buff, L"              1");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%15g", -1.0);
  ASSERT_STREQ_LEN(written, buff, L"             -1");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%15g", 1.0e5);
  ASSERT_STREQ_LEN(written, buff, L"         100000");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%15g", -1.0e5);
  ASSERT_STREQ_LEN(written, buff, L"        -100000");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%10g", 1.0e-5);
  ASSERT_STREQ_LEN(written, buff, L"     1e-05");

  // Precision Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", 1.23456789);
  ASSERT_STREQ_LEN(written, buff, L"1.2");

  // Trimming trailing zeroes causes the precision to be ignored here.
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1g", 0.0);
  ASSERT_STREQ_LEN(written, buff, L"0");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.0g", 0.0);
  ASSERT_STREQ_LEN(written, buff, L"0");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", 0.1);
  ASSERT_STREQ_LEN(written, buff, L"0.1");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", 1.09);
  ASSERT_STREQ_LEN(written, buff, L"1.1");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", 1.04);
  ASSERT_STREQ_LEN(written, buff, L"1");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", 1.19);
  ASSERT_STREQ_LEN(written, buff, L"1.2");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", 1.99);
  ASSERT_STREQ_LEN(written, buff, L"2");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", 9.99);
  ASSERT_STREQ_LEN(written, buff, L"10");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.3g", 99.9);
  ASSERT_STREQ_LEN(written, buff, L"99.9");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", 99.9);
  ASSERT_STREQ_LEN(written, buff, L"1e+02");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1g", 99.9);
  ASSERT_STREQ_LEN(written, buff, L"1e+02");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.5g", 1.25);
  ASSERT_STREQ_LEN(written, buff, L"1.25");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.0g", 1.25);
  ASSERT_STREQ_LEN(written, buff, L"1");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.0g", 1.75);
  ASSERT_STREQ_LEN(written, buff, L"2");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.20g", 1.234e-10);
  ASSERT_STREQ_LEN(written, buff, L"1.2340000000000000814e-10");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.3g", -9.99);
  ASSERT_STREQ_LEN(written, buff, L"-9.99");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", -9.99);
  ASSERT_STREQ_LEN(written, buff, L"-10");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1g", -9.99);
  ASSERT_STREQ_LEN(written, buff, L"-1e+01");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.5g", 1.008);
  ASSERT_STREQ_LEN(written, buff, L"1.008");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.5g", 1.008e3);
  ASSERT_STREQ_LEN(written, buff, L"1008");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.4g", 9999.0);
  ASSERT_STREQ_LEN(written, buff, L"9999");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.3g", 9999.0);
  ASSERT_STREQ_LEN(written, buff, L"1e+04");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.3g", 1256.0);
  ASSERT_STREQ_LEN(written, buff, L"1.26e+03");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.7g", 9.99999e-310);
  ASSERT_STREQ_LEN(written, buff, L"9.99999e-310");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.6g", 9.99999e-310);
  ASSERT_STREQ_LEN(written, buff, L"9.99999e-310");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.5g", 9.99999e-310);
  ASSERT_STREQ_LEN(written, buff, L"1e-309");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.4g", 9.99999e-310);
  ASSERT_STREQ_LEN(written, buff, L"1e-309");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.3g", 9.99999e-310);
  ASSERT_STREQ_LEN(written, buff, L"1e-309");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", 9.99999e-310);
  ASSERT_STREQ_LEN(written, buff, L"1e-309");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.1g", 9.99999e-310);
  ASSERT_STREQ_LEN(written, buff, L"1e-309");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.0g", 9.99999e-310);
  ASSERT_STREQ_LEN(written, buff, L"1e-309");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.10g", 0x1.0p-1074);
  ASSERT_STREQ_LEN(written, buff, L"4.940656458e-324");

  // Rounding Mode Tests.

  if (ForceRoundingMode r(RoundingMode::Nearest); r.success) {
    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", 1.75);
    ASSERT_STREQ_LEN(written, buff, L"1.8");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", 1.25);
    ASSERT_STREQ_LEN(written, buff, L"1.2");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", 1.125);
    ASSERT_STREQ_LEN(written, buff, L"1.1");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", 1.625);
    ASSERT_STREQ_LEN(written, buff, L"1.6");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", 1.375);
    ASSERT_STREQ_LEN(written, buff, L"1.4");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", 1.875);
    ASSERT_STREQ_LEN(written, buff, L"1.9");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", -1.75);
    ASSERT_STREQ_LEN(written, buff, L"-1.8");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", -1.25);
    ASSERT_STREQ_LEN(written, buff, L"-1.2");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", -1.125);
    ASSERT_STREQ_LEN(written, buff, L"-1.1");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", -1.625);
    ASSERT_STREQ_LEN(written, buff, L"-1.6");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", -1.375);
    ASSERT_STREQ_LEN(written, buff, L"-1.4");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", -1.875);
    ASSERT_STREQ_LEN(written, buff, L"-1.9");
  }

  if (ForceRoundingMode r(RoundingMode::Upward); r.success) {
    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", 1.75);
    ASSERT_STREQ_LEN(written, buff, L"1.8");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", 1.25);
    ASSERT_STREQ_LEN(written, buff, L"1.3");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", 1.125);
    ASSERT_STREQ_LEN(written, buff, L"1.2");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", 1.625);
    ASSERT_STREQ_LEN(written, buff, L"1.7");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", 1.375);
    ASSERT_STREQ_LEN(written, buff, L"1.4");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", 1.875);
    ASSERT_STREQ_LEN(written, buff, L"1.9");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", -1.75);
    ASSERT_STREQ_LEN(written, buff, L"-1.7");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", -1.25);
    ASSERT_STREQ_LEN(written, buff, L"-1.2");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", -1.125);
    ASSERT_STREQ_LEN(written, buff, L"-1.1");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", -1.625);
    ASSERT_STREQ_LEN(written, buff, L"-1.6");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", -1.375);
    ASSERT_STREQ_LEN(written, buff, L"-1.3");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", -1.875);
    ASSERT_STREQ_LEN(written, buff, L"-1.8");
  }

  if (ForceRoundingMode r(RoundingMode::Downward); r.success) {
    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", 1.75);
    ASSERT_STREQ_LEN(written, buff, L"1.7");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", 1.25);
    ASSERT_STREQ_LEN(written, buff, L"1.2");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", 1.125);
    ASSERT_STREQ_LEN(written, buff, L"1.1");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", 1.625);
    ASSERT_STREQ_LEN(written, buff, L"1.6");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", 1.375);
    ASSERT_STREQ_LEN(written, buff, L"1.3");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", 1.875);
    ASSERT_STREQ_LEN(written, buff, L"1.8");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", -1.75);
    ASSERT_STREQ_LEN(written, buff, L"-1.8");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", -1.25);
    ASSERT_STREQ_LEN(written, buff, L"-1.3");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", -1.125);
    ASSERT_STREQ_LEN(written, buff, L"-1.2");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", -1.625);
    ASSERT_STREQ_LEN(written, buff, L"-1.7");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", -1.375);
    ASSERT_STREQ_LEN(written, buff, L"-1.4");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", -1.875);
    ASSERT_STREQ_LEN(written, buff, L"-1.9");
  }

  if (ForceRoundingMode r(RoundingMode::TowardZero); r.success) {
    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", 1.75);
    ASSERT_STREQ_LEN(written, buff, L"1.7");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", 1.25);
    ASSERT_STREQ_LEN(written, buff, L"1.2");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", 1.125);
    ASSERT_STREQ_LEN(written, buff, L"1.1");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", 1.625);
    ASSERT_STREQ_LEN(written, buff, L"1.6");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", 1.375);
    ASSERT_STREQ_LEN(written, buff, L"1.3");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", 1.875);
    ASSERT_STREQ_LEN(written, buff, L"1.8");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", -1.75);
    ASSERT_STREQ_LEN(written, buff, L"-1.7");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", -1.25);
    ASSERT_STREQ_LEN(written, buff, L"-1.2");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", -1.125);
    ASSERT_STREQ_LEN(written, buff, L"-1.1");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", -1.625);
    ASSERT_STREQ_LEN(written, buff, L"-1.6");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", -1.375);
    ASSERT_STREQ_LEN(written, buff, L"-1.3");

    written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2g", -1.875);
    ASSERT_STREQ_LEN(written, buff, L"-1.8");
  }

  // Flag Tests.
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%+g", 1.0);
  ASSERT_STREQ_LEN(written, buff, L"+1");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%+g", -1.0);
  ASSERT_STREQ_LEN(written, buff, L"-1");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"% g", 1.0);
  ASSERT_STREQ_LEN(written, buff, L" 1");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"% g", -1.0);
  ASSERT_STREQ_LEN(written, buff, L"-1");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%-15g", 1.5);
  ASSERT_STREQ_LEN(written, buff, L"1.5            ");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%#.g", 1.0);
  ASSERT_STREQ_LEN(written, buff, L"1.");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%#g", 1.0);
  ASSERT_STREQ_LEN(written, buff, L"1.00000");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%#.0g", 1.5);
  ASSERT_STREQ_LEN(written, buff, L"2.");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%015g", 1.5);
  ASSERT_STREQ_LEN(written, buff, L"0000000000001.5");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%015g", -1.5);
  ASSERT_STREQ_LEN(written, buff, L"-000000000001.5");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%+- #0g", 0.0);
  ASSERT_STREQ_LEN(written, buff, L"+0.00000");

  // Combined Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%10.3g", 9.99);
  ASSERT_STREQ_LEN(written, buff, L"      9.99");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%10.2g", 9.99);
  ASSERT_STREQ_LEN(written, buff, L"        10");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%10.1g", 9.99);
  ASSERT_STREQ_LEN(written, buff, L"     1e+01");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%-10.3g", 9.99);
  ASSERT_STREQ_LEN(written, buff, L"9.99      ");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%-10.2g", 9.99);
  ASSERT_STREQ_LEN(written, buff, L"10        ");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%-10.1g", 9.99);
  ASSERT_STREQ_LEN(written, buff, L"1e+01     ");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%-10.1g", 1.0e-50);
  ASSERT_STREQ_LEN(written, buff, L"1e-50     ");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%30g", 1234567890123456789.0);
  ASSERT_STREQ_LEN(written, buff, L"                   1.23457e+18");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%-30g", 1234567890123456789.0);
  ASSERT_STREQ_LEN(written, buff, L"1.23457e+18                   ");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%25.15g", 9999999999999.99);
  ASSERT_STREQ_LEN(written, buff, L"         9999999999999.99");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%25.14g", 9999999999999.99);
  ASSERT_STREQ_LEN(written, buff, L"           10000000000000");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%25.13g", 9999999999999.99);
  ASSERT_STREQ_LEN(written, buff, L"                    1e+13");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%#12.3g %-12.3g", 0.1, 256.0);
  ASSERT_STREQ_LEN(written, buff, L"       0.100 256         ");

  written =
      LIBC_NAMESPACE::swprintf(buff, N, L"%+-#12.3g % 012.3g", 0.1256, 1256.0);
  ASSERT_STREQ_LEN(written, buff, L"+0.126        0001.26e+03");
}

TEST_F(LlvmLibcSwprintfTest, FloatAutoLongDoubleConv) {
  constexpr size_t N = 1000;
  wchar_t buff[N];
  int written;

  ForceRoundingMode r(RoundingMode::Nearest);

  // Length Modifier Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%Lg", 9.99999999999e-100L);
  ASSERT_STREQ_LEN(written, buff, L"1e-99");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%Lg", 1e100L);
  ASSERT_STREQ_LEN(written, buff, L"1e+100");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%Lg", 1.0L);
  ASSERT_STREQ_LEN(written, buff, L"1");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%Lg", 0.1L);
  ASSERT_STREQ_LEN(written, buff, L"0.1");
}

#if defined(LIBC_INTERNAL_PRINTF_CONVERT_FLOAT128)
TEST_F(LlvmLibcSwprintfTest, FloatAutoFloat128Conv) {
  constexpr size_t N = 1000;
  wchar_t buff[N];
  int written;

  ForceRoundingMode r(RoundingMode::Nearest);

  written =
      LIBC_NAMESPACE::swprintf(buff, N, L"%Qg", float128(9.99999999999e-100L));
  ASSERT_STREQ_LEN(written, buff, L"1e-99");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%Qg", float128(1e100L));
  ASSERT_STREQ_LEN(written, buff, L"1e+100");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%Qg", float128(1.0L));
  ASSERT_STREQ_LEN(written, buff, L"1");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%Qg", float128(0.1L));
  ASSERT_STREQ_LEN(written, buff, L"0.1");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%Qg", float128(1e1000L));
  ASSERT_STREQ_LEN(written, buff, L"1e+1000");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%Qg", float128(1e4900L));
  ASSERT_STREQ_LEN(written, buff, L"1e+4900");
}
#endif // LIBC_INTERNAL_PRINTF_CONVERT_FLOAT128

#endif // LIBC_COPT_PRINTF_DISABLE_FLOAT

#if defined(LIBC_COMPILER_HAS_FIXED_POINT) &&                                  \
    !defined(LIBC_COPT_PRINTF_DISABLE_FIXED_POINT)
TEST_F(LlvmLibcSwprintfTest, FixedConv) {
  constexpr size_t N = 1000;
  wchar_t buff[N];
  int written;

  // TODO: Replace hex literals with their appropriate fixed point literals.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%k", 0x0); // 0.0
  ASSERT_STREQ_LEN(written, buff, L"0.000000");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%k", 0x80000000); // -0.0
  ASSERT_STREQ_LEN(written, buff, L"-0.000000");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%r", 0xffff); // -fract max
  ASSERT_STREQ_LEN(written, buff, L"-0.999969");

  written =
      LIBC_NAMESPACE::swprintf(buff, N, L"%R", 0xffff); // unsigned fract max
  ASSERT_STREQ_LEN(written, buff, L"0.999985");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%k", 0xffffffff); // -accum max
  ASSERT_STREQ_LEN(written, buff, L"-65535.999969");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%K",
                                     0xffffffff); // unsigned accum max
  ASSERT_STREQ_LEN(written, buff, L"65535.999985");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%r", 0x7fff); // fract max
  ASSERT_STREQ_LEN(written, buff, L"0.999969");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%k", 0x7fffffff); // accum max
  ASSERT_STREQ_LEN(written, buff, L"65535.999969");

  // Length Modifier Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%hk", 0x0); // 0.0
  ASSERT_STREQ_LEN(written, buff, L"0.000000");

  written =
      LIBC_NAMESPACE::swprintf(buff, N, L"%hk", 0xffff); // -short accum max
  ASSERT_STREQ_LEN(written, buff, L"-255.992188");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%hr", 0x0); // 0.0
  ASSERT_STREQ_LEN(written, buff, L"0.000000");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%hr", 0xff); // -short fract max
  ASSERT_STREQ_LEN(written, buff, L"-0.992188");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%hK", 0x0); // 0.0
  ASSERT_STREQ_LEN(written, buff, L"0.000000");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%hK",
                                     0xffff); // unsigned short accum max
  ASSERT_STREQ_LEN(written, buff, L"255.996094");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%hR", 0x0); // 0.0
  ASSERT_STREQ_LEN(written, buff, L"0.000000");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%hR",
                                     0xff); // unsigned short fract max
  ASSERT_STREQ_LEN(written, buff, L"0.996094");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%lk", 0x0ll); // 0.0
  ASSERT_STREQ_LEN(written, buff, L"0.000000");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%lk",
                                     0xffffffffffffffff); //-long accum max
  ASSERT_STREQ_LEN(written, buff, L"-4294967296.000000");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%lr", 0x0); // 0.0
  ASSERT_STREQ_LEN(written, buff, L"0.000000");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%lr",
                                     0xffffffff); //-long fract max
  ASSERT_STREQ_LEN(written, buff, L"-1.000000");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%lK", 0x0ll); // 0.0
  ASSERT_STREQ_LEN(written, buff, L"0.000000");

  written =
      LIBC_NAMESPACE::swprintf(buff, N, L"%lK",
                               0xffffffffffffffff); // unsigned long accum max
  ASSERT_STREQ_LEN(written, buff, L"4294967296.000000");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%lR", 0x0); // 0.0
  ASSERT_STREQ_LEN(written, buff, L"0.000000");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%lR",
                                     0xffffffff); // unsigned long fract max
  ASSERT_STREQ_LEN(written, buff, L"1.000000");

  // Min Width Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%10k", 0x0000a000); // 1.25
  ASSERT_STREQ_LEN(written, buff, L"  1.250000");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%10k", 0x8000a000); //-1.25
  ASSERT_STREQ_LEN(written, buff, L" -1.250000");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%8k", 0x0000a000); // 1.25
  ASSERT_STREQ_LEN(written, buff, L"1.250000");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%9k", 0x8000a000); //-1.25
  ASSERT_STREQ_LEN(written, buff, L"-1.250000");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%4k", 0x0000a000); // 1.25
  ASSERT_STREQ_LEN(written, buff, L"1.250000");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%4k", 0x8000a000); //-1.25
  ASSERT_STREQ_LEN(written, buff, L"-1.250000");

  // Precision Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.16K",
                                     0xFFFFFFFF); // unsigned accum max
  ASSERT_STREQ_LEN(written, buff, L"65535.9999847412109375");

  written =
      LIBC_NAMESPACE::swprintf(buff, N, L"%.32lK",
                               0xFFFFFFFFFFFFFFFF); // unsigned long accum max
  ASSERT_STREQ_LEN(written, buff,
                   L"4294967295.99999999976716935634613037109375");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.0K",
                                     0xFFFFFFFF); // unsigned accum max
  ASSERT_STREQ_LEN(written, buff, L"65536");

  written =
      LIBC_NAMESPACE::swprintf(buff, N, L"%.0R", 0xFFFF); // unsigned fract max
  ASSERT_STREQ_LEN(written, buff, L"1");

  // Flag Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%+k", 0x0000a000); // 1.25
  ASSERT_STREQ_LEN(written, buff, L"+1.250000");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%+k", 0x8000a000); //-1.25
  ASSERT_STREQ_LEN(written, buff, L"-1.250000");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"% k", 0x0000a000); // 1.25
  ASSERT_STREQ_LEN(written, buff, L" 1.250000");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"% k", 0x8000a000); //-1.25
  ASSERT_STREQ_LEN(written, buff, L"-1.250000");

  // unsigned variants ignore sign flags.
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%+K", 0x00014000); // 1.25
  ASSERT_STREQ_LEN(written, buff, L"1.250000");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"% K", 0x00014000); // 1.25
  ASSERT_STREQ_LEN(written, buff, L"1.250000");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%-10k", 0x0000c000); // 1.5
  ASSERT_STREQ_LEN(written, buff, L"1.500000  ");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%#.k", 0x00008000); // 1.0
  ASSERT_STREQ_LEN(written, buff, L"1.");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%#.0k", 0x0000c000); // 1.5
  ASSERT_STREQ_LEN(written, buff, L"2.");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%010k", 0x0000c000); // 1.5
  ASSERT_STREQ_LEN(written, buff, L"001.500000");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%010k", 0x8000c000); //-1.5
  ASSERT_STREQ_LEN(written, buff, L"-01.500000");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%+- #0k", 0); // 0.0
  ASSERT_STREQ_LEN(written, buff, L"+0.000000");

  // Combined Tests.

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%10.2k", 0x0004feb8); // 9.99
  ASSERT_STREQ_LEN(written, buff, L"      9.99");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%5.1k", 0x0004feb8); // 9.99
  ASSERT_STREQ_LEN(written, buff, L" 10.0");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%-10.2k", 0x0004feb8); // 9.99
  ASSERT_STREQ_LEN(written, buff, L"9.99      ");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%-5.1k", 0x0004feb8); // 9.99
  ASSERT_STREQ_LEN(written, buff, L"10.0 ");

  written =
      LIBC_NAMESPACE::swprintf(buff, N, L"%-5.1k", 0x00000001); // accum min
  ASSERT_STREQ_LEN(written, buff, L"0.0  ");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%30k", 0x7fffffff); // accum max
  ASSERT_STREQ_LEN(written, buff, L"                  65535.999969");

  written =
      LIBC_NAMESPACE::swprintf(buff, N, L"%-30k", 0x7fffffff); // accum max
  ASSERT_STREQ_LEN(written, buff, L"65535.999969                  ");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%20.2lK",
                                     0x3b9ac9ffFD70A3D7); // 999999999.99
  ASSERT_STREQ_LEN(written, buff, L"        999999999.99");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%20.1lK",
                                     0x3b9ac9ffFD70A3D7); // 999999999.99
  ASSERT_STREQ_LEN(written, buff, L"        1000000000.0");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%12.3R %-12.3k", 0x1999,
                                     0x00800000); // 0.1, 256.0
  ASSERT_STREQ_LEN(written, buff, L"       0.100 256.000     ");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%+-#12.3lk % 012.3k",
                                     0x000000001013a92all,
                                     0x02740000); // 0.126, 1256.0
  ASSERT_STREQ_LEN(written, buff, L"+0.126        0001256.000");
}
#endif // defined(LIBC_COMPILER_HAS_FIXED_POINT) &&
       // !defined(LIBC_COPT_PRINTF_DISABLE_FIXED_POINT)

#ifndef LIBC_COPT_PRINTF_DISABLE_STRERROR
TEST_F(LlvmLibcSwprintfTest, StrerrorConv) {
  constexpr size_t N = 1000;
  wchar_t buff[N];
  int written;

  libc_errno = 0;
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%m");
  ASSERT_STREQ_LEN(written, buff, L"Success");

  libc_errno = ERANGE;
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%m");
  ASSERT_STREQ_LEN(written, buff, L"Numerical result out of range");

  // Check that it correctly consumes no arguments.
  libc_errno = 0;
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%m %d", 1);
  ASSERT_STREQ_LEN(written, buff, L"Success 1");

  // Width Tests

  libc_errno = 0;
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%10m");
  ASSERT_STREQ_LEN(written, buff, L"   Success");

  libc_errno = ERANGE;
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%10m");
  ASSERT_STREQ_LEN(written, buff, L"Numerical result out of range");

  // Precision Tests

  libc_errno = 0;
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.10m");
  ASSERT_STREQ_LEN(written, buff, L"Success");

  libc_errno = ERANGE;
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.10m");
  ASSERT_STREQ_LEN(written, buff, L"Numerical ");

  // Flag Tests (Only '-' since the others only affect ints)

  libc_errno = 0;
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%-10m");
  ASSERT_STREQ_LEN(written, buff, L"Success   ");

  libc_errno = ERANGE;
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%-10m");
  ASSERT_STREQ_LEN(written, buff, L"Numerical result out of range");

  // Alt Mode Tests
  // Since alt mode here is effectively a completely separate conversion, it
  // gets separate tests.

  libc_errno = 0;
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%#m");
  ASSERT_STREQ_LEN(written, buff, L"0");

  libc_errno = ERANGE;
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%#m");
  ASSERT_STREQ_LEN(written, buff, L"ERANGE");

  libc_errno = -9999;
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%#m");
  ASSERT_STREQ_LEN(written, buff, L"-9999");

  // Alt Mode Width

  libc_errno = 0;
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%#10m");
  ASSERT_STREQ_LEN(written, buff, L"         0");

  libc_errno = ERANGE;
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%#10m");
  ASSERT_STREQ_LEN(written, buff, L"    ERANGE");

  libc_errno = -9999;
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%#10m");
  ASSERT_STREQ_LEN(written, buff, L"     -9999");

  libc_errno = ERANGE;
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%#3m");
  ASSERT_STREQ_LEN(written, buff, L"ERANGE");

  libc_errno = -9999;
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%#3m");
  ASSERT_STREQ_LEN(written, buff, L"-9999");

  // Alt Mode Precision

  libc_errno = ERANGE;
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%#.10m");
  ASSERT_STREQ_LEN(written, buff, L"ERANGE");

  libc_errno = -9999;
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%#.10m");
  ASSERT_STREQ_LEN(written, buff, L"-0000009999");

  libc_errno = ERANGE;
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%#.3m");
  ASSERT_STREQ_LEN(written, buff, L"ERA");

  libc_errno = -9999;
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%#.3m");
  ASSERT_STREQ_LEN(written, buff, L"-9999");

  // We don't test precision (or int flags) on errno = 0 because it behaves
  // weirdly, see the docs for more information.
  libc_errno = 0;
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%#.1m");
  ASSERT_STREQ_LEN(written, buff, L"0");

  // Alt Mode Flags

  // '-' flag
  libc_errno = 0;
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%#-10m");
  ASSERT_STREQ_LEN(written, buff, L"0         ");

  libc_errno = ERANGE;
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%#-10m");
  ASSERT_STREQ_LEN(written, buff, L"ERANGE    ");

  libc_errno = -9999;
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%#-10m");
  ASSERT_STREQ_LEN(written, buff, L"-9999     ");

  libc_errno = ERANGE;
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%#-3m");
  ASSERT_STREQ_LEN(written, buff, L"ERANGE");

  libc_errno = -9999;
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%#-3m");
  ASSERT_STREQ_LEN(written, buff, L"-9999");

  // '+' flag
  libc_errno = ERANGE;
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%#+m");
  ASSERT_STREQ_LEN(written, buff, L"ERANGE");

  libc_errno = -9999;
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%#+m");
  ASSERT_STREQ_LEN(written, buff, L"-9999");

  libc_errno = 9999;
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%#+m");
  ASSERT_STREQ_LEN(written, buff, L"+9999");

  // ' ' flag
  libc_errno = ERANGE;
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%# m");
  ASSERT_STREQ_LEN(written, buff, L"ERANGE");

  libc_errno = -9999;
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%# m");
  ASSERT_STREQ_LEN(written, buff, L"-9999");

  libc_errno = 9999;
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%# m");
  ASSERT_STREQ_LEN(written, buff, L" 9999");

  // '0' flag

  libc_errno = ERANGE;
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%#010m");
  ASSERT_STREQ_LEN(written, buff, L"    ERANGE");

  libc_errno = -9999;
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%#010m");
  ASSERT_STREQ_LEN(written, buff, L"-000009999");

  libc_errno = ERANGE;
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%#03m");
  ASSERT_STREQ_LEN(written, buff, L"ERANGE");

  libc_errno = -9999;
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%#03m");
  ASSERT_STREQ_LEN(written, buff, L"-9999");
  libc_errno = 0;
}
#endif // LIBC_COPT_PRINTF_DISABLE_STRERROR

#ifndef LIBC_COPT_PRINTF_DISABLE_WRITE_INT
TEST_F(LlvmLibcSwprintfTest, WriteIntConv) {
  constexpr size_t N = 64;
  wchar_t buff[N];
  int written;
  int test_val = -1;

  test_val = -1;
  written = LIBC_NAMESPACE::swprintf(buff, N, L"12345%n67890", &test_val);
  EXPECT_EQ(test_val, 5);
  ASSERT_STREQ_LEN(written, buff, L"1234567890");

  test_val = -1;
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%n", &test_val);
  EXPECT_EQ(test_val, 0);
  ASSERT_STREQ_LEN(written, buff, L"");

  test_val = 0x100;
  written = LIBC_NAMESPACE::swprintf(buff, N, L"ABC%hhnDEF", &test_val);
  EXPECT_EQ(test_val, 0x103);
  ASSERT_STREQ_LEN(written, buff, L"ABCDEF");

  test_val = -1;
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%s%n", "87654321", &test_val);
  EXPECT_EQ(test_val, 8);
  ASSERT_STREQ_LEN(written, buff, L"87654321");

#ifndef LIBC_COPT_PRINTF_NO_NULLPTR_CHECKS
  written = LIBC_NAMESPACE::swprintf(buff, N, L"abc123%n", nullptr);
  EXPECT_LT(written, 0);
  ASSERT_ERRNO_FAILURE();
#endif // LIBC_COPT_PRINTF_NO_NULLPTR_CHECKS
}
#endif // LIBC_COPT_PRINTF_DISABLE_WRITE_INT

#ifndef LIBC_COPT_PRINTF_DISABLE_INDEX_MODE
TEST_F(LlvmLibcSwprintfTest, IndexModeParsing) {
  constexpr size_t N = 64;
  wchar_t buff[N];
  int written;

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%1$s", "abcDEF123");
  ASSERT_STREQ_LEN(written, buff, L"abcDEF123");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%1$s %%", "abcDEF123");
  ASSERT_STREQ_LEN(written, buff, L"abcDEF123 %");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%3$s %1$s %2$s", "is", "hard",
                                     "ordering");
  ASSERT_STREQ_LEN(written, buff, L"ordering is hard");

  written = LIBC_NAMESPACE::swprintf(
      buff, N,
      L"%10$s %9$s %8$c %7$s %6$s, %6$s %5$s %4$-*1$s %3$.*11$s %2$s. %%", 6,
      "pain", "alphabetical", "such", "is", "this", "do", 'u', "would", "why",
      1);
  ASSERT_STREQ_LEN(written, buff,
                   L"why would u do this, this is such   a pain. %");
}
#endif // LIBC_COPT_PRINTF_DISABLE_INDEX_MODE

#if !defined(LIBC_COPT_PRINTF_DISABLE_WIDE) &&                                 \
    !defined(LIBC_TARGET_ARCH_IS_GPU) // llvm.org/pr214433
TEST_F(LlvmLibcSwprintfTest, WideCharConversion) {
  constexpr size_t N = 16;
  wchar_t buff[N];
  int written;

  // 1 byte UTF-8 character.
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%lc", L'A');
  EXPECT_EQ(written, 1);
  ASSERT_STREQ_LEN(written, buff, L"A");

  // 1 byte UTF-8 character left justified.
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%-4lc", L'A');
  EXPECT_EQ(written, 4);
  ASSERT_STREQ_LEN(written, buff, L"A   ");

  // 1 byte UTF-8 character right justified.
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%4lc", L'A');
  EXPECT_EQ(written, 4);
  ASSERT_STREQ_LEN(written, buff, L"   A");

  // 2 byte UTF-8 character (1 wchar_t).
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%lc", L'¢');
  EXPECT_EQ(written, 1);
  ASSERT_STREQ_LEN(written, buff, L"¢");

  // 2 byte UTF-8 character left justified.
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%-4lc", L'¢');
  EXPECT_EQ(written, 4);
  ASSERT_STREQ_LEN(written, buff, L"¢   ");

  // 2 byte UTF-8 character right justified.
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%4lc", L'¢');
  EXPECT_EQ(written, 4);
  ASSERT_STREQ_LEN(written, buff, L"   ¢");

  // Euro sign is a 3-byte UTF-8 character (1 wchar_t).
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%lc", L'€');
  EXPECT_EQ(written, 1);
  ASSERT_STREQ_LEN(written, buff, L"€");

  // Euro sign left justified.
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%-4lc", L'€');
  EXPECT_EQ(written, 4);
  ASSERT_STREQ_LEN(written, buff, L"€   ");

  // Euro sign right justified.
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%4lc", L'€');
  EXPECT_EQ(written, 4);
  ASSERT_STREQ_LEN(written, buff, L"   €");

  // Euro sign right justified with plus flag.
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%+4lc", L'€');
  EXPECT_EQ(written, 4);
  ASSERT_STREQ_LEN(written, buff, L"   €");

  // Grinning face emoji is a 4-byte UTF-8 character (1 wchar_t).
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%lc", L'😀');
  EXPECT_EQ(written, 1);
  ASSERT_STREQ_LEN(written, buff, L"😀");

  // Grinning face emoji left justified.
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%-4lc", L'😀');
  EXPECT_EQ(written, 4);
  ASSERT_STREQ_LEN(written, buff, L"😀   ");

  // Grinning face emoji right justified.
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%4lc", L'😀');
  EXPECT_EQ(written, 4);
  ASSERT_STREQ_LEN(written, buff, L"   😀");

  // WEOF test.
  EXPECT_EQ(LIBC_NAMESPACE::swprintf(buff, N, L"%lc", WEOF), -1);
  ASSERT_ERRNO_EQ(EILSEQ);

  // Invalid narrow character test (outside ASCII 0..127).
  EXPECT_EQ(LIBC_NAMESPACE::swprintf(buff, N, L"%c", 128), -1);
  ASSERT_ERRNO_EQ(EILSEQ);
}

TEST_F(LlvmLibcSwprintfTest, WideCharStringConversion) {
  constexpr size_t N = 64;
  wchar_t buff[N];
  int written;

  // Basic test.
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%ls", L"Hello, World!");
  EXPECT_EQ(written, 13);
  ASSERT_STREQ_LEN(written, buff, L"Hello, World!");

  // Left justified.
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%-10ls", L"Hello, World!");
  EXPECT_EQ(written, 13);
  ASSERT_STREQ_LEN(written, buff, L"Hello, World!");

  // Precision test.
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.5ls", L"Hello, World!");
  EXPECT_EQ(written, 5);
  ASSERT_STREQ_LEN(written, buff, L"Hello");

  // Test ¢.
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%ls", L"¢");
  EXPECT_EQ(written, 1);
  ASSERT_STREQ_LEN(written, buff, L"¢");

  // Test €.
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%ls", L"€");
  EXPECT_EQ(written, 1);
  ASSERT_STREQ_LEN(written, buff, L"€");

  // Test 😀.
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%ls", L"😀");
  EXPECT_EQ(written, 1);
  ASSERT_STREQ_LEN(written, buff, L"😀");

  // Test wide string with multiple non-ASCII characters.
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%ls", L"¢€😀");
  EXPECT_EQ(written, 3);
  ASSERT_STREQ_LEN(written, buff, L"¢€😀");

  // Width smaller than string length.
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%2ls", L"¢€😀");
  EXPECT_EQ(written, 3);
  ASSERT_STREQ_LEN(written, buff, L"¢€😀");

  // Width with padding.
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%6ls", L"¢€😀");
  EXPECT_EQ(written, 6);
  ASSERT_STREQ_LEN(written, buff, L"   ¢€😀");

  // Precision with wide characters.
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.2ls", L"¢€😀");
  EXPECT_EQ(written, 2);
  ASSERT_STREQ_LEN(written, buff, L"¢€");

  // Precision with padding and wide characters.
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%5.2ls", L"¢€😀");
  EXPECT_EQ(written, 5);
  ASSERT_STREQ_LEN(written, buff, L"   ¢€");

  // Precision with left justification and wide characters.
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%-5.2ls", L"¢€😀");
  EXPECT_EQ(written, 5);
  ASSERT_STREQ_LEN(written, buff, L"¢€   ");

  // Multibyte narrow string (%s) conversion to wide characters.
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%s", "¢€😀");
  EXPECT_EQ(written, 3);
  ASSERT_STREQ_LEN(written, buff, L"¢€😀");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%6s", "¢€😀");
  EXPECT_EQ(written, 6);
  ASSERT_STREQ_LEN(written, buff, L"   ¢€😀");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%5.2s", "¢€😀");
  EXPECT_EQ(written, 5);
  ASSERT_STREQ_LEN(written, buff, L"   ¢€");

  written = LIBC_NAMESPACE::swprintf(buff, N, L"%-5.2s", "¢€😀");
  EXPECT_EQ(written, 5);
  ASSERT_STREQ_LEN(written, buff, L"¢€   ");

  // NULL string test.
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%ls", nullptr);
  EXPECT_EQ(written, 6);
  ASSERT_STREQ_LEN(written, buff, L"(null)");

  // NULL string with precision.
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%.3ls", nullptr);
  EXPECT_EQ(written, 3);
  ASSERT_STREQ_LEN(written, buff, L"(nu");

  // NULL string with precision and width.
  written = LIBC_NAMESPACE::swprintf(buff, N, L"%6.3ls", nullptr);
  EXPECT_EQ(written, 6);
  ASSERT_STREQ_LEN(written, buff, L"   (nu");
}
#endif // !LIBC_COPT_PRINTF_DISABLE_WIDE && !LIBC_TARGET_ARCH_IS_GPU

TEST_F(LlvmLibcSwprintfTest, CutOff) {
  constexpr size_t N = 100;
  wchar_t buff[N];
  int written;

  written = LIBC_NAMESPACE::swprintf(buff, 16,
                                     L"A simple string with no conversions.");
  EXPECT_EQ(written, 36);
  ASSERT_STREQ(buff, L"A simple string");

  written = LIBC_NAMESPACE::swprintf(buff, 5, L"%s", "1234567890");
  EXPECT_EQ(written, 10);
  ASSERT_STREQ(buff, L"1234");

  written = LIBC_NAMESPACE::swprintf(buff, 5, L"%ls", L"1234567890");
  EXPECT_EQ(written, 10);
  ASSERT_STREQ(buff, L"1234");

  written = LIBC_NAMESPACE::swprintf(buff, 67, L"%-101c", 'a');
  EXPECT_EQ(written, 101);
  ASSERT_STREQ(buff, L"a "
                     L"        " // Each of these is 8 spaces, and there are 8.
                     L"        " // In total there are 65 spaces
                     L"        " // L'a' + 65 spaces + L'\0' = 67
                     L"        "
                     L"        "
                     L"        "
                     L"        "
                     L"        ");

  // passing null as the output pointer is allowed as long as bufsz is 0.
  written = LIBC_NAMESPACE::swprintf(nullptr, 0, L"%s and more", "1234567890");
  EXPECT_EQ(written, 19);

  written = LIBC_NAMESPACE::swprintf(nullptr, 0, L"%*s", INT_MIN, "nothing");
  EXPECT_EQ(written, INT_MAX);
}

TEST_F(LlvmLibcSwprintfTest, NoCutOff) {
  constexpr size_t N = 64;
  wchar_t buff[N];
  int written;

  written = LIBC_NAMESPACE::swprintf(buff, 37,
                                     L"A simple string with no conversions.");
  EXPECT_EQ(written, 36);
  ASSERT_STREQ(buff, L"A simple string with no conversions.");

  written = LIBC_NAMESPACE::swprintf(buff, 20, L"%s", "1234567890");
  EXPECT_EQ(written, 10);
  ASSERT_STREQ(buff, L"1234567890");

  written = LIBC_NAMESPACE::swprintf(buff, 20, L"%ls", L"1234567890");
  EXPECT_EQ(written, 10);
  ASSERT_STREQ(buff, L"1234567890");
}

TEST_F(LlvmLibcSwprintfTest, CharsWrittenOverflow) {
  wchar_t buff[0];

  // Trigger an overflow in the return value of swprintf by writing more than
  // INT_MAX wide characters.
  int int_max = LIBC_NAMESPACE::cpp::numeric_limits<int>::max();
  int written = LIBC_NAMESPACE::swprintf(buff, 0, L"%*stest", int_max, "");
  EXPECT_LT(written, 0);
  ASSERT_ERRNO_FAILURE();
}

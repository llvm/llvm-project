//===-- Unittests for strtof ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/errno/libc_errno.h"
#include "src/stdlib/strtof.h"

#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/RoundingModeUtils.h"
#include "test/UnitTest/Test.h"

#include <limits.h>
#include <stddef.h>

using LIBC_NAMESPACE::fputil::testing::ForceRoundingModeTest;
using LIBC_NAMESPACE::fputil::testing::RoundingMode;

class LlvmLibcStrToFTest : public LIBC_NAMESPACE::testing::Test,
                           ForceRoundingModeTest<RoundingMode::Nearest> {
public:
  void run_test(const char *inputString, const ptrdiff_t expectedStrLen,
                const uint32_t expectedRawData, const int expectedErrno = 0) {
    // expectedRawData is the expected float result as a uint32_t, organized
    // according to IEEE754:
    //
    // +-- 1 Sign Bit      +-- 23 Mantissa bits
    // |                   |
    // |        +----------+----------+
    // |        |                     |
    // SEEEEEEEEMMMMMMMMMMMMMMMMMMMMMMM
    //  |      |
    //  +--+---+
    //     |
    //     +-- 8 Exponent Bits
    //
    //  This is so that the result can be compared in parts.
    char *str_end = nullptr;

    LIBC_NAMESPACE::fputil::FPBits<float> expected_fp =
        LIBC_NAMESPACE::fputil::FPBits<float>(expectedRawData);

    libc_errno = 0;
    float result = LIBC_NAMESPACE::strtof(inputString, &str_end);

    EXPECT_EQ(str_end - inputString, expectedStrLen);
    EXPECT_FP_EQ(result, static_cast<float>(expected_fp));
    EXPECT_EQ(libc_errno, expectedErrno);
  }
};

// This is the set of tests that I have working (verified correct when compared
// to system libc). This is here so I don't break more things when I try to fix
// them.

TEST_F(LlvmLibcStrToFTest, BasicDecimalTests) {
  run_test("1", 1, 0x3f800000);
  run_test("123", 3, 0x42f60000);
  run_test("1234567890", 10, 0x4e932c06u);
  run_test("123456789012345678901", 21, 0x60d629d4);
  run_test("0.1", 3, 0x3dcccccdu);
  run_test(".1", 2, 0x3dcccccdu);
  run_test("-0.123456789", 12, 0xbdfcd6eau);
  run_test("0.11111111111111111111", 22, 0x3de38e39u);
  run_test("0.0000000000000000000000001", 27, 0x15f79688u);
}

TEST_F(LlvmLibcStrToFTest, DecimalOutOfRangeTests) {
  run_test("555E36", 6, 0x7f800000, ERANGE);
  run_test("1e-10000", 8, 0x0, ERANGE);
}

TEST_F(LlvmLibcStrToFTest, DecimalsWithRoundingProblems) {
  run_test("20040229", 8, 0x4b98e512);
  run_test("20040401", 8, 0x4b98e568);
  run_test("9E9", 3, 0x50061c46);
}

TEST_F(LlvmLibcStrToFTest, DecimalSubnormals) {
  run_test("1.4012984643248170709237295832899161312802619418765e-45", 55, 0x1,
           ERANGE);
}

TEST_F(LlvmLibcStrToFTest, DecimalWithLongExponent) {
  run_test("1e2147483648", 12, 0x7f800000, ERANGE);
  run_test("1e2147483646", 12, 0x7f800000, ERANGE);
  run_test("100e2147483646", 14, 0x7f800000, ERANGE);
  run_test("1e-2147483647", 13, 0x0, ERANGE);
  run_test("1e-2147483649", 13, 0x0, ERANGE);
}

TEST_F(LlvmLibcStrToFTest, BasicHexadecimalTests) {
  run_test("0x1", 3, 0x3f800000);
  run_test("0x10", 4, 0x41800000);
  run_test("0x11", 4, 0x41880000);
  run_test("0x0.1234", 8, 0x3d91a000);
}

TEST_F(LlvmLibcStrToFTest, HexadecimalSubnormalTests) {
  run_test("0x0.0000000000000000000000000000000002", 38, 0x4000, ERANGE);

  // This is the largest subnormal number as represented in hex
  run_test("0x0.00000000000000000000000000000003fffff8", 42, 0x7fffff, ERANGE);
}

TEST_F(LlvmLibcStrToFTest, HexadecimalSubnormalRoundingTests) {
  // This is the largest subnormal number that gets rounded down to 0 (as a
  // float)
  run_test("0x0.00000000000000000000000000000000000004", 42, 0x0, ERANGE);

  // This is slightly larger, and thus rounded up
  run_test("0x0.000000000000000000000000000000000000041", 43, 0x00000001,
           ERANGE);

  // These check that we're rounding to even properly
  run_test("0x0.0000000000000000000000000000000000000b", 42, 0x00000001,
           ERANGE);
  run_test("0x0.0000000000000000000000000000000000000c", 42, 0x00000002,
           ERANGE);

  // These check that we're rounding to even properly even when the input bits
  // are longer than the bit fields can contain.
  run_test("0x1.000000000000000000000p-150", 30, 0x00000000, ERANGE);
  run_test("0x1.000010000000000001000p-150", 30, 0x00000001, ERANGE);
  run_test("0x1.000100000000000001000p-134", 30, 0x00008001, ERANGE);
  run_test("0x1.FFFFFC000000000001000p-127", 30, 0x007FFFFF, ERANGE);
  run_test("0x1.FFFFFE000000000000000p-127", 30, 0x00800000);
}

TEST_F(LlvmLibcStrToFTest, HexadecimalNormalRoundingTests) {
  // This also checks the round to even behavior by checking three adjacent
  // numbers.
  // This gets rounded down to even
  run_test("0x123456500", 11, 0x4f91a2b2);
  // This doesn't get rounded at all
  run_test("0x123456600", 11, 0x4f91a2b3);
  // This gets rounded up to even
  run_test("0x123456700", 11, 0x4f91a2b4);
  // Correct rounding for long input
  run_test("0x1.000001000000000000000", 25, 0x3f800000);
  run_test("0x1.000001000000000000100", 25, 0x3f800001);
}

TEST_F(LlvmLibcStrToFTest, HexadecimalsWithRoundingProblems) {
  run_test("0xFFFFFFFF", 10, 0x4f800000);
}

TEST_F(LlvmLibcStrToFTest, HexadecimalOutOfRangeTests) {
  run_test("0x123456789123456789123456789123456789", 38, 0x7f800000, ERANGE);
  run_test("-0x123456789123456789123456789123456789", 39, 0xff800000, ERANGE);
  run_test("0x0.00000000000000000000000000000000000001", 42, 0x0, ERANGE);
}

TEST_F(LlvmLibcStrToFTest, InfTests) {
  run_test("INF", 3, 0x7f800000);
  run_test("INFinity", 8, 0x7f800000);
  run_test("infnity", 3, 0x7f800000);
  run_test("infinit", 3, 0x7f800000);
  run_test("infinfinit", 3, 0x7f800000);
  run_test("innf", 0, 0x0);
  run_test("-inf", 4, 0xff800000);
  run_test("-iNfInItY", 9, 0xff800000);
}

TEST_F(LlvmLibcStrToFTest, SimpleNaNTests) {
  run_test("NaN", 3, 0x7fc00000);
  run_test("-nAn", 4, 0xffc00000);
}

// These NaNs are of the form `NaN(n-character-sequence)` where the
// n-character-sequence is 0 or more letters or numbers. If there is anything
// other than a letter or a number, then the valid number is just `NaN`. If
// the sequence is valid, then the interpretation of them is implementation
// defined, in this case it's passed to strtoll with an automatic base, and
// the result is put into the mantissa if it takes up the whole width of the
// parentheses.
TEST_F(LlvmLibcStrToFTest, NaNWithParenthesesEmptyTest) {
  run_test("NaN()", 5, 0x7fc00000);
}

TEST_F(LlvmLibcStrToFTest, NaNWithParenthesesValidNumberTests) {
  run_test("NaN(1234)", 9, 0x7fc004d2);
  run_test("NaN(0x1234)", 11, 0x7fc01234);
  run_test("NaN(01234)", 10, 0x7fc0029c);
}

TEST_F(LlvmLibcStrToFTest, NaNWithParenthesesInvalidSequenceTests) {
  run_test("NaN( 1234)", 3, 0x7fc00000);
  run_test("NaN(-1234)", 3, 0x7fc00000);
  run_test("NaN(asd&f)", 3, 0x7fc00000);
  run_test("NaN(123 )", 3, 0x7fc00000);
  run_test("NaN(123+asdf)", 3, 0x7fc00000);
  run_test("NaN(123", 3, 0x7fc00000);
}

TEST_F(LlvmLibcStrToFTest, NaNWithParenthesesValidSequenceInvalidNumberTests) {
  run_test("NaN(1a)", 7, 0x7fc00000);
  run_test("NaN(asdf)", 9, 0x7fc00000);
  run_test("NaN(1A1)", 8, 0x7fc00000);
  run_test("NaN(why_does_this_work)", 23, 0x7fc00000);
  run_test(
      "NaN(1234567890qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM_)",
      68, 0x7fc00000);
}

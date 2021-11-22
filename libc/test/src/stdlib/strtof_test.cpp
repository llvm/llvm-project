//===-- Unittests for strtof ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/stdlib/strtof.h"

#include "utils/UnitTest/Test.h"

#include <errno.h>
#include <limits.h>
#include <stddef.h>

class LlvmLibcStrToFTest : public __llvm_libc::testing::Test {
public:
  void runTest(const char *inputString, const ptrdiff_t expectedStrLen,
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
    char *strEnd = nullptr;

    __llvm_libc::fputil::FPBits<float> expectedFP =
        __llvm_libc::fputil::FPBits<float>(expectedRawData);

    errno = 0;
    float result = __llvm_libc::strtof(inputString, &strEnd);

    __llvm_libc::fputil::FPBits<float> actualFP =
        __llvm_libc::fputil::FPBits<float>(result);

    EXPECT_EQ(strEnd - inputString, expectedStrLen);

    EXPECT_EQ(actualFP.bits, expectedFP.bits);
    EXPECT_EQ(actualFP.getSign(), expectedFP.getSign());
    EXPECT_EQ(actualFP.getExponent(), expectedFP.getExponent());
    EXPECT_EQ(actualFP.getMantissa(), expectedFP.getMantissa());
    EXPECT_EQ(errno, expectedErrno);
  }
};

// This is the set of tests that I have working (verified correct when compared
// to system libc). This is here so I don't break more things when I try to fix
// them.

TEST_F(LlvmLibcStrToFTest, BasicDecimalTests) {
  runTest("1", 1, 0x3f800000);
  runTest("123", 3, 0x42f60000);
  runTest("1234567890", 10, 0x4e932c06u);
  runTest("123456789012345678901", 21, 0x60d629d4);
  runTest("0.1", 3, 0x3dcccccdu);
  runTest(".1", 2, 0x3dcccccdu);
  runTest("-0.123456789", 12, 0xbdfcd6eau);
  runTest("0.11111111111111111111", 22, 0x3de38e39u);
  runTest("0.0000000000000000000000001", 27, 0x15f79688u);
}

TEST_F(LlvmLibcStrToFTest, DecimalOutOfRangeTests) {
  runTest("555E36", 6, 0x7f800000, ERANGE);
  runTest("1e-10000", 8, 0x0, ERANGE);
}

TEST_F(LlvmLibcStrToFTest, DecimalsWithRoundingProblems) {
  runTest("20040229", 8, 0x4b98e512);
  runTest("20040401", 8, 0x4b98e568);
  runTest("9E9", 3, 0x50061c46);
}

TEST_F(LlvmLibcStrToFTest, DecimalSubnormals) {
  runTest("1.4012984643248170709237295832899161312802619418765e-45", 55, 0x1,
          ERANGE);
}

TEST_F(LlvmLibcStrToFTest, DecimalWithLongExponent) {
  runTest("1e2147483648", 12, 0x7f800000, ERANGE);
  runTest("1e2147483646", 12, 0x7f800000, ERANGE);
  runTest("100e2147483646", 14, 0x7f800000, ERANGE);
  runTest("1e-2147483647", 13, 0x0, ERANGE);
  runTest("1e-2147483649", 13, 0x0, ERANGE);
}

TEST_F(LlvmLibcStrToFTest, BasicHexadecimalTests) {
  runTest("0x1", 3, 0x3f800000);
  runTest("0x10", 4, 0x41800000);
  runTest("0x11", 4, 0x41880000);
  runTest("0x0.1234", 8, 0x3d91a000);
}

TEST_F(LlvmLibcStrToFTest, HexadecimalSubnormalTests) {
  runTest("0x0.0000000000000000000000000000000002", 38, 0x4000, ERANGE);

  // This is the largest subnormal number as represented in hex
  runTest("0x0.00000000000000000000000000000003fffff8", 42, 0x7fffff, ERANGE);
}

TEST_F(LlvmLibcStrToFTest, HexadecimalSubnormalRoundingTests) {
  // This is the largest subnormal number that gets rounded down to 0 (as a
  // float)
  runTest("0x0.00000000000000000000000000000000000004", 42, 0x0, ERANGE);

  // This is slightly larger, and thus rounded up
  runTest("0x0.000000000000000000000000000000000000041", 43, 0x00000001,
          ERANGE);

  // These check that we're rounding to even properly
  runTest("0x0.0000000000000000000000000000000000000b", 42, 0x00000001, ERANGE);
  runTest("0x0.0000000000000000000000000000000000000c", 42, 0x00000002, ERANGE);

  // These check that we're rounding to even properly even when the input bits
  // are longer than the bit fields can contain.
  runTest("0x1.000000000000000000000p-150", 30, 0x00000000, ERANGE);
  runTest("0x1.000010000000000001000p-150", 30, 0x00000001, ERANGE);
  runTest("0x1.000100000000000001000p-134", 30, 0x00008001, ERANGE);
  runTest("0x1.FFFFFC000000000001000p-127", 30, 0x007FFFFF, ERANGE);
  runTest("0x1.FFFFFE000000000000000p-127", 30, 0x00800000);
}

TEST_F(LlvmLibcStrToFTest, HexadecimalNormalRoundingTests) {
  // This also checks the round to even behavior by checking three adjacent
  // numbers.
  // This gets rounded down to even
  runTest("0x123456500", 11, 0x4f91a2b2);
  // This doesn't get rounded at all
  runTest("0x123456600", 11, 0x4f91a2b3);
  // This gets rounded up to even
  runTest("0x123456700", 11, 0x4f91a2b4);
  // Correct rounding for long input
  runTest("0x1.000001000000000000000", 25, 0x3f800000);
  runTest("0x1.000001000000000000100", 25, 0x3f800001);
}

TEST_F(LlvmLibcStrToFTest, HexadecimalsWithRoundingProblems) {
  runTest("0xFFFFFFFF", 10, 0x4f800000);
}

TEST_F(LlvmLibcStrToFTest, HexadecimalOutOfRangeTests) {
  runTest("0x123456789123456789123456789123456789", 38, 0x7f800000, ERANGE);
  runTest("-0x123456789123456789123456789123456789", 39, 0xff800000, ERANGE);
  runTest("0x0.00000000000000000000000000000000000001", 42, 0x0, ERANGE);
}

TEST_F(LlvmLibcStrToFTest, InfTests) {
  runTest("INF", 3, 0x7f800000);
  runTest("INFinity", 8, 0x7f800000);
  runTest("infnity", 3, 0x7f800000);
  runTest("infinit", 3, 0x7f800000);
  runTest("infinfinit", 3, 0x7f800000);
  runTest("innf", 0, 0x0);
  runTest("-inf", 4, 0xff800000);
  runTest("-iNfInItY", 9, 0xff800000);
}

TEST_F(LlvmLibcStrToFTest, SimpleNaNTests) {
  runTest("NaN", 3, 0x7fc00000);
  runTest("-nAn", 4, 0xffc00000);
}

// These NaNs are of the form `NaN(n-character-sequence)` where the
// n-character-sequence is 0 or more letters or numbers. If there is anything
// other than a letter or a number, then the valid number is just `NaN`. If
// the sequence is valid, then the interpretation of them is implementation
// defined, in this case it's passed to strtoll with an automatic base, and
// the result is put into the mantissa if it takes up the whole width of the
// parentheses.
TEST_F(LlvmLibcStrToFTest, NaNWithParenthesesEmptyTest) {
  runTest("NaN()", 5, 0x7fc00000);
}

TEST_F(LlvmLibcStrToFTest, NaNWithParenthesesValidNumberTests) {
  runTest("NaN(1234)", 9, 0x7fc004d2);
  runTest("NaN(0x1234)", 11, 0x7fc01234);
  runTest("NaN(01234)", 10, 0x7fc0029c);
}

TEST_F(LlvmLibcStrToFTest, NaNWithParenthesesInvalidSequenceTests) {
  runTest("NaN( 1234)", 3, 0x7fc00000);
  runTest("NaN(-1234)", 3, 0x7fc00000);
  runTest("NaN(asd&f)", 3, 0x7fc00000);
  runTest("NaN(123 )", 3, 0x7fc00000);
  runTest("NaN(123+asdf)", 3, 0x7fc00000);
  runTest("NaN(123", 3, 0x7fc00000);
}

TEST_F(LlvmLibcStrToFTest, NaNWithParenthesesValidSequenceInvalidNumberTests) {
  runTest("NaN(1a)", 7, 0x7fc00000);
  runTest("NaN(asdf)", 9, 0x7fc00000);
  runTest("NaN(1A1)", 8, 0x7fc00000);
}

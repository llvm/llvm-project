//===-- Unittests for wcstof ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wchar/wcstof.h"

#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/RoundingModeUtils.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::fputil::testing::ForceRoundingModeTest;
using LIBC_NAMESPACE::fputil::testing::RoundingMode;

class LlvmLibcWcstofTest : public LIBC_NAMESPACE::testing::ErrnoCheckingTest,
                           ForceRoundingModeTest<RoundingMode::Nearest> {
public:
  void run_test(const wchar_t *inputString, const ptrdiff_t expectedStrLen,
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
    wchar_t *str_end = nullptr;

    LIBC_NAMESPACE::fputil::FPBits<float> expected_fp =
        LIBC_NAMESPACE::fputil::FPBits<float>(expectedRawData);

    float result = LIBC_NAMESPACE::wcstof(inputString, &str_end);

    EXPECT_EQ(str_end - inputString, expectedStrLen);
    EXPECT_FP_EQ(result, expected_fp.get_val());
    ASSERT_ERRNO_EQ(expectedErrno);
  }
};

TEST_F(LlvmLibcWcstofTest, BasicDecimalTests) {
  run_test(L"1", 1, 0x3f800000);
  run_test(L"123", 3, 0x42f60000);
  run_test(L"1234567890", 10, 0x4e932c06u);
  run_test(L"123456789012345678901", 21, 0x60d629d4);
  run_test(L"0.1", 3, 0x3dcccccdu);
  run_test(L".1", 2, 0x3dcccccdu);
  run_test(L"-0.123456789", 12, 0xbdfcd6eau);
  run_test(L"0.11111111111111111111", 22, 0x3de38e39u);
  run_test(L"0.0000000000000000000000001", 27, 0x15f79688u);
}

TEST_F(LlvmLibcWcstofTest, DecimalOutOfRangeTests) {
  run_test(L"555E36", 6, 0x7f800000, ERANGE);
  run_test(L"1e-10000", 8, 0x0, ERANGE);
}

TEST_F(LlvmLibcWcstofTest, DecimalsWithRoundingProblems) {
  run_test(L"20040229", 8, 0x4b98e512);
  run_test(L"20040401", 8, 0x4b98e568);
  run_test(L"9E9", 3, 0x50061c46);
}

TEST_F(LlvmLibcWcstofTest, DecimalSubnormals) {
  run_test(L"1.4012984643248170709237295832899161312802619418765e-45", 55, 0x1,
           ERANGE);
}

TEST_F(LlvmLibcWcstofTest, DecimalWithLongExponent) {
  run_test(L"1e2147483648", 12, 0x7f800000, ERANGE);
  run_test(L"1e2147483646", 12, 0x7f800000, ERANGE);
  run_test(L"100e2147483646", 14, 0x7f800000, ERANGE);
  run_test(L"1e-2147483647", 13, 0x0, ERANGE);
  run_test(L"1e-2147483649", 13, 0x0, ERANGE);
}

TEST_F(LlvmLibcWcstofTest, BasicHexadecimalTests) {
  run_test(L"0x1", 3, 0x3f800000);
  run_test(L"0x10", 4, 0x41800000);
  run_test(L"0x11", 4, 0x41880000);
  run_test(L"0x0.1234", 8, 0x3d91a000);
}

TEST_F(LlvmLibcWcstofTest, HexadecimalSubnormalTests) {
  run_test(L"0x0.0000000000000000000000000000000002", 38, 0x4000, ERANGE);

  // This is the largest subnormal number as represented in hex
  run_test(L"0x0.00000000000000000000000000000003fffff8", 42, 0x7fffff, ERANGE);
}

TEST_F(LlvmLibcWcstofTest, HexadecimalSubnormalRoundingTests) {
  // This is the largest subnormal number that gets rounded down to 0 (as a
  // float)
  run_test(L"0x0.00000000000000000000000000000000000004", 42, 0x0, ERANGE);

  // This is slightly larger, and thus rounded up
  run_test(L"0x0.000000000000000000000000000000000000041", 43, 0x00000001,
           ERANGE);

  // These check that we're rounding to even properly
  run_test(L"0x0.0000000000000000000000000000000000000b", 42, 0x00000001,
           ERANGE);
  run_test(L"0x0.0000000000000000000000000000000000000c", 42, 0x00000002,
           ERANGE);

  // These check that we're rounding to even properly even when the input bits
  // are longer than the bit fields can contain.
  run_test(L"0x1.000000000000000000000p-150", 30, 0x00000000, ERANGE);
  run_test(L"0x1.000010000000000001000p-150", 30, 0x00000001, ERANGE);
  run_test(L"0x1.000100000000000001000p-134", 30, 0x00008001, ERANGE);
  run_test(L"0x1.FFFFFC000000000001000p-127", 30, 0x007FFFFF, ERANGE);
  run_test(L"0x1.FFFFFE000000000000000p-127", 30, 0x00800000);
}

TEST_F(LlvmLibcWcstofTest, HexadecimalNormalRoundingTests) {
  // This also checks the round to even behavior by checking three adjacent
  // numbers.
  // This gets rounded down to even
  run_test(L"0x123456500", 11, 0x4f91a2b2);
  // This doesn't get rounded at all
  run_test(L"0x123456600", 11, 0x4f91a2b3);
  // This gets rounded up to even
  run_test(L"0x123456700", 11, 0x4f91a2b4);
  // Correct rounding for long input
  run_test(L"0x1.000001000000000000000", 25, 0x3f800000);
  run_test(L"0x1.000001000000000000100", 25, 0x3f800001);
}

TEST_F(LlvmLibcWcstofTest, HexadecimalsWithRoundingProblems) {
  run_test(L"0xFFFFFFFF", 10, 0x4f800000);
}

TEST_F(LlvmLibcWcstofTest, HexadecimalOutOfRangeTests) {
  run_test(L"0x123456789123456789123456789123456789", 38, 0x7f800000, ERANGE);
  run_test(L"-0x123456789123456789123456789123456789", 39, 0xff800000, ERANGE);
  run_test(L"0x0.00000000000000000000000000000000000001", 42, 0x0, ERANGE);
}

TEST_F(LlvmLibcWcstofTest, InfTests) {
  run_test(L"INF", 3, 0x7f800000);
  run_test(L"INFinity", 8, 0x7f800000);
  run_test(L"infnity", 3, 0x7f800000);
  run_test(L"infinit", 3, 0x7f800000);
  run_test(L"infinfinit", 3, 0x7f800000);
  run_test(L"innf", 0, 0x0);
  run_test(L"-inf", 4, 0xff800000);
  run_test(L"-iNfInItY", 9, 0xff800000);
}

TEST_F(LlvmLibcWcstofTest, SimpleNaNTests) {
  run_test(L"NaN", 3, 0x7fc00000);
  run_test(L"-nAn", 4, 0xffc00000);
}

// These NaNs are of the form `NaN(n-character-sequence)` where the
// n-character-sequence is 0 or more letters or numbers. If there is anything
// other than a letter or a number, then the valid number is just `NaN`. If
// the sequence is valid, then the interpretation of them is implementation
// defined, in this case it's passed to strtoll with an automatic base, and
// the result is put into the mantissa if it takes up the whole width of the
// parentheses.
TEST_F(LlvmLibcWcstofTest, NaNWithParenthesesEmptyTest) {
  run_test(L"NaN()", 5, 0x7fc00000);
}

TEST_F(LlvmLibcWcstofTest, NaNWithParenthesesValidNumberTests) {
  run_test(L"NaN(1234)", 9, 0x7fc004d2);
  run_test(L"NaN(0x1234)", 11, 0x7fc01234);
  run_test(L"NaN(01234)", 10, 0x7fc0029c);
}

TEST_F(LlvmLibcWcstofTest, NaNWithParenthesesInvalidSequenceTests) {
  run_test(L"NaN( 1234)", 3, 0x7fc00000);
  run_test(L"NaN(-1234)", 3, 0x7fc00000);
  run_test(L"NaN(asd&f)", 3, 0x7fc00000);
  run_test(L"NaN(123 )", 3, 0x7fc00000);
  run_test(L"NaN(123+asdf)", 3, 0x7fc00000);
  run_test(L"NaN(123", 3, 0x7fc00000);
}

TEST_F(LlvmLibcWcstofTest, NaNWithParenthesesValidSequenceInvalidNumberTests) {
  run_test(L"NaN(1a)", 7, 0x7fc00000);
  run_test(L"NaN(asdf)", 9, 0x7fc00000);
  run_test(L"NaN(1A1)", 8, 0x7fc00000);
  run_test(L"NaN(underscores_are_ok)", 23, 0x7fc00000);
  run_test(
      L"NaN(1234567890qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM_)",
      68, 0x7fc00000);
}

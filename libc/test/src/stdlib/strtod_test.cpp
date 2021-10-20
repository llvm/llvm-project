//===-- Unittests for strtod ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/stdlib/strtod.h"

#include "utils/UnitTest/Test.h"

#include <errno.h>
#include <limits.h>
#include <stddef.h>

class LlvmLibcStrToDTest : public __llvm_libc::testing::Test {
public:
  void runTest(const char *inputString, const ptrdiff_t expectedStrLen,
               const uint64_t expectedRawData, const int expectedErrno = 0) {
    // expectedRawData is the expected double result as a uint64_t, organized
    // according to IEEE754:
    //
    // +-- 1 Sign Bit                        +-- 52 Mantissa bits
    // |                                     |
    // |           +-------------------------+------------------------+
    // |           |                                                  |
    // SEEEEEEEEEEEMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
    //  |         |
    //  +----+----+
    //       |
    //       +-- 11 Exponent Bits
    //
    //  This is so that the result can be compared in parts.
    char *strEnd = nullptr;

    __llvm_libc::fputil::FPBits<double> expectedFP =
        __llvm_libc::fputil::FPBits<double>(expectedRawData);

    errno = 0;
    double result = __llvm_libc::strtod(inputString, &strEnd);

    __llvm_libc::fputil::FPBits<double> actualFP =
        __llvm_libc::fputil::FPBits<double>(result);

    EXPECT_EQ(strEnd - inputString, expectedStrLen);

    EXPECT_EQ(actualFP.bits, expectedFP.bits);
    EXPECT_EQ(actualFP.getSign(), expectedFP.getSign());
    EXPECT_EQ(actualFP.getExponent(), expectedFP.getExponent());
    EXPECT_EQ(actualFP.getMantissa(), expectedFP.getMantissa());
    EXPECT_EQ(errno, expectedErrno);
  }
};

TEST_F(LlvmLibcStrToDTest, SimpleTest) {
  runTest("123", 3, uint64_t(0x405ec00000000000));

  // This should fail on Eisel-Lemire, forcing a fallback to simple decimal
  // conversion.
  runTest("12345678901234549760", 20, uint64_t(0x43e56a95319d63d8));

  // Found while looking for difficult test cases here:
  // https://github.com/nigeltao/parse-number-fxx-test-data/blob/main/more-test-cases/golang-org-issue-36657.txt
  runTest("1090544144181609348835077142190", 31, uint64_t(0x462b8779f2474dfb));

  runTest("0x123", 5, uint64_t(0x4072300000000000));
}

// These are tests that have caused problems in the past.
TEST_F(LlvmLibcStrToDTest, SpecificFailures) {
  runTest("3E70000000000000", 16, uint64_t(0x7FF0000000000000), ERANGE);
  runTest("358416272e-33", 13, uint64_t(0x3adbbb2a68c9d0b9));
  runTest("2.16656806400000023841857910156251e9", 36,
          uint64_t(0x41e0246690000001));
  runTest("27949676547093071875", 20, uint64_t(0x43f83e132bc608c9));
}

TEST_F(LlvmLibcStrToDTest, FuzzFailures) {
  runTest("-\xff\xff\xff\xff\xff\xff\xff\x01", 0, uint64_t(0));
  runTest("-.????", 0, uint64_t(0));
  runTest("44444444444444444444444444444444444444444444444444A44444444444444444"
          "44444444444*\x99\xff\xff\xff\xff",
          50, uint64_t(0x4a3e68fdd0e0b2d8));
  runTest("-NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNKNNNNNNNNNNNNNNNNNN?"
          "NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN?",
          0, uint64_t(0));
  runTest("0x.666E40", 9, uint64_t(0x3fd99b9000000000));
}

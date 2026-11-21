//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for strtof128.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/float128.h"
#include "src/__support/macros/properties/types.h"
#include "src/__support/uint128.h"
#include "src/stdlib/strtof128.h"

#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

#include <stddef.h>

#ifndef LIBC_TYPES_HAS_NATIVE_FLOAT128
using float128 = LIBC_NAMESPACE::fputil::Float128;
#endif // LIBC_TYPES_HAS_NATIVE_FLOAT128

class LlvmLibcStrToF128Test
    : public LIBC_NAMESPACE::testing::ErrnoCheckingTest {
public:
  void run_test(const char *inputString, const ptrdiff_t expectedStrLen,
                const UInt128 expectedRawData, const int expectedErrno = 0) {
    // expectedRawData is the expected float128 result as a UInt128,
    // organized according to IEEE754 quadruple precision format:
    //
    // +-- 1 Sign Bit                               +-- 112 Mantissa bits
    // |                                            |
    // |               +----------------------------+--------------------------+
    // |               |                                                       |
    // SEEEEEEEEEEEEEEEMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM...M
    //  |             |
    //  +------+------+
    //         |
    //         +-- 15 Exponent Bits
    char *str_end = nullptr;

    using FPBits = LIBC_NAMESPACE::fputil::FPBits<float128>;
    FPBits expected_fp = FPBits(expectedRawData);

    float128 result = LIBC_NAMESPACE::strtof128(inputString, &str_end);

    FPBits actual_fp = FPBits(result);

    EXPECT_EQ(str_end - inputString, expectedStrLen);

    EXPECT_EQ(actual_fp.uintval(), expected_fp.uintval());
    EXPECT_EQ(actual_fp.is_neg(), expected_fp.is_neg());
    EXPECT_EQ(actual_fp.get_exponent(), expected_fp.get_exponent());
    EXPECT_EQ(actual_fp.get_mantissa(), expected_fp.get_mantissa());
    ASSERT_ERRNO_EQ(expectedErrno);
  }
};

TEST_F(LlvmLibcStrToF128Test, SimpleTest) {
  run_test("123", 3, UInt128(0x4005ec0000000000) << 64);

  // This should fail on Eisel-Lemire, forcing a fallback to simple decimal
  // conversion.
  run_test("12345678901234549760", 20,
           (UInt128(0x403e56a95319d63d) << 64) + UInt128(0x8800000000000000));

  // Found while looking for difficult test cases here:
  // https://github.com/nigeltao/parse-number-fxx-test-data/blob/main/more-test-cases/golang-org-issue-36657.txt
  run_test("1090544144181609348835077142190", 31,
           (UInt128(0x4062b8779f2474df) << 64) + UInt128(0xa804bfd8c6d5c000));

  run_test("0x123", 5, (UInt128(0x4007230000000000) << 64));
}

TEST_F(LlvmLibcStrToF128Test, MaxSizeNumbers) {
  run_test("1.1897314953572317650e4932", 26,
           (UInt128(0x7ffeffffffffffff) << 64) + UInt128(0xfffd57322e3f8675),
           0);
  run_test("1.18973149535723176508e4932", 27,
           (UInt128(0x7ffeffffffffffff) << 64) + UInt128(0xffffd2478338036c),
           0);
}

TEST_F(LlvmLibcStrToF128Test, SubnormalTests) {
  run_test("1e-4950", 7, (UInt128(0x000000000000000000057c9647e1a018)), ERANGE);
  run_test("1.89e-4951", 10, (UInt128(0x0000000000000000000109778a006738)),
           ERANGE);
  run_test("4e-4966", 7, (UInt128(0x00000000000000000000000000000001)), ERANGE);
}

TEST_F(LlvmLibcStrToF128Test, SmallNormalTests) {
  run_test("3.37e-4932", 10,
           (UInt128(0x10099ee12110a) << 64) + UInt128(0xe24b75c0f50dc0c), 0);
}

TEST_F(LlvmLibcStrToF128Test, ComplexHexadecimalTests) {
  run_test("0x1p16383", 9, (UInt128(0x7ffe000000000000) << 64), 0);
  run_test("0x123456789abcdef", 17,
           (UInt128(0x403723456789abcd) << 64) + UInt128(0xef00000000000000));
  run_test("0x123456789abcdef0123456789ABCDEF", 33,
           (UInt128(0x407723456789abcd) << 64) + UInt128(0xef0123456789abce));
}

TEST_F(LlvmLibcStrToF128Test, InfTests) {
  run_test("INF", 3, (UInt128(0x7fff000000000000) << 64));
  run_test("INFinity", 8, (UInt128(0x7fff000000000000) << 64));
  run_test("-inf", 4, (UInt128(0xffff000000000000) << 64));
}

TEST_F(LlvmLibcStrToF128Test, NaNTests) {
  run_test("NaN", 3, (UInt128(0x7fff800000000000) << 64));
  run_test("-nAn", 4, (UInt128(0xffff800000000000) << 64));
  run_test("NaN()", 5, (UInt128(0x7fff800000000000) << 64));
  run_test("NaN(1234)", 9,
           (UInt128(0x7fff800000000000) << 64) + UInt128(0x4d2));
  run_test("NaN(0xffffffffffff)", 19,
           (UInt128(0x7fff800000000000) << 64) + UInt128(0xffffffffffff));
  run_test("NaN(0xfffffffffffff)", 20,
           (UInt128(0x7fff800000000000) << 64) + UInt128(0xfffffffffffff));
  run_test("NaN(0xffffffffffffffff)", 23,
           (UInt128(0x7fff800000000000) << 64) + UInt128(0xffffffffffffffff));
  run_test("NaN( 1234)", 3, (UInt128(0x7fff800000000000) << 64));
}

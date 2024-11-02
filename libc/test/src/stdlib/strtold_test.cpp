//===-- Unittests for strtold ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/stdlib/strtold.h"

#include "utils/UnitTest/Test.h"

#include <errno.h>
#include <limits.h>
#include <stddef.h>

#if defined(LONG_DOUBLE_IS_DOUBLE)
#define SELECT_CONST(val, _, __) val
#elif defined(SPECIAL_X86_LONG_DOUBLE)
#define SELECT_CONST(_, val, __) val
#else
#define SELECT_CONST(_, __, val) val
#endif

class LlvmLibcStrToLDTest : public __llvm_libc::testing::Test {
public:
#if defined(LONG_DOUBLE_IS_DOUBLE)
  void run_test(const char *inputString, const ptrdiff_t expectedStrLen,
                const uint64_t expectedRawData, const int expectedErrno = 0)
#else
  void run_test(const char *inputString, const ptrdiff_t expectedStrLen,
                const __uint128_t expectedRawData, const int expectedErrno = 0)
#endif
  {
    // expectedRawData64 is the expected long double result as a uint64_t,
    // organized according to the IEEE754 double precision format:
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

    // expectedRawData80 is the expected long double result as a __uint128_t,
    // organized according to the x86 extended precision format:
    //
    // +-- 1 Sign Bit
    // |
    // |               +-- 1 Integer part bit (1 unless this is a subnormal)
    // |               |
    // SEEEEEEEEEEEEEEEIMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM...M
    //  |             | |                                                      |
    //  +------+------+ +---------------------------+--------------------------+
    //         |                                    |
    //         +-- 15 Exponent Bits                 +-- 63 Mantissa bits

    // expectedRawData128 is the expected long double result as a __uint128_t,
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

    __llvm_libc::fputil::FPBits<long double> expected_fp =
        __llvm_libc::fputil::FPBits<long double>(expectedRawData);
    const int expected_errno = expectedErrno;

    errno = 0;
    long double result = __llvm_libc::strtold(inputString, &str_end);

    __llvm_libc::fputil::FPBits<long double> actual_fp =
        __llvm_libc::fputil::FPBits<long double>();
    actual_fp = __llvm_libc::fputil::FPBits<long double>(result);

    EXPECT_EQ(str_end - inputString, expectedStrLen);

    EXPECT_EQ(actual_fp.bits, expected_fp.bits);
    EXPECT_EQ(actual_fp.get_sign(), expected_fp.get_sign());
    EXPECT_EQ(actual_fp.get_exponent(), expected_fp.get_exponent());
    EXPECT_EQ(actual_fp.get_mantissa(), expected_fp.get_mantissa());
    EXPECT_EQ(errno, expected_errno);
  }
};

TEST_F(LlvmLibcStrToLDTest, SimpleTest) {
  run_test("123", 3,
           SELECT_CONST(uint64_t(0x405ec00000000000),
                        __uint128_t(0x4005f60000) << 40,
                        __uint128_t(0x4005ec0000000000) << 64));

  // This should fail on Eisel-Lemire, forcing a fallback to simple decimal
  // conversion.
  run_test("12345678901234549760", 20,
           SELECT_CONST(uint64_t(0x43e56a95319d63d8),
                        (__uint128_t(0x403eab54a9) << 40) +
                            __uint128_t(0x8ceb1ec400),
                        (__uint128_t(0x403e56a95319d63d) << 64) +
                            __uint128_t(0x8800000000000000)));

  // Found while looking for difficult test cases here:
  // https://github.com/nigeltao/parse-number-fxx-test-data/blob/main/more-test-cases/golang-org-issue-36657.txt
  run_test("1090544144181609348835077142190", 31,
           SELECT_CONST(uint64_t(0x462b8779f2474dfb),
                        (__uint128_t(0x4062dc3bcf) << 40) +
                            __uint128_t(0x923a6fd402),
                        (__uint128_t(0x4062b8779f2474df) << 64) +
                            __uint128_t(0xa804bfd8c6d5c000)));

  run_test("0x123", 5,
           SELECT_CONST(uint64_t(0x4072300000000000),
                        (__uint128_t(0x4007918000) << 40),
                        (__uint128_t(0x4007230000000000) << 64)));
}

// These are tests that have caused problems for doubles in the past.
TEST_F(LlvmLibcStrToLDTest, Float64SpecificFailures) {
  run_test("3E70000000000000", 16,
           SELECT_CONST(uint64_t(0x7FF0000000000000),
                        (__uint128_t(0x7fff800000) << 40),
                        (__uint128_t(0x7fff000000000000) << 64)),
           ERANGE);
  run_test("358416272e-33", 13,
           SELECT_CONST(uint64_t(0x3adbbb2a68c9d0b9),
                        (__uint128_t(0x3fadddd953) << 40) +
                            __uint128_t(0x464e85c400),
                        (__uint128_t(0x3fadbbb2a68c9d0b) << 64) +
                            __uint128_t(0x8800e7969e1c5fc8)));
  run_test("2.16656806400000023841857910156251e9", 36,
           SELECT_CONST(uint64_t(0x41e0246690000001),
                        (__uint128_t(0x401e812334) << 40) +
                            __uint128_t(0x8000000400),
                        (__uint128_t(0x401e024669000000) << 64) +
                            __uint128_t(0x800000000000018)));
  run_test("27949676547093071875", 20,
           SELECT_CONST(uint64_t(0x43f83e132bc608c9),
                        (__uint128_t(0x403fc1f099) << 40) +
                            __uint128_t(0x5e30464402),
                        (__uint128_t(0x403f83e132bc608c) << 64) +
                            __uint128_t(0x8803000000000000)));
}

TEST_F(LlvmLibcStrToLDTest, MaxSizeNumbers) {
  run_test("1.1897314953572317650e4932", 26,
           SELECT_CONST(uint64_t(0x7FF0000000000000),
                        (__uint128_t(0x7ffeffffff) << 40) +
                            __uint128_t(0xffffffffff),
                        (__uint128_t(0x7ffeffffffffffff) << 64) +
                            __uint128_t(0xfffd57322e3f8675)),
           SELECT_CONST(ERANGE, 0, 0));
  run_test("1.18973149535723176508e4932", 27,
           SELECT_CONST(uint64_t(0x7FF0000000000000),
                        (__uint128_t(0x7fff800000) << 40),
                        (__uint128_t(0x7ffeffffffffffff) << 64) +
                            __uint128_t(0xffffd2478338036c)),
           SELECT_CONST(ERANGE, ERANGE, 0));
}

// These tests check subnormal behavior for 80 bit and 128 bit floats. They will
// be too small for 64 bit floats.
TEST_F(LlvmLibcStrToLDTest, SubnormalTests) {
  run_test("1e-4950", 7,
           SELECT_CONST(uint64_t(0), (__uint128_t(0x00000000000000000003)),
                        (__uint128_t(0x000000000000000000057c9647e1a018))),
           ERANGE);
  run_test("1.89e-4951", 10,
           SELECT_CONST(uint64_t(0), (__uint128_t(0x00000000000000000001)),
                        (__uint128_t(0x0000000000000000000109778a006738))),
           ERANGE);
  run_test("4e-4966", 7,
           SELECT_CONST(uint64_t(0), (__uint128_t(0)),
                        (__uint128_t(0x00000000000000000000000000000001))),
           ERANGE);
}

TEST_F(LlvmLibcStrToLDTest, SmallNormalTests) {
  run_test(
      "3.37e-4932", 10,
      SELECT_CONST(uint64_t(0),
                   (__uint128_t(0x1804cf7) << 40) + __uint128_t(0x908850712),
                   (__uint128_t(0x10099ee12110a) << 64) +
                       __uint128_t(0xe24b75c0f50dc0c)),
      SELECT_CONST(ERANGE, 0, 0));
}

TEST_F(LlvmLibcStrToLDTest, ComplexHexadecimalTests) {
  run_test("0x1p16383", 9,
           SELECT_CONST(0x7ff0000000000000, (__uint128_t(0x7ffe800000) << 40),
                        (__uint128_t(0x7ffe000000000000) << 64)),
           SELECT_CONST(ERANGE, 0, 0));
  run_test("0x123456789abcdef", 17,
           SELECT_CONST(0x43723456789abcdf,
                        (__uint128_t(0x403791a2b3) << 40) +
                            __uint128_t(0xc4d5e6f780),
                        (__uint128_t(0x403723456789abcd) << 64) +
                            __uint128_t(0xef00000000000000)));
  run_test("0x123456789abcdef0123456789ABCDEF", 33,
           SELECT_CONST(0x47723456789abcdf,
                        (__uint128_t(0x407791a2b3) << 40) +
                            __uint128_t(0xc4d5e6f781),
                        (__uint128_t(0x407723456789abcd) << 64) +
                            __uint128_t(0xef0123456789abce)));
}

TEST_F(LlvmLibcStrToLDTest, InfTests) {
  run_test("INF", 3,
           SELECT_CONST(0x7ff0000000000000, (__uint128_t(0x7fff800000) << 40),
                        (__uint128_t(0x7fff000000000000) << 64)));
  run_test("INFinity", 8,
           SELECT_CONST(0x7ff0000000000000, (__uint128_t(0x7fff800000) << 40),
                        (__uint128_t(0x7fff000000000000) << 64)));
  run_test("-inf", 4,
           SELECT_CONST(0xfff0000000000000, (__uint128_t(0xffff800000) << 40),
                        (__uint128_t(0xffff000000000000) << 64)));
}

TEST_F(LlvmLibcStrToLDTest, NaNTests) {
  run_test("NaN", 3,
           SELECT_CONST(0x7ff8000000000000, (__uint128_t(0x7fffc00000) << 40),
                        (__uint128_t(0x7fff800000000000) << 64)));
  run_test("-nAn", 4,
           SELECT_CONST(0xfff8000000000000, (__uint128_t(0xffffc00000) << 40),
                        (__uint128_t(0xffff800000000000) << 64)));
  run_test("NaN()", 5,
           SELECT_CONST(0x7ff8000000000000, (__uint128_t(0x7fffc00000) << 40),
                        (__uint128_t(0x7fff800000000000) << 64)));
  run_test("NaN(1234)", 9,
           SELECT_CONST(0x7ff80000000004d2,
                        (__uint128_t(0x7fffc00000) << 40) + __uint128_t(0x4d2),
                        (__uint128_t(0x7fff800000000000) << 64) +
                            __uint128_t(0x4d2)));
  run_test("NaN(0xffffffffffff)", 19,
           SELECT_CONST(0x7ff8ffffffffffff,
                        (__uint128_t(0x7fffc000ff) << 40) +
                            __uint128_t(0xffffffffff),
                        (__uint128_t(0x7fff800000000000) << 64) +
                            __uint128_t(0xffffffffffff)));
  run_test("NaN(0xfffffffffffff)", 20,
           SELECT_CONST(0x7fffffffffffffff,
                        (__uint128_t(0x7fffc00fff) << 40) +
                            __uint128_t(0xffffffffff),
                        (__uint128_t(0x7fff800000000000) << 64) +
                            __uint128_t(0xfffffffffffff)));
  run_test("NaN(0xffffffffffffffff)", 23,
           SELECT_CONST(0x7fffffffffffffff,
                        (__uint128_t(0x7fffffffff) << 40) +
                            __uint128_t(0xffffffffff),
                        (__uint128_t(0x7fff800000000000) << 64) +
                            __uint128_t(0xffffffffffffffff)));
  run_test("NaN( 1234)", 3,
           SELECT_CONST(0x7ff8000000000000, (__uint128_t(0x7fffc00000) << 40),
                        (__uint128_t(0x7fff800000000000) << 64)));
}

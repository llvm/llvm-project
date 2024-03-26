//===-- Utility class to test canonicalize[f|l] -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_MATH_SMOKE_CANONICALIZETEST_H
#define LLVM_LIBC_TEST_SRC_MATH_SMOKE_CANONICALIZETEST_H

#include "src/__support/FPUtil/FPBits.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

#include "include/llvm-libc-macros/math-macros.h"

#define TEST_SPECIAL(x, y, expected, expected_exception)                       \
  EXPECT_EQ(expected, f(&x, &y));                                              \
  EXPECT_FP_EXCEPTION(expected_exception);                                     \
  LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT)

#define TEST_REGULAR(x, y, expected) TEST_SPECIAL(x, y, expected, 0)

#define LIBC_NAMESPACE __llvm_libc_19_0_0_git

template <typename T>
class CanonicalizeTest : public LIBC_NAMESPACE::testing::Test {

  DECLARE_SPECIAL_CONSTANTS(T)

public:
  typedef int (*CanonicalizeFunc)(T *, const T *);

  void testSpecialNumbers(CanonicalizeFunc f) {
    T cx;

    TEST_SPECIAL(cx, zero, 0, 0);
    EXPECT_FP_EQ(cx, zero);

    TEST_SPECIAL(cx, neg_zero, 0, 0);
    EXPECT_FP_EQ(cx, neg_zero);

    TEST_SPECIAL(cx, inf, 0, 0);
    EXPECT_FP_EQ(cx, inf);

    TEST_SPECIAL(cx, neg_inf, 0, 0);
    EXPECT_FP_EQ(cx, neg_inf);

    TEST_SPECIAL(cx, sNaN, 1, FE_INVALID);
    EXPECT_FP_EQ(cx, aNaN);
  }

  void testX64_80SpecialNumbers(CanonicalizeFunc f) {
    if constexpr (LIBC_NAMESPACE::fputil::get_fp_type<T>() ==
                  LIBC_NAMESPACE::fputil::FPType::X86_Binary80) {
      T cx;
      // Exponent   |       Significand      | Meaning
      //            | Bits 63-62 | Bits 61-0 |
      // All Ones   |     00     |    Zero   | Pseudo Infinity, Value = Infinty

      FPBits test1((UInt128(0x7FFF) << 64) + UInt128(0x0000000000000000));
      const T test1_val = test1.get_val();
      TEST_SPECIAL(cx, test1_val, 0, 0);
      EXPECT_FP_EQ(cx, inf);

      // Exponent   |       Significand      | Meaning
      //            | Bits 63-62 | Bits 61-0 |
      // All Ones   |     00     |  Non-Zero | Pseudo NaN, Value = SNaN

      FPBits test2_1((UInt128(0x7FFF) << 64) + UInt128(0x0000000000000001));
      const T test2_1_val = test2_1.get_val();
      TEST_SPECIAL(cx, test2_1_val, 1, FE_INVALID);
      EXPECT_FP_EQ(cx, aNaN);

      FPBits test2_2((UInt128(0x7FFF) << 64) + UInt128(0x0000004270000001));
      const T test2_2_val = test2_2.get_val();
      TEST_SPECIAL(cx, test2_2_val, 1, FE_INVALID);
      EXPECT_FP_EQ(cx, aNaN);

      FPBits test2_3((UInt128(0x7FFF) << 64) + UInt128(0x0000000008261001));
      const T test2_3_val = test2_3.get_val();
      TEST_SPECIAL(cx, test2_3_val, 1, FE_INVALID);
      EXPECT_FP_EQ(cx, aNaN);

      FPBits test2_4((UInt128(0x7FFF) << 64) + UInt128(0x0000780008261001));
      const T test2_4_val = test2_4.get_val();
      TEST_SPECIAL(cx, test2_4_val, 1, FE_INVALID);
      EXPECT_FP_EQ(cx, aNaN);

      // Exponent   |       Significand      | Meaning
      //            | Bits 63-62 | Bits 61-0 |
      // All Ones   |     01     | Anything  | Pseudo NaN, Value = SNaN

      FPBits test3_1((UInt128(0x7FFF) << 64) + UInt128(0x4000000000000000));
      const T test3_1_val = test3_1.get_val();
      TEST_SPECIAL(cx, test3_1_val, 1, FE_INVALID);
      EXPECT_FP_EQ(cx, aNaN);

      FPBits test3_2((UInt128(0x7FFF) << 64) + UInt128(0x4000004270000001));
      const T test3_2_val = test3_2.get_val();
      TEST_SPECIAL(cx, test3_2_val, 1, FE_INVALID);
      EXPECT_FP_EQ(cx, aNaN);

      FPBits test3_3((UInt128(0x7FFF) << 64) + UInt128(0x4000000008261001));
      const T test3_3_val = test3_3.get_val();
      TEST_SPECIAL(cx, test3_3_val, 1, FE_INVALID);
      EXPECT_FP_EQ(cx, aNaN);

      FPBits test3_4((UInt128(0x7FFF) << 64) + UInt128(0x4007800008261001));
      const T test3_4_val = test3_4.get_val();
      TEST_SPECIAL(cx, test3_4_val, 1, FE_INVALID);
      EXPECT_FP_EQ(cx, aNaN);

      // Exponent   |       Significand      | Meaning
      //            |   Bit 63   | Bits 62-0 |
      // All zeroes |   One      | Anything  | Pseudo Denormal, Value =
      //            |            |           | (−1)**s × m × 2**−16382

      FPBits test4_1((UInt128(0x0000) << 64) + UInt128(0x8000000000000000));
      const T test4_1_val = test4_1.get_val();
      TEST_SPECIAL(cx, test4_1_val, 0, 0);
      EXPECT_FP_EQ(
          cx, FPBits::make_value(test4_1.get_explicit_mantissa(), 0).get_val());

      FPBits test4_2((UInt128(0x0000) << 64) + UInt128(0x8000004270000001));
      const T test4_2_val = test4_2.get_val();
      TEST_SPECIAL(cx, test4_2_val, 0, 0);
      EXPECT_FP_EQ(
          cx, FPBits::make_value(test4_2.get_explicit_mantissa(), 0).get_val());

      FPBits test4_3((UInt128(0x0000) << 64) + UInt128(0x8000000008261001));
      const T test4_3_val = test4_3.get_val();
      TEST_SPECIAL(cx, test4_3_val, 0, 0);
      EXPECT_FP_EQ(
          cx, FPBits::make_value(test4_3.get_explicit_mantissa(), 0).get_val());

      // Exponent   |       Significand      | Meaning
      //            |   Bit 63   | Bits 62-0 |
      // All Other  |   Zero     | Anything  | Unnormal, Value =
      //  Values    |            |           | (−1)**s × m × 2**−16382

      FPBits test5_1(UInt128(0x0000000000000001));
      const T test5_1_val = test5_1.get_val();
      TEST_SPECIAL(cx, test5_1_val, 0, 0);
      EXPECT_FP_EQ(
          cx, FPBits::make_value(test5_1.get_explicit_mantissa(), 0).get_val());

      FPBits test5_2(UInt128(0x0000004270000001));
      const T test5_2_val = test5_2.get_val();
      TEST_SPECIAL(cx, test5_2_val, 0, 0);
      EXPECT_FP_EQ(
          cx, FPBits::make_value(test5_2.get_explicit_mantissa(), 0).get_val());

      FPBits test5_3(UInt128(0x0000000008261001));
      const T test5_3_val = test5_3.get_val();
      TEST_SPECIAL(cx, test5_3_val, 0, 0);
      EXPECT_FP_EQ(
          cx, FPBits::make_value(test5_3.get_explicit_mantissa(), 0).get_val());

      FPBits test5_4(UInt128(0x0000002816000000));
      const T test5_4_val = test5_4.get_val();
      TEST_SPECIAL(cx, test5_4_val, 0, 0);
      EXPECT_FP_EQ(
          cx, FPBits::make_value(test5_4.get_explicit_mantissa(), 0).get_val());

      FPBits test5_5(UInt128(0x0000004270000001));
      const T test5_5_val = test5_5.get_val();
      TEST_SPECIAL(cx, test5_5_val, 0, 0);
      EXPECT_FP_EQ(
          cx, FPBits::make_value(test5_5.get_explicit_mantissa(), 0).get_val());

      FPBits test5_6(UInt128(0x0000000008261001));
      const T test5_6_val = test5_6.get_val();
      TEST_SPECIAL(cx, test5_6_val, 0, 0);
      EXPECT_FP_EQ(
          cx, FPBits::make_value(test5_6.get_explicit_mantissa(), 0).get_val());
    }
  }

  void testRegularNumbers(CanonicalizeFunc f) {
    T cx;
    const T test_var_1 = T(1.0);
    TEST_REGULAR(cx, test_var_1, 0);
    EXPECT_FP_EQ(cx, test_var_1);
    const T test_var_2 = T(-1.0);
    TEST_REGULAR(cx, test_var_2, 0);
    EXPECT_FP_EQ(cx, test_var_2);
    const T test_var_3 = T(10.0);
    TEST_REGULAR(cx, test_var_3, 0);
    EXPECT_FP_EQ(cx, test_var_3);
    const T test_var_4 = T(-10.0);
    TEST_REGULAR(cx, test_var_4, 0);
    EXPECT_FP_EQ(cx, test_var_4);
    const T test_var_5 = T(1234.0);
    TEST_REGULAR(cx, test_var_5, 0);
    EXPECT_FP_EQ(cx, test_var_5);
    const T test_var_6 = T(-1234.0);
    TEST_REGULAR(cx, test_var_6, 0);
    EXPECT_FP_EQ(cx, test_var_6);
  }
};

#define LIST_CANONICALIZE_TESTS(T, func)                                       \
  using LlvmLibcCanonicalizeTest = CanonicalizeTest<T>;                        \
  TEST_F(LlvmLibcCanonicalizeTest, SpecialNumbers) {                           \
    testSpecialNumbers(&func);                                                 \
  }                                                                            \
  TEST_F(LlvmLibcCanonicalizeTest, RegularNubmers) {                           \
    testRegularNumbers(&func);                                                 \
  }

#define X86_80_SPECIAL_CANONICALIZE_TEST(T, func)                              \
  using LlvmLibcCanonicalizeTest = CanonicalizeTest<T>;                        \
  TEST_F(LlvmLibcCanonicalizeTest, X64_80SpecialNumbers) {                     \
    testX64_80SpecialNumbers(&func);                                           \
  }

#endif // LLVM_LIBC_TEST_SRC_MATH_SMOKE_CANONICALIZETEST_H

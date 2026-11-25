//===-- Utility class to test different flavors of iscanonical --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_MATH_SMOKE_ISCANONICALTEST_H
#define LLVM_LIBC_TEST_SRC_MATH_SMOKE_ISCANONICALTEST_H

#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

template <typename T>
class IsCanonicalTest : public LIBC_NAMESPACE::testing::FEnvSafeTest {

  DECLARE_SPECIAL_CONSTANTS(T)

public:
  typedef int (*IsCanonicalFunc)(T);

  void testSpecialNumbers(IsCanonicalFunc func) {
    LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
    EXPECT_EQ(func(aNaN), 1);
    EXPECT_EQ(func(neg_aNaN), 1);
    EXPECT_EQ(func(sNaN), 1);
    EXPECT_FP_EXCEPTION(0);
    EXPECT_EQ(func(neg_sNaN), 1);
    EXPECT_FP_EXCEPTION(0);
    EXPECT_EQ(func(inf), 1);
    EXPECT_EQ(func(neg_inf), 1);
    EXPECT_EQ(func(min_normal), 1);
    EXPECT_EQ(func(max_normal), 1);
    EXPECT_EQ(func(neg_max_normal), 1);
    EXPECT_EQ(func(min_denormal), 1);
    EXPECT_EQ(func(neg_min_denormal), 1);
    EXPECT_EQ(func(max_denormal), 1);
    EXPECT_EQ(func(zero), 1);
    EXPECT_EQ(func(neg_zero), 1);
    EXPECT_FP_EXCEPTION(0);
  }

  void testRoundedNumbers(IsCanonicalFunc func) {
    EXPECT_EQ(func(T(1.0)), 1);
    EXPECT_EQ(func(T(-1.0)), 1);
    EXPECT_EQ(func(T(10.0)), 1);
    EXPECT_EQ(func(T(-10.0)), 1);
    EXPECT_EQ(func(T(1234.0)), 1);
    EXPECT_EQ(func(T(-1234.0)), 1);
  }

  void testX86Binary80SpecialNumbers(IsCanonicalFunc func) {
    if constexpr (LIBC_NAMESPACE::fputil::get_fp_type<T>() ==
                  LIBC_NAMESPACE::fputil::FPType::X86_Binary80) {
      FPBits pseudo_denormal = FPBits::min_normal();
      pseudo_denormal.set_biased_exponent(0);
      FPBits unnormal = FPBits::min_normal();
      unnormal.set_implicit_bit(false);
      FPBits pseudo_inf = FPBits::inf();
      pseudo_inf.set_implicit_bit(false);
      FPBits pseudo_nan = FPBits::quiet_nan();
      pseudo_nan.set_implicit_bit(false);
      FPBits pseudo_snan = FPBits::signaling_nan();
      pseudo_snan.set_implicit_bit(false);
      const FPBits inputs[] = {pseudo_denormal, unnormal, pseudo_inf,
                               pseudo_nan, pseudo_snan};
      for (FPBits bits : inputs) {
        for (int negative = 0; negative < 2; ++negative) {
          bits.set_sign(negative ? Sign::NEG : Sign::POS);
          const T value = bits.get_val();
          LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
          EXPECT_EQ(func(value), 0);
          EXPECT_FP_EXCEPTION(0);
        }
      }
    }
  }
};

#define LIST_ISCANONICAL_TESTS(Name, T, func)                                  \
  using LlvmLibc##Name##Test = IsCanonicalTest<T>;                             \
  TEST_F(LlvmLibc##Name##Test, SpecialNumbers) { testSpecialNumbers(&func); }  \
  TEST_F(LlvmLibc##Name##Test, RoundedNumbers) { testRoundedNumbers(&func); }  \
  TEST_F(LlvmLibc##Name##Test, X86Binary80SpecialNumbers) {                    \
    testX86Binary80SpecialNumbers(&func);                                      \
  }

#endif // LLVM_LIBC_TEST_SRC_MATH_SMOKE_ISCANONICALTEST_H

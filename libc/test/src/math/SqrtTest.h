//===-- Utility class to test sqrt[f|l] -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/Bit.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include "utils/UnitTest/FPMatcher.h"
#include "utils/UnitTest/Test.h"

#include <math.h>

namespace mpfr = __llvm_libc::testing::mpfr;

template <typename T> class SqrtTest : public __llvm_libc::testing::Test {

  DECLARE_SPECIAL_CONSTANTS(T)

  static constexpr UIntType HIDDEN_BIT =
      UIntType(1) << __llvm_libc::fputil::MantissaWidth<T>::VALUE;

public:
  typedef T (*SqrtFunc)(T);

  void test_special_numbers(SqrtFunc func) {
    ASSERT_FP_EQ(aNaN, func(aNaN));
    ASSERT_FP_EQ(inf, func(inf));
    ASSERT_FP_EQ(aNaN, func(neg_inf));
    ASSERT_FP_EQ(0.0, func(0.0));
    ASSERT_FP_EQ(-0.0, func(-0.0));
    ASSERT_FP_EQ(aNaN, func(T(-1.0)));
    ASSERT_FP_EQ(T(1.0), func(T(1.0)));
    ASSERT_FP_EQ(T(2.0), func(T(4.0)));
    ASSERT_FP_EQ(T(3.0), func(T(9.0)));
  }

  void test_denormal_values(SqrtFunc func) {
    for (UIntType mant = 1; mant < HIDDEN_BIT; mant <<= 1) {
      FPBits denormal(T(0.0));
      denormal.set_mantissa(mant);

      test_all_rounding_modes(func, T(denormal));
    }

    constexpr UIntType COUNT = 1'000'001;
    constexpr UIntType STEP = HIDDEN_BIT / COUNT;
    for (UIntType i = 0, v = 0; i <= COUNT; ++i, v += STEP) {
      T x = __llvm_libc::bit_cast<T>(v);
      test_all_rounding_modes(func, x);
    }
  }

  void test_normal_range(SqrtFunc func) {
    constexpr UIntType COUNT = 10'000'001;
    constexpr UIntType STEP = UIntType(-1) / COUNT;
    for (UIntType i = 0, v = 0; i <= COUNT; ++i, v += STEP) {
      T x = __llvm_libc::bit_cast<T>(v);
      if (isnan(x) || (x < 0)) {
        continue;
      }
      test_all_rounding_modes(func, x);
    }
  }

  void test_all_rounding_modes(SqrtFunc func, T x) {
    mpfr::ForceRoundingMode r1(mpfr::RoundingMode::Nearest);
    EXPECT_MPFR_MATCH(mpfr::Operation::Sqrt, x, func(x), 0.5,
                      mpfr::RoundingMode::Nearest);

    mpfr::ForceRoundingMode r2(mpfr::RoundingMode::Upward);
    EXPECT_MPFR_MATCH(mpfr::Operation::Sqrt, x, func(x), 0.5,
                      mpfr::RoundingMode::Upward);

    mpfr::ForceRoundingMode r3(mpfr::RoundingMode::Downward);
    EXPECT_MPFR_MATCH(mpfr::Operation::Sqrt, x, func(x), 0.5,
                      mpfr::RoundingMode::Downward);

    mpfr::ForceRoundingMode r4(mpfr::RoundingMode::TowardZero);
    EXPECT_MPFR_MATCH(mpfr::Operation::Sqrt, x, func(x), 0.5,
                      mpfr::RoundingMode::TowardZero);
  }
};

#define LIST_SQRT_TESTS(T, func)                                               \
  using LlvmLibcSqrtTest = SqrtTest<T>;                                        \
  TEST_F(LlvmLibcSqrtTest, SpecialNumbers) { test_special_numbers(&func); }    \
  TEST_F(LlvmLibcSqrtTest, DenormalValues) { test_denormal_values(&func); }    \
  TEST_F(LlvmLibcSqrtTest, NormalRange) { test_normal_range(&func); }

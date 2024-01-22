//===-- Utility class to test sqrt[f|l] -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/bit.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

#include <math.h>

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

template <typename T> class SqrtTest : public LIBC_NAMESPACE::testing::Test {

  DECLARE_SPECIAL_CONSTANTS(T)

  static constexpr StorageType HIDDEN_BIT =
      StorageType(1) << LIBC_NAMESPACE::fputil::FPBits<T>::FRACTION_LEN;

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
    for (StorageType mant = 1; mant < HIDDEN_BIT; mant <<= 1) {
      FPBits denormal(T(0.0));
      denormal.set_mantissa(mant);
      T x = T(denormal);
      EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Sqrt, x, func(x), 0.5);
    }

    constexpr StorageType COUNT = 200'001;
    constexpr StorageType STEP = HIDDEN_BIT / COUNT;
    for (StorageType i = 0, v = 0; i <= COUNT; ++i, v += STEP) {
      T x = LIBC_NAMESPACE::cpp::bit_cast<T>(v);
      EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Sqrt, x, func(x), 0.5);
    }
  }

  void test_normal_range(SqrtFunc func) {
    constexpr StorageType COUNT = 200'001;
    constexpr StorageType STEP = STORAGE_MAX / COUNT;
    for (StorageType i = 0, v = 0; i <= COUNT; ++i, v += STEP) {
      T x = LIBC_NAMESPACE::cpp::bit_cast<T>(v);
      if (isnan(x) || (x < 0)) {
        continue;
      }
      EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Sqrt, x, func(x), 0.5);
    }
  }
};

#define LIST_SQRT_TESTS(T, func)                                               \
  using LlvmLibcSqrtTest = SqrtTest<T>;                                        \
  TEST_F(LlvmLibcSqrtTest, SpecialNumbers) { test_special_numbers(&func); }    \
  TEST_F(LlvmLibcSqrtTest, DenormalValues) { test_denormal_values(&func); }    \
  TEST_F(LlvmLibcSqrtTest, NormalRange) { test_normal_range(&func); }

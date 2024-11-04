//===-- Utility class to test different flavors of remquo -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_MATH_REMQUOTEST_H
#define LLVM_LIBC_TEST_SRC_MATH_REMQUOTEST_H

#include "src/__support/FPUtil/BasicOperations.h"
#include "src/__support/FPUtil/FPBits.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include <math.h>

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

template <typename T>
class RemQuoTestTemplate : public LIBC_NAMESPACE::testing::Test {
  using FPBits = LIBC_NAMESPACE::fputil::FPBits<T>;
  using StorageType = typename FPBits::StorageType;

  const T zero = T(FPBits::zero());
  const T neg_zero = T(FPBits::neg_zero());
  const T inf = T(FPBits::inf());
  const T neg_inf = T(FPBits::neg_inf());
  const T nan = T(FPBits::build_quiet_nan(1));

public:
  typedef T (*RemQuoFunc)(T, T, int *);

  void testSpecialNumbers(RemQuoFunc func) {
    int quotient;
    T x, y;

    y = T(1.0);
    x = inf;
    EXPECT_FP_EQ(nan, func(x, y, &quotient));
    x = neg_inf;
    EXPECT_FP_EQ(nan, func(x, y, &quotient));

    x = T(1.0);
    y = zero;
    EXPECT_FP_EQ(nan, func(x, y, &quotient));
    y = neg_zero;
    EXPECT_FP_EQ(nan, func(x, y, &quotient));

    y = nan;
    x = T(1.0);
    EXPECT_FP_EQ(nan, func(x, y, &quotient));

    y = T(1.0);
    x = nan;
    EXPECT_FP_EQ(nan, func(x, y, &quotient));

    x = nan;
    y = nan;
    EXPECT_FP_EQ(nan, func(x, y, &quotient));

    x = zero;
    y = T(1.0);
    EXPECT_FP_EQ(func(x, y, &quotient), zero);

    x = neg_zero;
    y = T(1.0);
    EXPECT_FP_EQ(func(x, y, &quotient), neg_zero);

    x = T(1.125);
    y = inf;
    EXPECT_FP_EQ(func(x, y, &quotient), x);
    EXPECT_EQ(quotient, 0);
  }

  void testEqualNumeratorAndDenominator(RemQuoFunc func) {
    T x = T(1.125), y = T(1.125);
    int q;

    // When the remainder is zero, the standard requires it to
    // have the same sign as x.

    EXPECT_FP_EQ(func(x, y, &q), zero);
    EXPECT_EQ(q, 1);

    EXPECT_FP_EQ(func(x, -y, &q), zero);
    EXPECT_EQ(q, -1);

    EXPECT_FP_EQ(func(-x, y, &q), neg_zero);
    EXPECT_EQ(q, -1);

    EXPECT_FP_EQ(func(-x, -y, &q), neg_zero);
    EXPECT_EQ(q, 1);
  }

  void testSubnormalRange(RemQuoFunc func) {
    constexpr StorageType COUNT = 100'001;
    constexpr StorageType STEP =
        (FPBits::MAX_SUBNORMAL - FPBits::MIN_SUBNORMAL) / COUNT;
    for (StorageType v = FPBits::MIN_SUBNORMAL, w = FPBits::MAX_SUBNORMAL;
         v <= FPBits::MAX_SUBNORMAL && w >= FPBits::MIN_SUBNORMAL;
         v += STEP, w -= STEP) {
      T x = T(FPBits(v)), y = T(FPBits(w));
      mpfr::BinaryOutput<T> result;
      mpfr::BinaryInput<T> input{x, y};
      result.f = func(x, y, &result.i);
      ASSERT_MPFR_MATCH(mpfr::Operation::RemQuo, input, result, 0.0);
    }
  }

  void testNormalRange(RemQuoFunc func) {
    constexpr StorageType COUNT = 1'001;
    constexpr StorageType STEP =
        (FPBits::MAX_NORMAL - FPBits::MIN_NORMAL) / COUNT;
    for (StorageType v = FPBits::MIN_NORMAL, w = FPBits::MAX_NORMAL;
         v <= FPBits::MAX_NORMAL && w >= FPBits::MIN_NORMAL;
         v += STEP, w -= STEP) {
      T x = T(FPBits(v)), y = T(FPBits(w));
      mpfr::BinaryOutput<T> result;
      mpfr::BinaryInput<T> input{x, y};
      result.f = func(x, y, &result.i);

      // In normal range on x86 platforms, the long double implicit 1 bit can be
      // zero making the numbers NaN. Hence we test for them separately.
      if (isnan(x) || isnan(y)) {
        ASSERT_FP_EQ(result.f, nan);
        continue;
      }

      ASSERT_MPFR_MATCH(mpfr::Operation::RemQuo, input, result, 0.0);
    }
  }
};

#define LIST_REMQUO_TESTS(T, func)                                             \
  using LlvmLibcRemQuoTest = RemQuoTestTemplate<T>;                            \
  TEST_F(LlvmLibcRemQuoTest, SpecialNumbers) { testSpecialNumbers(&func); }    \
  TEST_F(LlvmLibcRemQuoTest, EqualNumeratorAndDenominator) {                   \
    testEqualNumeratorAndDenominator(&func);                                   \
  }                                                                            \
  TEST_F(LlvmLibcRemQuoTest, SubnormalRange) { testSubnormalRange(&func); }    \
  TEST_F(LlvmLibcRemQuoTest, NormalRange) { testNormalRange(&func); }

#endif // LLVM_LIBC_TEST_SRC_MATH_REMQUOTEST_H

//===-- Utility class to test different flavors of remquo -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_MATH_REMQUOTEST_H
#define LLVM_LIBC_TEST_SRC_MATH_REMQUOTEST_H

#include "src/__support/FPUtil/FPBits.h"
#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::Sign;

template <typename T>
class RemQuoTestTemplate : public LIBC_NAMESPACE::testing::FEnvSafeTest {
  using FPBits = LIBC_NAMESPACE::fputil::FPBits<T>;
  using StorageType = typename FPBits::StorageType;

  const T inf = FPBits::inf(Sign::POS).get_val();
  const T neg_inf = FPBits::inf(Sign::NEG).get_val();
  const T zero = FPBits::zero(Sign::POS).get_val();
  const T neg_zero = FPBits::zero(Sign::NEG).get_val();
  const T nan = FPBits::quiet_nan().get_val();

public:
  typedef T (*RemQuoFunc)(T, T, int *);

  void testSpecialNumbers(RemQuoFunc func) {
    int quotient;
    T x, y;

    y = T(1.0);
    x = inf;
    EXPECT_FP_EQ_WITH_EXCEPTION(nan, func(x, y, &quotient), FE_INVALID);
    EXPECT_MATH_ERRNO(EDOM);
    x = neg_inf;
    EXPECT_FP_EQ_WITH_EXCEPTION(nan, func(x, y, &quotient), FE_INVALID);
    EXPECT_MATH_ERRNO(EDOM);

    x = T(1.0);
    y = zero;
    EXPECT_FP_EQ_WITH_EXCEPTION(nan, func(x, y, &quotient), FE_INVALID);
    EXPECT_MATH_ERRNO(EDOM);
    y = neg_zero;
    EXPECT_FP_EQ_WITH_EXCEPTION(nan, func(x, y, &quotient), FE_INVALID);
    EXPECT_MATH_ERRNO(EDOM);

    y = nan;
    x = T(1.0);
    EXPECT_FP_EQ(nan, func(x, y, &quotient));

    y = T(1.0);
    x = nan;
    EXPECT_FP_EQ(nan, func(x, y, &quotient));

    x = nan;
    y = nan;
    EXPECT_FP_EQ(nan, func(x, y, &quotient));

    const T snan = FPBits::signaling_nan().get_val();
    const T finite = T(1.0);
    const T others[] = {finite, zero, inf, nan, snan};
    T result = zero;
    for (T other : others) {
      EXPECT_FP_EQ_WITH_EXCEPTION(nan, result = func(snan, other, &quotient),
                                  FE_INVALID);
      EXPECT_MATH_ERRNO(0);
      EXPECT_FALSE(FPBits(result).is_signaling_nan());
      EXPECT_FP_EQ_WITH_EXCEPTION(nan, result = func(other, snan, &quotient),
                                  FE_INVALID);
      EXPECT_MATH_ERRNO(0);
      EXPECT_FALSE(FPBits(result).is_signaling_nan());
    }

    // A quiet NaN takes precedence over an otherwise invalid domain.
    EXPECT_FP_EQ_WITH_EXCEPTION(nan, func(nan, zero, &quotient), 0);
    EXPECT_MATH_ERRNO(0);
    EXPECT_FP_EQ_WITH_EXCEPTION(nan, func(inf, nan, &quotient), 0);
    EXPECT_MATH_ERRNO(0);

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
};

#define LIST_REMQUO_TESTS(Name, T, func)                                       \
  using LlvmLibc##Name##Test = RemQuoTestTemplate<T>;                          \
  TEST_F(LlvmLibc##Name##Test, SpecialNumbers) { testSpecialNumbers(&func); }  \
  TEST_F(LlvmLibc##Name##Test, EqualNumeratorAndDenominator) {                 \
    testEqualNumeratorAndDenominator(&func);                                   \
  }

#endif // LLVM_LIBC_TEST_SRC_MATH_REMQUOTEST_H

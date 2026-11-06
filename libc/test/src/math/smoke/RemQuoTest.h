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
  DECLARE_SPECIAL_CONSTANTS(T)

public:
  typedef T (*RemQuoFunc)(T, T, int *);

  void testSpecialNumbers(RemQuoFunc func) {
    int quotient;
    T x, y;

    y = T(1.0);
    x = inf;
    EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, func(x, y, &quotient), FE_INVALID);
    EXPECT_MATH_ERRNO(EDOM);
    x = neg_inf;
    EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, func(x, y, &quotient), FE_INVALID);
    EXPECT_MATH_ERRNO(EDOM);

    x = T(1.0);
    y = zero;
    EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, func(x, y, &quotient), FE_INVALID);
    EXPECT_MATH_ERRNO(EDOM);
    y = neg_zero;
    EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, func(x, y, &quotient), FE_INVALID);
    EXPECT_MATH_ERRNO(EDOM);

    y = aNaN;
    x = T(1.0);
    EXPECT_FP_EQ(aNaN, func(x, y, &quotient));

    y = T(1.0);
    x = aNaN;
    EXPECT_FP_EQ(aNaN, func(x, y, &quotient));

    x = aNaN;
    y = aNaN;
    EXPECT_FP_EQ(aNaN, func(x, y, &quotient));

    const T finite = T(1.0);
    const T others[] = {finite, zero, inf, aNaN, sNaN};
    T result = zero;
    for (T other : others) {
      EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, result = func(sNaN, other, &quotient),
                                  FE_INVALID);
      EXPECT_MATH_ERRNO(0);
      EXPECT_FALSE(FPBits(result).is_signaling_nan());
      EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, result = func(other, sNaN, &quotient),
                                  FE_INVALID);
      EXPECT_MATH_ERRNO(0);
      EXPECT_FALSE(FPBits(result).is_signaling_nan());
    }

    // A quiet NaN takes precedence over an otherwise invalid domain.
    EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, func(aNaN, zero, &quotient), 0);
    EXPECT_MATH_ERRNO(0);
    EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, func(inf, aNaN, &quotient), 0);
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

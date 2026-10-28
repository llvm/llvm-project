//===-- Utility class to test different flavors of remquo -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_MATH_REMQUOTEST_H
#define LLVM_LIBC_TEST_SRC_MATH_REMQUOTEST_H

#include "hdr/fenv_macros.h"
#include "src/__support/FPUtil/FEnvImpl.h"
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

    EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, func(zero, zero, &quotient), FE_INVALID);
    EXPECT_MATH_ERRNO(EDOM);
    EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, func(inf, neg_inf, &quotient),
                                FE_INVALID);
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

    const T zeros[] = {zero, neg_zero};
    const T denominators[] = {finite, T(-1.0), inf, neg_inf};
    for (T numerator : zeros) {
      for (T denominator : denominators) {
        quotient = 123;
        EXPECT_FP_EQ_WITH_EXCEPTION(numerator,
                                    func(numerator, denominator, &quotient), 0);
        EXPECT_MATH_ERRNO(0);
        EXPECT_EQ(quotient, 0);
      }
    }

    const T numerators[] = {T(1.125), T(-1.125), min_denormal,
                            neg_min_denormal};
    const T infinities[] = {inf, neg_inf};
    for (T numerator : numerators) {
      for (T denominator : infinities) {
        testRegular(func, numerator, denominator, numerator, 0);
      }
    }
  }

  void testEqualNumeratorAndDenominator(RemQuoFunc func) {
    const T values[] = {T(1.125), min_denormal, max_denormal, min_normal};

    // When the remainder is zero, the standard requires it to
    // have the same sign as x.

    for (T x : values) {
      testRegular(func, x, x, zero, 1);
      testRegular(func, x, -x, zero, -1);
      testRegular(func, -x, x, neg_zero, -1);
      testRegular(func, -x, -x, neg_zero, 1);
    }
  }

private:
  void testRegular(RemQuoFunc func, T x, T y, T expected,
                   int expected_quotient) {
#ifdef FE_DENORM
    constexpr int DENORM_EXCEPT = FE_DENORM;
#elif defined(FE_DENORMAL)
    constexpr int DENORM_EXCEPT = FE_DENORMAL;
#elif defined(__FE_DENORM)
    constexpr int DENORM_EXCEPT = __FE_DENORM;
#else
    constexpr int DENORM_EXCEPT = 0;
#endif
    int quotient = 123;
    LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
    EXPECT_FP_EQ(expected, func(x, y, &quotient));
    EXPECT_MATH_ERRNO(0);
    // Denormal operands may raise a nonstandard exception on some targets.
    LIBC_NAMESPACE::fputil::clear_except(DENORM_EXCEPT);
    EXPECT_FP_EXCEPTION(0);
    EXPECT_EQ(quotient, expected_quotient);
  }
};

#define LIST_REMQUO_TESTS(Name, T, func)                                       \
  using LlvmLibc##Name##Test = RemQuoTestTemplate<T>;                          \
  TEST_F(LlvmLibc##Name##Test, SpecialNumbers) { testSpecialNumbers(&func); }  \
  TEST_F(LlvmLibc##Name##Test, EqualNumeratorAndDenominator) {                 \
    testEqualNumeratorAndDenominator(&func);                                   \
  }

#endif // LLVM_LIBC_TEST_SRC_MATH_REMQUOTEST_H

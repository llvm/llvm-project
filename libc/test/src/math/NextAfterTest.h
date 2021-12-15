//===-- Utility class to test different flavors of nextafter ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_MATH_NEXTAFTERTEST_H
#define LLVM_LIBC_TEST_SRC_MATH_NEXTAFTERTEST_H

#include "src/__support/CPP/TypeTraits.h"
#include "src/__support/FPUtil/BasicOperations.h"
#include "src/__support/FPUtil/FPBits.h"
#include "utils/UnitTest/FPMatcher.h"
#include "utils/UnitTest/Test.h"
#include <math.h>

template <typename T>
class NextAfterTestTemplate : public __llvm_libc::testing::Test {
  using FPBits = __llvm_libc::fputil::FPBits<T>;
  using MantissaWidth = __llvm_libc::fputil::MantissaWidth<T>;
  using UIntType = typename FPBits::UIntType;

  static constexpr int bitWidthOfType =
      __llvm_libc::fputil::FloatProperties<T>::BIT_WIDTH;

  const T zero = T(FPBits::zero());
  const T neg_zero = T(FPBits::neg_zero());
  const T inf = T(FPBits::inf());
  const T neg_inf = T(FPBits::neg_inf());
  const T nan = T(FPBits::build_nan(1));
  const UIntType MIN_SUBNORMAL = FPBits::MIN_SUBNORMAL;
  const UIntType MAX_SUBNORMAL = FPBits::MAX_SUBNORMAL;
  const UIntType MIN_NORMAL = FPBits::MIN_NORMAL;
  const UIntType MAX_NORMAL = FPBits::MAX_NORMAL;

public:
  typedef T (*NextAfterFunc)(T, T);

  void testNaN(NextAfterFunc func) {
    ASSERT_FP_EQ(func(nan, 0), nan);
    ASSERT_FP_EQ(func(0, nan), nan);
  }

  void testBoundaries(NextAfterFunc func) {
    ASSERT_FP_EQ(func(zero, neg_zero), neg_zero);
    ASSERT_FP_EQ(func(neg_zero, zero), zero);

    // 'from' is zero|neg_zero.
    T x = zero;
    T result = func(x, T(1));
    UIntType expectedBits = 1;
    T expected = *reinterpret_cast<T *>(&expectedBits);
    ASSERT_FP_EQ(result, expected);

    result = func(x, T(-1));
    expectedBits = (UIntType(1) << (bitWidthOfType - 1)) + 1;
    expected = *reinterpret_cast<T *>(&expectedBits);
    ASSERT_FP_EQ(result, expected);

    x = neg_zero;
    result = func(x, 1);
    expectedBits = 1;
    expected = *reinterpret_cast<T *>(&expectedBits);
    ASSERT_FP_EQ(result, expected);

    result = func(x, -1);
    expectedBits = (UIntType(1) << (bitWidthOfType - 1)) + 1;
    expected = *reinterpret_cast<T *>(&expectedBits);
    ASSERT_FP_EQ(result, expected);

    // 'from' is max subnormal value.
    x = *reinterpret_cast<const T *>(&MAX_SUBNORMAL);
    result = func(x, 1);
    expected = *reinterpret_cast<const T *>(&MIN_NORMAL);
    ASSERT_FP_EQ(result, expected);

    result = func(x, 0);
    expectedBits = MAX_SUBNORMAL - 1;
    expected = *reinterpret_cast<T *>(&expectedBits);
    ASSERT_FP_EQ(result, expected);

    x = -x;

    result = func(x, -1);
    expectedBits = (UIntType(1) << (bitWidthOfType - 1)) + MIN_NORMAL;
    expected = *reinterpret_cast<T *>(&expectedBits);
    ASSERT_FP_EQ(result, expected);

    result = func(x, 0);
    expectedBits = (UIntType(1) << (bitWidthOfType - 1)) + MAX_SUBNORMAL - 1;
    expected = *reinterpret_cast<T *>(&expectedBits);
    ASSERT_FP_EQ(result, expected);

    // 'from' is min subnormal value.
    x = *reinterpret_cast<const T *>(&MIN_SUBNORMAL);
    result = func(x, 1);
    expectedBits = MIN_SUBNORMAL + 1;
    expected = *reinterpret_cast<T *>(&expectedBits);
    ASSERT_FP_EQ(result, expected);
    ASSERT_FP_EQ(func(x, 0), 0);

    x = -x;
    result = func(x, -1);
    expectedBits = (UIntType(1) << (bitWidthOfType - 1)) + MIN_SUBNORMAL + 1;
    expected = *reinterpret_cast<T *>(&expectedBits);
    ASSERT_FP_EQ(result, expected);
    ASSERT_FP_EQ(func(x, 0), T(-0.0));

    // 'from' is min normal.
    x = *reinterpret_cast<const T *>(&MIN_NORMAL);
    result = func(x, 0);
    expectedBits = MAX_SUBNORMAL;
    expected = *reinterpret_cast<T *>(&expectedBits);
    ASSERT_FP_EQ(result, expected);

    result = func(x, inf);
    expectedBits = MIN_NORMAL + 1;
    expected = *reinterpret_cast<T *>(&expectedBits);
    ASSERT_FP_EQ(result, expected);

    x = -x;
    result = func(x, 0);
    expectedBits = (UIntType(1) << (bitWidthOfType - 1)) + MAX_SUBNORMAL;
    expected = *reinterpret_cast<T *>(&expectedBits);
    ASSERT_FP_EQ(result, expected);

    result = func(x, -inf);
    expectedBits = (UIntType(1) << (bitWidthOfType - 1)) + MIN_NORMAL + 1;
    expected = *reinterpret_cast<T *>(&expectedBits);
    ASSERT_FP_EQ(result, expected);

    // 'from' is max normal and 'to' is infinity.
    x = *reinterpret_cast<const T *>(&MAX_NORMAL);
    result = func(x, inf);
    ASSERT_FP_EQ(result, inf);

    result = func(-x, -inf);
    ASSERT_FP_EQ(result, -inf);

    // 'from' is infinity.
    x = inf;
    result = func(x, 0);
    expectedBits = MAX_NORMAL;
    expected = *reinterpret_cast<T *>(&expectedBits);
    ASSERT_FP_EQ(result, expected);
    ASSERT_FP_EQ(func(x, inf), inf);

    x = neg_inf;
    result = func(x, 0);
    expectedBits = (UIntType(1) << (bitWidthOfType - 1)) + MAX_NORMAL;
    expected = *reinterpret_cast<T *>(&expectedBits);
    ASSERT_FP_EQ(result, expected);
    ASSERT_FP_EQ(func(x, neg_inf), neg_inf);

    // 'from' is a power of 2.
    x = T(32.0);
    result = func(x, 0);
    FPBits xBits = FPBits(x);
    FPBits resultBits = FPBits(result);
    ASSERT_EQ(resultBits.get_unbiased_exponent(),
              uint16_t(xBits.get_unbiased_exponent() - 1));
    ASSERT_EQ(resultBits.get_mantissa(),
              (UIntType(1) << MantissaWidth::VALUE) - 1);

    result = func(x, T(33.0));
    resultBits = FPBits(result);
    ASSERT_EQ(resultBits.get_unbiased_exponent(),
              xBits.get_unbiased_exponent());
    ASSERT_EQ(resultBits.get_mantissa(), xBits.get_mantissa() + UIntType(1));

    x = -x;

    result = func(x, 0);
    resultBits = FPBits(result);
    ASSERT_EQ(resultBits.get_unbiased_exponent(),
              uint16_t(xBits.get_unbiased_exponent() - 1));
    ASSERT_EQ(resultBits.get_mantissa(),
              (UIntType(1) << MantissaWidth::VALUE) - 1);

    result = func(x, T(-33.0));
    resultBits = FPBits(result);
    ASSERT_EQ(resultBits.get_unbiased_exponent(),
              xBits.get_unbiased_exponent());
    ASSERT_EQ(resultBits.get_mantissa(), xBits.get_mantissa() + UIntType(1));
  }
};

#define LIST_NEXTAFTER_TESTS(T, func)                                          \
  using LlvmLibcNextAfterTest = NextAfterTestTemplate<T>;                      \
  TEST_F(LlvmLibcNextAfterTest, TestNaN) { testNaN(&func); }                   \
  TEST_F(LlvmLibcNextAfterTest, TestBoundaries) { testBoundaries(&func); }

#endif // LLVM_LIBC_TEST_SRC_MATH_NEXTAFTERTEST_H

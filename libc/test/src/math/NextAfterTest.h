//===-- Utility class to test different flavors of nextafter ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_MATH_NEXTAFTERTEST_H
#define LLVM_LIBC_TEST_SRC_MATH_NEXTAFTERTEST_H

#include "utils/CPP/TypeTraits.h"
#include "utils/FPUtil/BasicOperations.h"
#include "utils/FPUtil/FPBits.h"
#include "utils/FPUtil/TestHelpers.h"
#include "utils/UnitTest/Test.h"
#include <math.h>

template <typename T>
class NextAfterTestTemplate : public __llvm_libc::testing::Test {
  using FPBits = __llvm_libc::fputil::FPBits<T>;
  using MantissaWidth = __llvm_libc::fputil::MantissaWidth<T>;
  using UIntType = typename FPBits::UIntType;

#if (defined(__x86_64__) || defined(__i386__))
  static constexpr int bitWidthOfType =
      __llvm_libc::cpp::IsSame<T, long double>::Value ? 80 : (sizeof(T) * 8);
#else
  static constexpr int bitWidthOfType = sizeof(T) * 8;
#endif

  const T zero = FPBits::zero();
  const T negZero = FPBits::negZero();
  const T inf = FPBits::inf();
  const T negInf = FPBits::negInf();
  const T nan = FPBits::buildNaN(1);
  const UIntType minSubnormal = FPBits::minSubnormal;
  const UIntType maxSubnormal = FPBits::maxSubnormal;
  const UIntType minNormal = FPBits::minNormal;
  const UIntType maxNormal = FPBits::maxNormal;

public:
  typedef T (*NextAfterFunc)(T, T);

  void testNaN(NextAfterFunc func) {
    ASSERT_FP_EQ(func(nan, 0), nan);
    ASSERT_FP_EQ(func(0, nan), nan);
  }

  void testBoundaries(NextAfterFunc func) {
    ASSERT_FP_EQ(func(zero, negZero), negZero);
    ASSERT_FP_EQ(func(negZero, zero), zero);

    // 'from' is zero|negZero.
    T x = zero;
    T result = func(x, T(1));
    UIntType expectedBits = 1;
    T expected = *reinterpret_cast<T *>(&expectedBits);
    ASSERT_FP_EQ(result, expected);

    result = func(x, T(-1));
    expectedBits = (UIntType(1) << (bitWidthOfType - 1)) + 1;
    expected = *reinterpret_cast<T *>(&expectedBits);
    ASSERT_FP_EQ(result, expected);

    x = negZero;
    result = func(x, 1);
    expectedBits = 1;
    expected = *reinterpret_cast<T *>(&expectedBits);
    ASSERT_FP_EQ(result, expected);

    result = func(x, -1);
    expectedBits = (UIntType(1) << (bitWidthOfType - 1)) + 1;
    expected = *reinterpret_cast<T *>(&expectedBits);
    ASSERT_FP_EQ(result, expected);

    // 'from' is max subnormal value.
    x = *reinterpret_cast<const T *>(&maxSubnormal);
    result = func(x, 1);
    expected = *reinterpret_cast<const T *>(&minNormal);
    ASSERT_FP_EQ(result, expected);

    result = func(x, 0);
    expectedBits = maxSubnormal - 1;
    expected = *reinterpret_cast<T *>(&expectedBits);
    ASSERT_FP_EQ(result, expected);

    x = -x;

    result = func(x, -1);
    expectedBits = (UIntType(1) << (bitWidthOfType - 1)) + minNormal;
    expected = *reinterpret_cast<T *>(&expectedBits);
    ASSERT_FP_EQ(result, expected);

    result = func(x, 0);
    expectedBits = (UIntType(1) << (bitWidthOfType - 1)) + maxSubnormal - 1;
    expected = *reinterpret_cast<T *>(&expectedBits);
    ASSERT_FP_EQ(result, expected);

    // 'from' is min subnormal value.
    x = *reinterpret_cast<const T *>(&minSubnormal);
    result = func(x, 1);
    expectedBits = minSubnormal + 1;
    expected = *reinterpret_cast<T *>(&expectedBits);
    ASSERT_FP_EQ(result, expected);
    ASSERT_FP_EQ(func(x, 0), 0);

    x = -x;
    result = func(x, -1);
    expectedBits = (UIntType(1) << (bitWidthOfType - 1)) + minSubnormal + 1;
    expected = *reinterpret_cast<T *>(&expectedBits);
    ASSERT_FP_EQ(result, expected);
    ASSERT_FP_EQ(func(x, 0), T(-0.0));

    // 'from' is min normal.
    x = *reinterpret_cast<const T *>(&minNormal);
    result = func(x, 0);
    expectedBits = maxSubnormal;
    expected = *reinterpret_cast<T *>(&expectedBits);
    ASSERT_FP_EQ(result, expected);

    result = func(x, inf);
    expectedBits = minNormal + 1;
    expected = *reinterpret_cast<T *>(&expectedBits);
    ASSERT_FP_EQ(result, expected);

    x = -x;
    result = func(x, 0);
    expectedBits = (UIntType(1) << (bitWidthOfType - 1)) + maxSubnormal;
    expected = *reinterpret_cast<T *>(&expectedBits);
    ASSERT_FP_EQ(result, expected);

    result = func(x, -inf);
    expectedBits = (UIntType(1) << (bitWidthOfType - 1)) + minNormal + 1;
    expected = *reinterpret_cast<T *>(&expectedBits);
    ASSERT_FP_EQ(result, expected);

    // 'from' is max normal and 'to' is infinity.
    x = *reinterpret_cast<const T *>(&maxNormal);
    result = func(x, inf);
    ASSERT_FP_EQ(result, inf);

    result = func(-x, -inf);
    ASSERT_FP_EQ(result, -inf);

    // 'from' is infinity.
    x = inf;
    result = func(x, 0);
    expectedBits = maxNormal;
    expected = *reinterpret_cast<T *>(&expectedBits);
    ASSERT_FP_EQ(result, expected);
    ASSERT_FP_EQ(func(x, inf), inf);

    x = negInf;
    result = func(x, 0);
    expectedBits = (UIntType(1) << (bitWidthOfType - 1)) + maxNormal;
    expected = *reinterpret_cast<T *>(&expectedBits);
    ASSERT_FP_EQ(result, expected);
    ASSERT_FP_EQ(func(x, negInf), negInf);

    // 'from' is a power of 2.
    x = T(32.0);
    result = func(x, 0);
    FPBits xBits = FPBits(x);
    FPBits resultBits = FPBits(result);
    ASSERT_EQ(resultBits.exponent, uint16_t(xBits.exponent - 1));
    ASSERT_EQ(resultBits.mantissa, (UIntType(1) << MantissaWidth::value) - 1);

    result = func(x, T(33.0));
    resultBits = FPBits(result);
    ASSERT_EQ(resultBits.exponent, xBits.exponent);
    ASSERT_EQ(resultBits.mantissa, xBits.mantissa + UIntType(1));

    x = -x;

    result = func(x, 0);
    resultBits = FPBits(result);
    ASSERT_EQ(resultBits.exponent, uint16_t(xBits.exponent - 1));
    ASSERT_EQ(resultBits.mantissa, (UIntType(1) << MantissaWidth::value) - 1);

    result = func(x, T(-33.0));
    resultBits = FPBits(result);
    ASSERT_EQ(resultBits.exponent, xBits.exponent);
    ASSERT_EQ(resultBits.mantissa, xBits.mantissa + UIntType(1));
  }
};

#define LIST_NEXTAFTER_TESTS(T, func)                                          \
  using LlvmLibcNextAfterTest = NextAfterTestTemplate<T>;                      \
  TEST_F(LlvmLibcNextAfterTest, TestNaN) { testNaN(&func); }                   \
  TEST_F(LlvmLibcNextAfterTest, TestBoundaries) { testBoundaries(&func); }

#endif // LLVM_LIBC_TEST_SRC_MATH_NEXTAFTERTEST_H

//===-- Utility class to test different flavors of nextafter ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_MATH_NEXTAFTERTEST_H
#define LLVM_LIBC_TEST_SRC_MATH_NEXTAFTERTEST_H

#include "src/__support/CPP/Bit.h"
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

  static constexpr int BIT_WIDTH_OF_TYPE =
      __llvm_libc::fputil::FloatProperties<T>::BIT_WIDTH;

  const T zero = T(FPBits::zero());
  const T neg_zero = T(FPBits::neg_zero());
  const T inf = T(FPBits::inf());
  const T neg_inf = T(FPBits::neg_inf());
  const T nan = T(FPBits::build_nan(1));
  const UIntType min_subnormal = FPBits::MIN_SUBNORMAL;
  const UIntType max_subnormal = FPBits::MAX_SUBNORMAL;
  const UIntType min_normal = FPBits::MIN_NORMAL;
  const UIntType max_normal = FPBits::MAX_NORMAL;

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
    UIntType expected_bits = 1;
    T expected = __llvm_libc::bit_cast<T>(expected_bits);
    ASSERT_FP_EQ(result, expected);

    result = func(x, T(-1));
    expected_bits = (UIntType(1) << (BIT_WIDTH_OF_TYPE - 1)) + 1;
    expected = __llvm_libc::bit_cast<T>(expected_bits);
    ASSERT_FP_EQ(result, expected);

    x = neg_zero;
    result = func(x, 1);
    expected_bits = 1;
    expected = __llvm_libc::bit_cast<T>(expected_bits);
    ASSERT_FP_EQ(result, expected);

    result = func(x, -1);
    expected_bits = (UIntType(1) << (BIT_WIDTH_OF_TYPE - 1)) + 1;
    expected = __llvm_libc::bit_cast<T>(expected_bits);
    ASSERT_FP_EQ(result, expected);

    // 'from' is max subnormal value.
    x = __llvm_libc::bit_cast<T>(max_subnormal);
    result = func(x, 1);
    expected = __llvm_libc::bit_cast<T>(min_normal);
    ASSERT_FP_EQ(result, expected);

    result = func(x, 0);
    expected_bits = max_subnormal - 1;
    expected = __llvm_libc::bit_cast<T>(expected_bits);
    ASSERT_FP_EQ(result, expected);

    x = -x;

    result = func(x, -1);
    expected_bits = (UIntType(1) << (BIT_WIDTH_OF_TYPE - 1)) + min_normal;
    expected = __llvm_libc::bit_cast<T>(expected_bits);
    ASSERT_FP_EQ(result, expected);

    result = func(x, 0);
    expected_bits =
        (UIntType(1) << (BIT_WIDTH_OF_TYPE - 1)) + max_subnormal - 1;
    expected = __llvm_libc::bit_cast<T>(expected_bits);
    ASSERT_FP_EQ(result, expected);

    // 'from' is min subnormal value.
    x = __llvm_libc::bit_cast<T>(min_subnormal);
    result = func(x, 1);
    expected_bits = min_subnormal + 1;
    expected = __llvm_libc::bit_cast<T>(expected_bits);
    ASSERT_FP_EQ(result, expected);
    ASSERT_FP_EQ(func(x, 0), 0);

    x = -x;
    result = func(x, -1);
    expected_bits =
        (UIntType(1) << (BIT_WIDTH_OF_TYPE - 1)) + min_subnormal + 1;
    expected = __llvm_libc::bit_cast<T>(expected_bits);
    ASSERT_FP_EQ(result, expected);
    ASSERT_FP_EQ(func(x, 0), T(-0.0));

    // 'from' is min normal.
    x = __llvm_libc::bit_cast<T>(min_normal);
    result = func(x, 0);
    expected_bits = max_subnormal;
    expected = __llvm_libc::bit_cast<T>(expected_bits);
    ASSERT_FP_EQ(result, expected);

    result = func(x, inf);
    expected_bits = min_normal + 1;
    expected = __llvm_libc::bit_cast<T>(expected_bits);
    ASSERT_FP_EQ(result, expected);

    x = -x;
    result = func(x, 0);
    expected_bits = (UIntType(1) << (BIT_WIDTH_OF_TYPE - 1)) + max_subnormal;
    expected = __llvm_libc::bit_cast<T>(expected_bits);
    ASSERT_FP_EQ(result, expected);

    result = func(x, -inf);
    expected_bits = (UIntType(1) << (BIT_WIDTH_OF_TYPE - 1)) + min_normal + 1;
    expected = __llvm_libc::bit_cast<T>(expected_bits);
    ASSERT_FP_EQ(result, expected);

    // 'from' is max normal and 'to' is infinity.
    x = __llvm_libc::bit_cast<T>(max_normal);
    result = func(x, inf);
    ASSERT_FP_EQ(result, inf);

    result = func(-x, -inf);
    ASSERT_FP_EQ(result, -inf);

    // 'from' is infinity.
    x = inf;
    result = func(x, 0);
    expected_bits = max_normal;
    expected = __llvm_libc::bit_cast<T>(expected_bits);
    ASSERT_FP_EQ(result, expected);
    ASSERT_FP_EQ(func(x, inf), inf);

    x = neg_inf;
    result = func(x, 0);
    expected_bits = (UIntType(1) << (BIT_WIDTH_OF_TYPE - 1)) + max_normal;
    expected = __llvm_libc::bit_cast<T>(expected_bits);
    ASSERT_FP_EQ(result, expected);
    ASSERT_FP_EQ(func(x, neg_inf), neg_inf);

    // 'from' is a power of 2.
    x = T(32.0);
    result = func(x, 0);
    FPBits x_bits = FPBits(x);
    FPBits result_bits = FPBits(result);
    ASSERT_EQ(result_bits.get_unbiased_exponent(),
              uint16_t(x_bits.get_unbiased_exponent() - 1));
    ASSERT_EQ(result_bits.get_mantissa(),
              (UIntType(1) << MantissaWidth::VALUE) - 1);

    result = func(x, T(33.0));
    result_bits = FPBits(result);
    ASSERT_EQ(result_bits.get_unbiased_exponent(),
              x_bits.get_unbiased_exponent());
    ASSERT_EQ(result_bits.get_mantissa(), x_bits.get_mantissa() + UIntType(1));

    x = -x;

    result = func(x, 0);
    result_bits = FPBits(result);
    ASSERT_EQ(result_bits.get_unbiased_exponent(),
              uint16_t(x_bits.get_unbiased_exponent() - 1));
    ASSERT_EQ(result_bits.get_mantissa(),
              (UIntType(1) << MantissaWidth::VALUE) - 1);

    result = func(x, T(-33.0));
    result_bits = FPBits(result);
    ASSERT_EQ(result_bits.get_unbiased_exponent(),
              x_bits.get_unbiased_exponent());
    ASSERT_EQ(result_bits.get_mantissa(), x_bits.get_mantissa() + UIntType(1));
  }
};

#define LIST_NEXTAFTER_TESTS(T, func)                                          \
  using LlvmLibcNextAfterTest = NextAfterTestTemplate<T>;                      \
  TEST_F(LlvmLibcNextAfterTest, TestNaN) { testNaN(&func); }                   \
  TEST_F(LlvmLibcNextAfterTest, TestBoundaries) { testBoundaries(&func); }

#endif // LLVM_LIBC_TEST_SRC_MATH_NEXTAFTERTEST_H

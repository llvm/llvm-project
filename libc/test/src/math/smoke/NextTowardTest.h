//===-- Utility class to test different flavors of nexttoward ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_MATH_NEXTTOWARDTEST_H
#define LLVM_LIBC_TEST_SRC_MATH_NEXTTOWARDTEST_H

#include "src/__support/CPP/bit.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/FPUtil/BasicOperations.h"
#include "src/__support/FPUtil/FPBits.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include <fenv.h>
#include <math.h>

#define ASSERT_FP_EQ_WITH_EXCEPTION(result, expected, expected_exception)      \
  ASSERT_FP_EQ(result, expected);                                              \
  ASSERT_FP_EXCEPTION(expected_exception);                                     \
  LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT)

#define ASSERT_FP_EQ_WITH_UNDERFLOW(result, expected)                          \
  ASSERT_FP_EQ_WITH_EXCEPTION(result, expected, FE_INEXACT | FE_UNDERFLOW)

#define ASSERT_FP_EQ_WITH_OVERFLOW(result, expected)                           \
  ASSERT_FP_EQ_WITH_EXCEPTION(result, expected, FE_INEXACT | FE_OVERFLOW)

template <typename T>
class NextTowardTestTemplate : public LIBC_NAMESPACE::testing::Test {
  using FPBits = LIBC_NAMESPACE::fputil::FPBits<T>;
  using ToFPBits = LIBC_NAMESPACE::fputil::FPBits<long double>;
  using StorageType = typename FPBits::StorageType;
  using Sign = LIBC_NAMESPACE::fputil::Sign;

  const T inf = T(FPBits::inf(Sign::POS));
  const T neg_inf = T(FPBits::inf(Sign::NEG));
  const T zero = T(FPBits::zero(Sign::POS));
  const T neg_zero = T(FPBits::zero(Sign::NEG));
  const T nan = T(FPBits::build_quiet_nan(1));

  const long double to_zero = ToFPBits::zero();
  const long double to_neg_zero = ToFPBits::zero(Sign::NEG);
  const long double to_nan = ToFPBits::build_quiet_nan(1);

  const StorageType min_subnormal = FPBits::MIN_SUBNORMAL;
  const StorageType max_subnormal = FPBits::MAX_SUBNORMAL;
  const StorageType min_normal = FPBits::MIN_NORMAL;
  const StorageType max_normal = FPBits::MAX_NORMAL;

public:
  typedef T (*NextTowardFunc)(T, long double);

  void testNaN(NextTowardFunc func) {
    ASSERT_FP_EQ(func(nan, 0), nan);
    ASSERT_FP_EQ(func(0, to_nan), nan);
  }

  void testBoundaries(NextTowardFunc func) {
    ASSERT_FP_EQ(func(zero, to_neg_zero), neg_zero);
    ASSERT_FP_EQ(func(neg_zero, to_zero), zero);

    // 'from' is zero|neg_zero.
    T x = zero;
    T result = func(x, 1);
    StorageType expected_bits = 1;
    T expected = LIBC_NAMESPACE::cpp::bit_cast<T>(expected_bits);
    ASSERT_FP_EQ_WITH_UNDERFLOW(result, expected);

    result = func(x, -1);
    expected_bits = FPBits::SIGN_MASK + 1;
    expected = LIBC_NAMESPACE::cpp::bit_cast<T>(expected_bits);
    ASSERT_FP_EQ_WITH_UNDERFLOW(result, expected);

    x = neg_zero;
    result = func(x, 1);
    expected_bits = 1;
    expected = LIBC_NAMESPACE::cpp::bit_cast<T>(expected_bits);
    ASSERT_FP_EQ_WITH_UNDERFLOW(result, expected);

    result = func(x, -1);
    expected_bits = FPBits::SIGN_MASK + 1;
    expected = LIBC_NAMESPACE::cpp::bit_cast<T>(expected_bits);
    ASSERT_FP_EQ_WITH_UNDERFLOW(result, expected);

    // 'from' is max subnormal value.
    x = LIBC_NAMESPACE::cpp::bit_cast<T>(max_subnormal);
    result = func(x, 1);
    expected = LIBC_NAMESPACE::cpp::bit_cast<T>(min_normal);
    ASSERT_FP_EQ(result, expected);

    result = func(x, 0);
    expected_bits = max_subnormal - 1;
    expected = LIBC_NAMESPACE::cpp::bit_cast<T>(expected_bits);
    ASSERT_FP_EQ_WITH_UNDERFLOW(result, expected);

    x = -x;

    result = func(x, -1);
    expected_bits = FPBits::SIGN_MASK + min_normal;
    expected = LIBC_NAMESPACE::cpp::bit_cast<T>(expected_bits);
    ASSERT_FP_EQ(result, expected);

    result = func(x, 0);
    expected_bits = FPBits::SIGN_MASK + max_subnormal - 1;
    expected = LIBC_NAMESPACE::cpp::bit_cast<T>(expected_bits);
    ASSERT_FP_EQ_WITH_UNDERFLOW(result, expected);

    // 'from' is min subnormal value.
    x = LIBC_NAMESPACE::cpp::bit_cast<T>(min_subnormal);
    result = func(x, 1);
    expected_bits = min_subnormal + 1;
    expected = LIBC_NAMESPACE::cpp::bit_cast<T>(expected_bits);
    ASSERT_FP_EQ_WITH_UNDERFLOW(result, expected);
    ASSERT_FP_EQ_WITH_UNDERFLOW(func(x, 0), 0);

    x = -x;
    result = func(x, -1);
    expected_bits = FPBits::SIGN_MASK + min_subnormal + 1;
    expected = LIBC_NAMESPACE::cpp::bit_cast<T>(expected_bits);
    ASSERT_FP_EQ_WITH_UNDERFLOW(result, expected);
    ASSERT_FP_EQ_WITH_UNDERFLOW(func(x, 0), T(-0.0));

    // 'from' is min normal.
    x = LIBC_NAMESPACE::cpp::bit_cast<T>(min_normal);
    result = func(x, 0);
    expected_bits = max_subnormal;
    expected = LIBC_NAMESPACE::cpp::bit_cast<T>(expected_bits);
    ASSERT_FP_EQ_WITH_UNDERFLOW(result, expected);

    result = func(x, inf);
    expected_bits = min_normal + 1;
    expected = LIBC_NAMESPACE::cpp::bit_cast<T>(expected_bits);
    ASSERT_FP_EQ(result, expected);

    x = -x;
    result = func(x, 0);
    expected_bits = FPBits::SIGN_MASK + max_subnormal;
    expected = LIBC_NAMESPACE::cpp::bit_cast<T>(expected_bits);
    ASSERT_FP_EQ_WITH_UNDERFLOW(result, expected);

    result = func(x, -inf);
    expected_bits = FPBits::SIGN_MASK + min_normal + 1;
    expected = LIBC_NAMESPACE::cpp::bit_cast<T>(expected_bits);
    ASSERT_FP_EQ(result, expected);

    // 'from' is max normal
    x = LIBC_NAMESPACE::cpp::bit_cast<T>(max_normal);
    result = func(x, 0);
    expected_bits = max_normal - 1;
    expected = LIBC_NAMESPACE::cpp::bit_cast<T>(expected_bits);
    ASSERT_FP_EQ(result, expected);
    ASSERT_FP_EQ_WITH_OVERFLOW(func(x, inf), inf);

    x = -x;
    result = func(x, 0);
    expected_bits = FPBits::SIGN_MASK + max_normal - 1;
    expected = LIBC_NAMESPACE::cpp::bit_cast<T>(expected_bits);
    ASSERT_FP_EQ(result, expected);
    ASSERT_FP_EQ_WITH_OVERFLOW(func(x, -inf), -inf);

    // 'from' is infinity.
    x = inf;
    result = func(x, 0);
    expected_bits = max_normal;
    expected = LIBC_NAMESPACE::cpp::bit_cast<T>(expected_bits);
    ASSERT_FP_EQ(result, expected);
    ASSERT_FP_EQ(func(x, inf), inf);

    x = neg_inf;
    result = func(x, 0);
    expected_bits = FPBits::SIGN_MASK + max_normal;
    expected = LIBC_NAMESPACE::cpp::bit_cast<T>(expected_bits);
    ASSERT_FP_EQ(result, expected);
    ASSERT_FP_EQ(func(x, neg_inf), neg_inf);

    // 'from' is a power of 2.
    x = T(32.0);
    result = func(x, 0);
    FPBits x_bits = FPBits(x);
    FPBits result_bits = FPBits(result);
    ASSERT_EQ(result_bits.get_biased_exponent(),
              uint16_t(x_bits.get_biased_exponent() - 1));
    ASSERT_EQ(result_bits.get_mantissa(), FPBits::FRACTION_MASK);

    result = func(x, 33.0);
    result_bits = FPBits(result);
    ASSERT_EQ(result_bits.get_biased_exponent(), x_bits.get_biased_exponent());
    ASSERT_EQ(result_bits.get_mantissa(),
              x_bits.get_mantissa() + StorageType(1));

    x = -x;

    result = func(x, 0);
    result_bits = FPBits(result);
    ASSERT_EQ(result_bits.get_biased_exponent(),
              uint16_t(x_bits.get_biased_exponent() - 1));
    ASSERT_EQ(result_bits.get_mantissa(), FPBits::FRACTION_MASK);

    result = func(x, -33.0);
    result_bits = FPBits(result);
    ASSERT_EQ(result_bits.get_biased_exponent(), x_bits.get_biased_exponent());
    ASSERT_EQ(result_bits.get_mantissa(),
              x_bits.get_mantissa() + StorageType(1));
  }
};

#define LIST_NEXTTOWARD_TESTS(T, func)                                         \
  using LlvmLibcNextTowardTest = NextTowardTestTemplate<T>;                    \
  TEST_F(LlvmLibcNextTowardTest, TestNaN) { testNaN(&func); }                  \
  TEST_F(LlvmLibcNextTowardTest, TestBoundaries) { testBoundaries(&func); }

#endif // LLVM_LIBC_TEST_SRC_MATH_NEXTTOWARDTEST_H

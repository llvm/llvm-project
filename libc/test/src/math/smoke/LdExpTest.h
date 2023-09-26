//===-- Utility class to test different flavors of ldexp --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_MATH_LDEXPTEST_H
#define LLVM_LIBC_TEST_SRC_MATH_LDEXPTEST_H

#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/NormalFloat.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>

template <typename T>
class LdExpTestTemplate : public LIBC_NAMESPACE::testing::Test {
  using FPBits = LIBC_NAMESPACE::fputil::FPBits<T>;
  using NormalFloat = LIBC_NAMESPACE::fputil::NormalFloat<T>;
  using UIntType = typename FPBits::UIntType;
  static constexpr UIntType MANTISSA_WIDTH =
      LIBC_NAMESPACE::fputil::MantissaWidth<T>::VALUE;
  // A normalized mantissa to be used with tests.
  static constexpr UIntType MANTISSA = NormalFloat::ONE + 0x1234;

  const T zero = T(LIBC_NAMESPACE::fputil::FPBits<T>::zero());
  const T neg_zero = T(LIBC_NAMESPACE::fputil::FPBits<T>::neg_zero());
  const T inf = T(LIBC_NAMESPACE::fputil::FPBits<T>::inf());
  const T neg_inf = T(LIBC_NAMESPACE::fputil::FPBits<T>::neg_inf());
  const T nan = T(LIBC_NAMESPACE::fputil::FPBits<T>::build_quiet_nan(1));

public:
  typedef T (*LdExpFunc)(T, int);

  void testSpecialNumbers(LdExpFunc func) {
    int exp_array[5] = {-INT_MAX - 1, -10, 0, 10, INT_MAX};
    for (int exp : exp_array) {
      ASSERT_FP_EQ(zero, func(zero, exp));
      ASSERT_FP_EQ(neg_zero, func(neg_zero, exp));
      ASSERT_FP_EQ(inf, func(inf, exp));
      ASSERT_FP_EQ(neg_inf, func(neg_inf, exp));
      ASSERT_FP_EQ(nan, func(nan, exp));
    }
  }

  void testPowersOfTwo(LdExpFunc func) {
    int32_t exp_array[5] = {1, 2, 3, 4, 5};
    int32_t val_array[6] = {1, 2, 4, 8, 16, 32};
    for (int32_t exp : exp_array) {
      for (int32_t val : val_array) {
        ASSERT_FP_EQ(T(val << exp), func(T(val), exp));
        ASSERT_FP_EQ(T(-1 * (val << exp)), func(T(-val), exp));
      }
    }
  }

  void testOverflow(LdExpFunc func) {
    NormalFloat x(FPBits::MAX_EXPONENT - 10, NormalFloat::ONE + 0xF00BA, 0);
    for (int32_t exp = 10; exp < 100; ++exp) {
      ASSERT_FP_EQ(inf, func(T(x), exp));
      ASSERT_FP_EQ(neg_inf, func(-T(x), exp));
    }
  }

  void testUnderflowToZeroOnNormal(LdExpFunc func) {
    // In this test, we pass a normal nubmer to func and expect zero
    // to be returned due to underflow.
    int32_t base_exponent = FPBits::EXPONENT_BIAS + int32_t(MANTISSA_WIDTH);
    int32_t exp_array[] = {base_exponent + 5, base_exponent + 4,
                           base_exponent + 3, base_exponent + 2,
                           base_exponent + 1};
    T x = NormalFloat(0, MANTISSA, 0);
    for (int32_t exp : exp_array) {
      ASSERT_FP_EQ(func(x, -exp), x > 0 ? zero : neg_zero);
    }
  }

  void testUnderflowToZeroOnSubnormal(LdExpFunc func) {
    // In this test, we pass a normal nubmer to func and expect zero
    // to be returned due to underflow.
    int32_t base_exponent = FPBits::EXPONENT_BIAS + int32_t(MANTISSA_WIDTH);
    int32_t exp_array[] = {base_exponent + 5, base_exponent + 4,
                           base_exponent + 3, base_exponent + 2,
                           base_exponent + 1};
    T x = NormalFloat(-FPBits::EXPONENT_BIAS, MANTISSA, 0);
    for (int32_t exp : exp_array) {
      ASSERT_FP_EQ(func(x, -exp), x > 0 ? zero : neg_zero);
    }
  }

  void testNormalOperation(LdExpFunc func) {
    T val_array[] = {
        // Normal numbers
        NormalFloat(100, MANTISSA, 0), NormalFloat(-100, MANTISSA, 0),
        NormalFloat(100, MANTISSA, 1), NormalFloat(-100, MANTISSA, 1),
        // Subnormal numbers
        NormalFloat(-FPBits::EXPONENT_BIAS, MANTISSA, 0),
        NormalFloat(-FPBits::EXPONENT_BIAS, MANTISSA, 1)};
    for (int32_t exp = 0; exp <= static_cast<int32_t>(MANTISSA_WIDTH); ++exp) {
      for (T x : val_array) {
        // We compare the result of ldexp with the result
        // of the native multiplication/division instruction.

        // We need to use a NormalFloat here (instead of 1 << exp), because
        // there are 32 bit systems that don't support 128bit long ints but
        // support long doubles. This test can do 1 << 64, which would fail
        // in these systems.
        NormalFloat two_to_exp = NormalFloat(static_cast<T>(1.L));
        two_to_exp = two_to_exp.mul2(exp);

        ASSERT_FP_EQ(func(x, exp), x * two_to_exp);
        ASSERT_FP_EQ(func(x, -exp), x / two_to_exp);
      }
    }

    // Normal which trigger mantissa overflow.
    T x = NormalFloat(-FPBits::EXPONENT_BIAS + 1,
                      UIntType(2) * NormalFloat::ONE - UIntType(1), 0);
    ASSERT_FP_EQ(func(x, -1), x / 2);
    ASSERT_FP_EQ(func(-x, -1), -x / 2);

    // Start with a normal number high exponent but pass a very low number for
    // exp. The result should be a subnormal number.
    x = NormalFloat(FPBits::EXPONENT_BIAS, NormalFloat::ONE, 0);
    int exp = -FPBits::MAX_EXPONENT - 5;
    T result = func(x, exp);
    FPBits result_bits(result);
    ASSERT_FALSE(result_bits.is_zero());
    // Verify that the result is indeed subnormal.
    ASSERT_EQ(result_bits.get_unbiased_exponent(), uint16_t(0));
    // But if the exp is so less that normalization leads to zero, then
    // the result should be zero.
    result = func(x, -FPBits::MAX_EXPONENT - int(MANTISSA_WIDTH) - 5);
    ASSERT_TRUE(FPBits(result).is_zero());

    // Start with a subnormal number but pass a very high number for exponent.
    // The result should not be infinity.
    x = NormalFloat(-FPBits::EXPONENT_BIAS + 1, NormalFloat::ONE >> 10, 0);
    exp = FPBits::MAX_EXPONENT + 5;
    ASSERT_FALSE(FPBits(func(x, exp)).is_inf());
    // But if the exp is large enough to oversome than the normalization shift,
    // then it should result in infinity.
    exp = FPBits::MAX_EXPONENT + 15;
    ASSERT_FP_EQ(func(x, exp), inf);
  }
};

#define LIST_LDEXP_TESTS(T, func)                                              \
  using LlvmLibcLdExpTest = LdExpTestTemplate<T>;                              \
  TEST_F(LlvmLibcLdExpTest, SpecialNumbers) { testSpecialNumbers(&func); }     \
  TEST_F(LlvmLibcLdExpTest, PowersOfTwo) { testPowersOfTwo(&func); }           \
  TEST_F(LlvmLibcLdExpTest, OverFlow) { testOverflow(&func); }                 \
  TEST_F(LlvmLibcLdExpTest, UnderflowToZeroOnNormal) {                         \
    testUnderflowToZeroOnNormal(&func);                                        \
  }                                                                            \
  TEST_F(LlvmLibcLdExpTest, UnderflowToZeroOnSubnormal) {                      \
    testUnderflowToZeroOnSubnormal(&func);                                     \
  }                                                                            \
  TEST_F(LlvmLibcLdExpTest, NormalOperation) { testNormalOperation(&func); }

#endif // LLVM_LIBC_TEST_SRC_MATH_LDEXPTEST_H

//===-- Utility class to test different flavors of nearbyint ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_MATH_NEARBYINTTEST_H
#define LLVM_LIBC_TEST_SRC_MATH_NEARBYINTTEST_H

#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include <fenv.h>

static constexpr int ROUNDING_MODES[4] = {FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO,
                                          FE_TONEAREST};

template <typename T>
class NearbyIntTestTemplate : public LIBC_NAMESPACE::testing::Test {

  DECLARE_SPECIAL_CONSTANTS(T)

public:
  typedef T (*NearbyIntFunc)(T);

  void testNaN(NearbyIntFunc func) {
    ASSERT_FP_EQ(func(aNaN), aNaN);
  }

  void testInfinities(NearbyIntFunc func) {
    ASSERT_FP_EQ(func(inf), inf);
    ASSERT_FP_EQ(func(neg_inf), neg_inf);
  }

  void testZeroes(NearbyIntFunc func) {
    ASSERT_FP_EQ(func(zero), zero);
    ASSERT_FP_EQ(func(neg_zero), neg_zero);
  }

  void testIntegers(NearbyIntFunc func) {
    for (int mode : ROUNDING_MODES) {
      LIBC_NAMESPACE::fputil::set_round(mode);

      ASSERT_FP_EQ(func(T(1.0)), T(1.0));
      ASSERT_FP_EQ(func(T(-1.0)), T(-1.0));

      ASSERT_FP_EQ(func(T(1234.0)), T(1234.0));
      ASSERT_FP_EQ(func(T(-1234.0)), T(-1234.0));

      ASSERT_FP_EQ(func(T(10.0)), T(10.0));
      ASSERT_FP_EQ(func(T(-10.0)), T(-10.0));

      FPBits ints_start(T(0));
      ints_start.set_biased_exponent(FPBits::SIG_LEN + FPBits::EXP_BIAS);
      T expected = ints_start.get_val();
      ASSERT_FP_EQ(func(expected), expected);
    }
  }

  void testSubnormalToNearest(NearbyIntFunc func) {
    ASSERT_FP_EQ(func(min_denormal), zero);
    ASSERT_FP_EQ(func(-min_denormal), neg_zero);
  }

  void testSubnormalToZero(NearbyIntFunc func) {
    LIBC_NAMESPACE::fputil::set_round(FE_TOWARDZERO);
    ASSERT_FP_EQ(func(min_denormal), zero);
    ASSERT_FP_EQ(func(-min_denormal), neg_zero);
  }

  void testSubnormalToPosInf(NearbyIntFunc func) {
    LIBC_NAMESPACE::fputil::set_round(FE_UPWARD);
    ASSERT_FP_EQ(func(min_denormal), FPBits::one().get_val());
    ASSERT_FP_EQ(func(-min_denormal), neg_zero);
  }

  void testSubnormalToNegInf(NearbyIntFunc func) {
    LIBC_NAMESPACE::fputil::set_round(FE_DOWNWARD);
    FPBits negative_one = FPBits::one(Sign::NEG);
    ASSERT_FP_EQ(func(min_denormal), zero);
    ASSERT_FP_EQ(func(-min_denormal), negative_one.get_val());
  }
};

#define LIST_NEARBYINT_TESTS(T, func)                                          \
  using LlvmLibcNearbyIntTest = NearbyIntTestTemplate<T>;                      \
  TEST_F(LlvmLibcNearbyIntTest, TestNaN) { testNaN(&func); }                   \
  TEST_F(LlvmLibcNearbyIntTest, TestInfinities) { testInfinities(&func); }     \
  TEST_F(LlvmLibcNearbyIntTest, TestZeroes) { testZeroes(&func); }             \
  TEST_F(LlvmLibcNearbyIntTest, TestIntegers) { testIntegers(&func); }         \
  TEST_F(LlvmLibcNearbyIntTest, TestSubnormalToNearest) {                      \
    testSubnormalToNearest(&func);                                             \
  }                                                                            \
  TEST_F(LlvmLibcNearbyIntTest, TestSubnormalToZero) {                         \
    testSubnormalToZero(&func);                                                \
  }                                                                            \
  TEST_F(LlvmLibcNearbyIntTest, TestSubnormalToPosInf) {                       \
    testSubnormalToPosInf(&func);                                              \
  }                                                                            \
  TEST_F(LlvmLibcNearbyIntTest, TestSubnormalToNegInf) {                       \
    testSubnormalToNegInf(&func);                                              \
  }

#endif // LLVM_LIBC_TEST_SRC_MATH_NEARBYINTTEST_H

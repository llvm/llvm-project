//===-- Utility class to test fminimum_num[f|l] -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_MATH_SMOKE_FMINIMUMNUMTEST_H
#define LLVM_LIBC_TEST_SRC_MATH_SMOKE_FMINIMUMNUMTEST_H

#include "src/__support/FPUtil/FPBits.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

template <typename T>
class FMinimumNumTest : public LIBC_NAMESPACE::testing::Test {

  DECLARE_SPECIAL_CONSTANTS(T)

public:
  typedef T (*FMinimumNumFunc)(T, T);

  void testNaN(FMinimumNumFunc func) {
    EXPECT_FP_EQ(inf, func(FPBits::quiet_nan().get_val(), inf));
    EXPECT_FP_EQ(inf, func(FPBits::signaling_nan().get_val(), inf));
    EXPECT_FP_EQ(neg_inf, func(neg_inf, FPBits::quiet_nan().get_val()));
    EXPECT_FP_EQ(neg_inf, func(neg_inf, FPBits::signaling_nan().get_val()));
    EXPECT_FP_EQ(FPBits::quiet_nan().get_val(), func(aNaN, aNaN));
    EXPECT_FP_EQ(0.0, func(FPBits::quiet_nan().get_val(), 0.0));
    EXPECT_FP_EQ(-0.0, func(-0.0, FPBits::quiet_nan().get_val()));
    EXPECT_FP_EQ(0.0, func(FPBits::signaling_nan().get_val(), 0.0));
    EXPECT_FP_EQ(-0.0, func(-0.0, FPBits::signaling_nan().get_val()));
    EXPECT_FP_EQ(T(-1.2345), func(FPBits::quiet_nan().get_val(), T(-1.2345)));
    EXPECT_FP_EQ(T(1.2345), func(T(1.2345), FPBits::quiet_nan().get_val()));
    EXPECT_FP_EQ(T(-1.2345),
                 func(FPBits::signaling_nan().get_val(), T(-1.2345)));
    EXPECT_FP_EQ(T(1.2345), func(T(1.2345), FPBits::signaling_nan().get_val()));
    EXPECT_FP_IS_NAN_WITH_EXCEPTION(
        func(aNaN, FPBits::signaling_nan().get_val()), FE_INVALID);
    EXPECT_FP_IS_NAN_WITH_EXCEPTION(
        func(FPBits::signaling_nan().get_val(), aNaN), FE_INVALID);
  }

  void testInfArg(FMinimumNumFunc func) {
    EXPECT_FP_EQ(neg_inf, func(neg_inf, inf));
    EXPECT_FP_EQ(0.0, func(inf, 0.0));
    EXPECT_FP_EQ(-0.0, func(-0.0, inf));
    EXPECT_FP_EQ(T(1.2345), func(inf, T(1.2345)));
    EXPECT_FP_EQ(T(-1.2345), func(T(-1.2345), inf));
  }

  void testNegInfArg(FMinimumNumFunc func) {
    EXPECT_FP_EQ(neg_inf, func(inf, neg_inf));
    EXPECT_FP_EQ(neg_inf, func(neg_inf, 0.0));
    EXPECT_FP_EQ(neg_inf, func(-0.0, neg_inf));
    EXPECT_FP_EQ(neg_inf, func(neg_inf, T(-1.2345)));
    EXPECT_FP_EQ(neg_inf, func(T(1.2345), neg_inf));
  }

  void testBothZero(FMinimumNumFunc func) {
    EXPECT_FP_EQ(0.0, func(0.0, 0.0));
    EXPECT_FP_EQ(-0.0, func(-0.0, 0.0));
    EXPECT_FP_EQ(-0.0, func(0.0, -0.0));
    EXPECT_FP_EQ(-0.0, func(-0.0, -0.0));
  }

  void testRange(FMinimumNumFunc func) {
    constexpr StorageType COUNT = 100'001;
    constexpr StorageType STEP = STORAGE_MAX / COUNT;
    for (StorageType i = 0, v = 0, w = STORAGE_MAX; i <= COUNT;
         ++i, v += STEP, w -= STEP) {
      FPBits xbits(v), ybits(w);
      if (xbits.is_inf_or_nan())
        continue;
      if (ybits.is_inf_or_nan())
        continue;
      T x = xbits.get_val();
      T y = ybits.get_val();
      if ((x == 0) && (y == 0))
        continue;

      if (x > y) {
        EXPECT_FP_EQ(y, func(x, y));
      } else {
        EXPECT_FP_EQ(x, func(x, y));
      }
    }
  }
};

#define LIST_FMINIMUM_NUM_TESTS(T, func)                                       \
  using LlvmLibcFMinimumNumTest = FMinimumNumTest<T>;                          \
  TEST_F(LlvmLibcFMinimumNumTest, NaN) { testNaN(&func); }                     \
  TEST_F(LlvmLibcFMinimumNumTest, InfArg) { testInfArg(&func); }               \
  TEST_F(LlvmLibcFMinimumNumTest, NegInfArg) { testNegInfArg(&func); }         \
  TEST_F(LlvmLibcFMinimumNumTest, BothZero) { testBothZero(&func); }           \
  TEST_F(LlvmLibcFMinimumNumTest, Range) { testRange(&func); }

#endif // LLVM_LIBC_TEST_SRC_MATH_SMOKE_FMINIMUMNUMTEST_H

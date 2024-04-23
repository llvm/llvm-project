//===-- Utility class to test fminimum_mag_num[f|l] -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_MATH_SMOKE_FMINIMUMMAG_NUMTEST_H
#define LLVM_LIBC_TEST_SRC_MATH_SMOKE_FMINIMUMMAG_NUMTEST_H

#include "src/__support/FPUtil/BasicOperations.h"
#include "src/__support/FPUtil/FPBits.h"
#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

template <typename T>
class FMinimumMagNumTest : public LIBC_NAMESPACE::testing::FEnvSafeTest {

  DECLARE_SPECIAL_CONSTANTS(T)

public:
  typedef T (*FMinimumMagNumFunc)(T, T);

  void testNaN(FMinimumMagNumFunc func) {
    EXPECT_FP_EQ(inf, func(aNaN, inf));
    EXPECT_FP_EQ_WITH_EXCEPTION(inf, func(sNaN, inf), FE_INVALID);
    EXPECT_FP_EQ(neg_inf, func(neg_inf, aNaN));
    EXPECT_FP_EQ_WITH_EXCEPTION(neg_inf, func(neg_inf, sNaN), FE_INVALID);
    EXPECT_EQ(FPBits(aNaN).uintval(), FPBits(func(aNaN, aNaN)).uintval());
    EXPECT_FP_EQ(0.0, func(aNaN, 0.0));
    EXPECT_FP_EQ(-0.0, func(-0.0, aNaN));
    EXPECT_FP_EQ_WITH_EXCEPTION(0.0, func(sNaN, 0.0), FE_INVALID);
    EXPECT_FP_EQ_WITH_EXCEPTION(-0.0, func(-0.0, sNaN), FE_INVALID);
    EXPECT_FP_EQ(T(-1.2345), func(aNaN, T(-1.2345)));
    EXPECT_FP_EQ(T(1.2345), func(T(1.2345), aNaN));
    EXPECT_FP_EQ_WITH_EXCEPTION(T(-1.2345), func(sNaN, T(-1.2345)), FE_INVALID);
    EXPECT_FP_EQ_WITH_EXCEPTION(T(1.2345), func(T(1.2345), sNaN), FE_INVALID);
    EXPECT_FP_IS_NAN_WITH_EXCEPTION(func(aNaN, sNaN), FE_INVALID);
    EXPECT_FP_IS_NAN_WITH_EXCEPTION(func(sNaN, aNaN), FE_INVALID);
    EXPECT_EQ(FPBits(aNaN).uintval(), FPBits(func(aNaN, sNaN)).uintval());
    EXPECT_EQ(FPBits(aNaN).uintval(), FPBits(func(sNaN, aNaN)).uintval());
    EXPECT_EQ(FPBits(aNaN).uintval(), FPBits(func(sNaN, sNaN)).uintval());
  }

  void testInfArg(FMinimumMagNumFunc func) {
    EXPECT_FP_EQ(neg_inf, func(neg_inf, inf));
    EXPECT_FP_EQ(0.0, func(inf, 0.0));
    EXPECT_FP_EQ(-0.0, func(-0.0, inf));
    EXPECT_FP_EQ(T(1.2345), func(inf, T(1.2345)));
    EXPECT_FP_EQ(T(-1.2345), func(T(-1.2345), inf));
  }

  void testNegInfArg(FMinimumMagNumFunc func) {
    EXPECT_FP_EQ(neg_inf, func(inf, neg_inf));
    EXPECT_FP_EQ(0.0, func(neg_inf, 0.0));
    EXPECT_FP_EQ(-0.0, func(-0.0, neg_inf));
    EXPECT_FP_EQ(T(-1.2345), func(neg_inf, T(-1.2345)));
    EXPECT_FP_EQ(T(1.2345), func(T(1.2345), neg_inf));
  }

  void testBothZero(FMinimumMagNumFunc func) {
    EXPECT_FP_EQ(0.0, func(0.0, 0.0));
    EXPECT_FP_EQ(-0.0, func(-0.0, 0.0));
    EXPECT_FP_EQ(-0.0, func(0.0, -0.0));
    EXPECT_FP_EQ(-0.0, func(-0.0, -0.0));
  }

  void testRange(FMinimumMagNumFunc func) {
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

      if (LIBC_NAMESPACE::fputil::abs(x) > LIBC_NAMESPACE::fputil::abs(y)) {
        EXPECT_FP_EQ(y, func(x, y));
      } else {
        EXPECT_FP_EQ(x, func(x, y));
      }
    }
  }
};

#define LIST_FMINIMUM_MAG_NUM_TESTS(T, func)                                   \
  using LlvmLibcFMinimumMagNumTest = FMinimumMagNumTest<T>;                    \
  TEST_F(LlvmLibcFMinimumMagNumTest, NaN) { testNaN(&func); }                  \
  TEST_F(LlvmLibcFMinimumMagNumTest, InfArg) { testInfArg(&func); }            \
  TEST_F(LlvmLibcFMinimumMagNumTest, NegInfArg) { testNegInfArg(&func); }      \
  TEST_F(LlvmLibcFMinimumMagNumTest, BothZero) { testBothZero(&func); }        \
  TEST_F(LlvmLibcFMinimumMagNumTest, Range) { testRange(&func); }

#endif // LLVM_LIBC_TEST_SRC_MATH_SMOKE_FMINIMUMMAG_NUMTEST_H

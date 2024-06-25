//===-- Utility class to test fmin[f|l] -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_MATH_SMOKE_FMINTEST_H
#define LLVM_LIBC_TEST_SRC_MATH_SMOKE_FMINTEST_H

#include "src/__support/CPP/algorithm.h"
#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

template <typename T>
class FMinTest : public LIBC_NAMESPACE::testing::FEnvSafeTest {

  DECLARE_SPECIAL_CONSTANTS(T)

public:
  typedef T (*FMinFunc)(T, T);

  void testNaN(FMinFunc func) {
    EXPECT_FP_EQ(inf, func(aNaN, inf));
    EXPECT_FP_EQ(neg_inf, func(neg_inf, aNaN));
    EXPECT_FP_EQ(0.0, func(aNaN, 0.0));
    EXPECT_FP_EQ(-0.0, func(-0.0, aNaN));
    EXPECT_FP_EQ(T(-1.2345), func(aNaN, T(-1.2345)));
    EXPECT_FP_EQ(T(1.2345), func(T(1.2345), aNaN));
    EXPECT_FP_EQ(aNaN, func(aNaN, aNaN));
  }

  void testInfArg(FMinFunc func) {
    EXPECT_FP_EQ(neg_inf, func(neg_inf, inf));
    EXPECT_FP_EQ(0.0, func(inf, 0.0));
    EXPECT_FP_EQ(-0.0, func(-0.0, inf));
    EXPECT_FP_EQ(T(1.2345), func(inf, T(1.2345)));
    EXPECT_FP_EQ(T(-1.2345), func(T(-1.2345), inf));
  }

  void testNegInfArg(FMinFunc func) {
    EXPECT_FP_EQ(neg_inf, func(inf, neg_inf));
    EXPECT_FP_EQ(neg_inf, func(neg_inf, 0.0));
    EXPECT_FP_EQ(neg_inf, func(-0.0, neg_inf));
    EXPECT_FP_EQ(neg_inf, func(neg_inf, T(-1.2345)));
    EXPECT_FP_EQ(neg_inf, func(T(1.2345), neg_inf));
  }

  void testBothZero(FMinFunc func) {
    EXPECT_FP_EQ(0.0, func(0.0, 0.0));
    EXPECT_FP_EQ(-0.0, func(-0.0, 0.0));
    EXPECT_FP_EQ(-0.0, func(0.0, -0.0));
    EXPECT_FP_EQ(-0.0, func(-0.0, -0.0));
  }

  void testRange(FMinFunc func) {
    constexpr int COUNT = 100'001;
    constexpr StorageType STEP = LIBC_NAMESPACE::cpp::max(
        static_cast<StorageType>(STORAGE_MAX / COUNT), StorageType(1));
    StorageType v = 0, w = STORAGE_MAX;
    for (int i = 0; i <= COUNT; ++i, v += STEP, w -= STEP) {
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

#define LIST_FMIN_TESTS(T, func)                                               \
  using LlvmLibcFMinTest = FMinTest<T>;                                        \
  TEST_F(LlvmLibcFMinTest, NaN) { testNaN(&func); }                            \
  TEST_F(LlvmLibcFMinTest, InfArg) { testInfArg(&func); }                      \
  TEST_F(LlvmLibcFMinTest, NegInfArg) { testNegInfArg(&func); }                \
  TEST_F(LlvmLibcFMinTest, BothZero) { testBothZero(&func); }                  \
  TEST_F(LlvmLibcFMinTest, Range) { testRange(&func); }

#endif // LLVM_LIBC_TEST_SRC_MATH_SMOKE_FMINTEST_H

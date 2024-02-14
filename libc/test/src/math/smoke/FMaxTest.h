//===-- Utility class to test fmin[f|l] -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

template <typename T> class FMaxTest : public LIBC_NAMESPACE::testing::Test {

  DECLARE_SPECIAL_CONSTANTS(T)

public:
  typedef T (*FMaxFunc)(T, T);

  void testNaN(FMaxFunc func) {
    EXPECT_FP_EQ(inf, func(aNaN, inf));
    EXPECT_FP_EQ(neg_inf, func(neg_inf, aNaN));
    EXPECT_FP_EQ(0.0, func(aNaN, 0.0));
    EXPECT_FP_EQ(-0.0, func(-0.0, aNaN));
    EXPECT_FP_EQ(T(-1.2345), func(aNaN, T(-1.2345)));
    EXPECT_FP_EQ(T(1.2345), func(T(1.2345), aNaN));
    EXPECT_FP_EQ(aNaN, func(aNaN, aNaN));
  }

  void testInfArg(FMaxFunc func) {
    EXPECT_FP_EQ(inf, func(neg_inf, inf));
    EXPECT_FP_EQ(inf, func(inf, 0.0));
    EXPECT_FP_EQ(inf, func(-0.0, inf));
    EXPECT_FP_EQ(inf, func(inf, T(1.2345)));
    EXPECT_FP_EQ(inf, func(T(-1.2345), inf));
  }

  void testNegInfArg(FMaxFunc func) {
    EXPECT_FP_EQ(inf, func(inf, neg_inf));
    EXPECT_FP_EQ(0.0, func(neg_inf, 0.0));
    EXPECT_FP_EQ(-0.0, func(-0.0, neg_inf));
    EXPECT_FP_EQ(T(-1.2345), func(neg_inf, T(-1.2345)));
    EXPECT_FP_EQ(T(1.2345), func(T(1.2345), neg_inf));
  }

  void testBothZero(FMaxFunc func) {
    EXPECT_FP_EQ(0.0, func(0.0, 0.0));
    EXPECT_FP_EQ(0.0, func(-0.0, 0.0));
    EXPECT_FP_EQ(0.0, func(0.0, -0.0));
    EXPECT_FP_EQ(-0.0, func(-0.0, -0.0));
  }

  void testRange(FMaxFunc func) {
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
        EXPECT_FP_EQ(x, func(x, y));
      } else {
        EXPECT_FP_EQ(y, func(x, y));
      }
    }
  }
};

#define LIST_FMAX_TESTS(T, func)                                               \
  using LlvmLibcFMaxTest = FMaxTest<T>;                                        \
  TEST_F(LlvmLibcFMaxTest, NaN) { testNaN(&func); }                            \
  TEST_F(LlvmLibcFMaxTest, InfArg) { testInfArg(&func); }                      \
  TEST_F(LlvmLibcFMaxTest, NegInfArg) { testNegInfArg(&func); }                \
  TEST_F(LlvmLibcFMaxTest, BothZero) { testBothZero(&func); }                  \
  TEST_F(LlvmLibcFMaxTest, Range) { testRange(&func); }

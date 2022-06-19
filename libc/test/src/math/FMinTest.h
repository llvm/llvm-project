//===-- Utility class to test fmin[f|l] -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "utils/MPFRWrapper/MPFRUtils.h"
#include "utils/UnitTest/FPMatcher.h"
#include "utils/UnitTest/Test.h"

#include <math.h>

namespace mpfr = __llvm_libc::testing::mpfr;

template <typename T> class FMinTest : public __llvm_libc::testing::Test {

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
    constexpr UIntType COUNT = 10000001;
    constexpr UIntType STEP = UIntType(-1) / COUNT;
    for (UIntType i = 0, v = 0, w = UIntType(-1); i <= COUNT;
         ++i, v += STEP, w -= STEP) {
      T x = T(FPBits(v)), y = T(FPBits(w));
      if (isnan(x) || isinf(x))
        continue;
      if (isnan(y) || isinf(y))
        continue;
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

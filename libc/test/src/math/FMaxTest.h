//===-- Utility class to test fmin[f|l] -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/TestHelpers.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include "utils/UnitTest/Test.h"

#include <math.h>

namespace mpfr = __llvm_libc::testing::mpfr;

template <typename T> class FMaxTest : public __llvm_libc::testing::Test {

  DECLARE_SPECIAL_CONSTANTS(T)

public:
  typedef T (*FMaxFunc)(T, T);

  void testNaN(FMaxFunc func) {
    EXPECT_FP_EQ(inf, func(aNaN, inf));
    EXPECT_FP_EQ(negInf, func(negInf, aNaN));
    EXPECT_FP_EQ(0.0, func(aNaN, 0.0));
    EXPECT_FP_EQ(-0.0, func(-0.0, aNaN));
    EXPECT_FP_EQ(T(-1.2345), func(aNaN, T(-1.2345)));
    EXPECT_FP_EQ(T(1.2345), func(T(1.2345), aNaN));
    EXPECT_FP_EQ(aNaN, func(aNaN, aNaN));
  }

  void testInfArg(FMaxFunc func) {
    EXPECT_FP_EQ(inf, func(negInf, inf));
    EXPECT_FP_EQ(inf, func(inf, 0.0));
    EXPECT_FP_EQ(inf, func(-0.0, inf));
    EXPECT_FP_EQ(inf, func(inf, T(1.2345)));
    EXPECT_FP_EQ(inf, func(T(-1.2345), inf));
  }

  void testNegInfArg(FMaxFunc func) {
    EXPECT_FP_EQ(inf, func(inf, negInf));
    EXPECT_FP_EQ(0.0, func(negInf, 0.0));
    EXPECT_FP_EQ(-0.0, func(-0.0, negInf));
    EXPECT_FP_EQ(T(-1.2345), func(negInf, T(-1.2345)));
    EXPECT_FP_EQ(T(1.2345), func(T(1.2345), negInf));
  }

  void testBothZero(FMaxFunc func) {
    EXPECT_FP_EQ(0.0, func(0.0, 0.0));
    EXPECT_FP_EQ(0.0, func(-0.0, 0.0));
    EXPECT_FP_EQ(0.0, func(0.0, -0.0));
    EXPECT_FP_EQ(-0.0, func(-0.0, -0.0));
  }

  void testRange(FMaxFunc func) {
    constexpr UIntType count = 10000001;
    constexpr UIntType step = UIntType(-1) / count;
    for (UIntType i = 0, v = 0, w = UIntType(-1); i <= count;
         ++i, v += step, w -= step) {
      T x = T(FPBits(v)), y = T(FPBits(w));
      if (isnan(x) || isinf(x))
        continue;
      if (isnan(y) || isinf(y))
        continue;
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

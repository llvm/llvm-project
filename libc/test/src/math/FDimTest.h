//===-- Utility class to test different flavors of fdim ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "src/__support/FPUtil/BasicOperations.h"
#include "src/__support/FPUtil/FPBits.h"
#include "utils/UnitTest/FPMatcher.h"
#include "utils/UnitTest/Test.h"
#include <math.h>

template <typename T>
class FDimTestTemplate : public __llvm_libc::testing::Test {
public:
  using FuncPtr = T (*)(T, T);
  using FPBits = __llvm_libc::fputil::FPBits<T>;
  using UIntType = typename FPBits::UIntType;

  void testNaNArg(FuncPtr func) {
    EXPECT_FP_EQ(nan, func(nan, inf));
    EXPECT_FP_EQ(nan, func(negInf, nan));
    EXPECT_FP_EQ(nan, func(nan, zero));
    EXPECT_FP_EQ(nan, func(negZero, nan));
    EXPECT_FP_EQ(nan, func(nan, T(-1.2345)));
    EXPECT_FP_EQ(nan, func(T(1.2345), nan));
    EXPECT_FP_EQ(func(nan, nan), nan);
  }

  void testInfArg(FuncPtr func) {
    EXPECT_FP_EQ(zero, func(negInf, inf));
    EXPECT_FP_EQ(inf, func(inf, zero));
    EXPECT_FP_EQ(zero, func(negZero, inf));
    EXPECT_FP_EQ(inf, func(inf, T(1.2345)));
    EXPECT_FP_EQ(zero, func(T(-1.2345), inf));
  }

  void testNegInfArg(FuncPtr func) {
    EXPECT_FP_EQ(inf, func(inf, negInf));
    EXPECT_FP_EQ(zero, func(negInf, zero));
    EXPECT_FP_EQ(inf, func(negZero, negInf));
    EXPECT_FP_EQ(zero, func(negInf, T(-1.2345)));
    EXPECT_FP_EQ(inf, func(T(1.2345), negInf));
  }

  void testBothZero(FuncPtr func) {
    EXPECT_FP_EQ(zero, func(zero, zero));
    EXPECT_FP_EQ(zero, func(zero, negZero));
    EXPECT_FP_EQ(zero, func(negZero, zero));
    EXPECT_FP_EQ(zero, func(negZero, negZero));
  }

  void testInRange(FuncPtr func) {
    constexpr UIntType count = 10000001;
    constexpr UIntType step = UIntType(-1) / count;
    for (UIntType i = 0, v = 0, w = UIntType(-1); i <= count;
         ++i, v += step, w -= step) {
      T x = T(FPBits(v)), y = T(FPBits(w));
      if (isnan(x) || isinf(x))
        continue;
      if (isnan(y) || isinf(y))
        continue;

      if (x > y) {
        EXPECT_FP_EQ(x - y, func(x, y));
      } else {
        EXPECT_FP_EQ(zero, func(x, y));
      }
    }
  }

private:
  // constexpr does not work on FPBits yet, so we cannot have these constants as
  // static.
  const T nan = T(__llvm_libc::fputil::FPBits<T>::buildNaN(1));
  const T inf = T(__llvm_libc::fputil::FPBits<T>::inf());
  const T negInf = T(__llvm_libc::fputil::FPBits<T>::negInf());
  const T zero = T(__llvm_libc::fputil::FPBits<T>::zero());
  const T negZero = T(__llvm_libc::fputil::FPBits<T>::negZero());
};

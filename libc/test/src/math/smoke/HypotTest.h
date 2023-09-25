//===-- Utility class to test different flavors of hypot ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_MATH_HYPOTTEST_H
#define LLVM_LIBC_TEST_SRC_MATH_HYPOTTEST_H

#include "src/__support/FPUtil/FPBits.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

#include <math.h>

template <typename T>
class HypotTestTemplate : public __llvm_libc::testing::Test {
private:
  using Func = T (*)(T, T);
  using FPBits = __llvm_libc::fputil::FPBits<T>;
  using UIntType = typename FPBits::UIntType;
  const T nan = T(FPBits::build_quiet_nan(1));
  const T inf = T(FPBits::inf());
  const T neg_inf = T(FPBits::neg_inf());
  const T zero = T(FPBits::zero());
  const T neg_zero = T(FPBits::neg_zero());
  const T max_normal = T(FPBits(FPBits::MAX_NORMAL));
  const T min_normal = T(FPBits(FPBits::MIN_NORMAL));
  const T max_subnormal = T(FPBits(FPBits::MAX_SUBNORMAL));
  const T min_subnormal = T(FPBits(FPBits::MIN_SUBNORMAL));

public:
  void test_special_numbers(Func func) {
    constexpr int N = 4;
    // Pythagorean triples.
    constexpr T PYT[N][3] = {{3, 4, 5}, {5, 12, 13}, {8, 15, 17}, {7, 24, 25}};

    EXPECT_FP_EQ(func(inf, nan), inf);
    EXPECT_FP_EQ(func(nan, neg_inf), inf);
    EXPECT_FP_EQ(func(nan, nan), nan);
    EXPECT_FP_EQ(func(nan, zero), nan);
    EXPECT_FP_EQ(func(neg_zero, nan), nan);

    for (int i = 0; i < N; ++i) {
      EXPECT_FP_EQ_ALL_ROUNDING(PYT[i][2], func(PYT[i][0], PYT[i][1]));
      EXPECT_FP_EQ_ALL_ROUNDING(PYT[i][2], func(-PYT[i][0], PYT[i][1]));
      EXPECT_FP_EQ_ALL_ROUNDING(PYT[i][2], func(PYT[i][0], -PYT[i][1]));
      EXPECT_FP_EQ_ALL_ROUNDING(PYT[i][2], func(-PYT[i][0], -PYT[i][1]));

      EXPECT_FP_EQ_ALL_ROUNDING(PYT[i][2], func(PYT[i][1], PYT[i][0]));
      EXPECT_FP_EQ_ALL_ROUNDING(PYT[i][2], func(-PYT[i][1], PYT[i][0]));
      EXPECT_FP_EQ_ALL_ROUNDING(PYT[i][2], func(PYT[i][1], -PYT[i][0]));
      EXPECT_FP_EQ_ALL_ROUNDING(PYT[i][2], func(-PYT[i][1], -PYT[i][0]));
    }
  }
};

#endif // LLVM_LIBC_TEST_SRC_MATH_HYPOTTEST_H

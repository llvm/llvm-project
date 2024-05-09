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
#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

#include "hdr/math_macros.h"

template <typename T>
class HypotTestTemplate : public LIBC_NAMESPACE::testing::FEnvSafeTest {
private:
  using Func = T (*)(T, T);
  using FPBits = LIBC_NAMESPACE::fputil::FPBits<T>;
  using StorageType = typename FPBits::StorageType;

  const T nan = FPBits::quiet_nan().get_val();
  const T inf = FPBits::inf(Sign::POS).get_val();
  const T neg_inf = FPBits::inf(Sign::NEG).get_val();
  const T zero = FPBits::zero(Sign::POS).get_val();
  const T neg_zero = FPBits::zero(Sign::NEG).get_val();

  const T max_normal = FPBits::max_normal().get_val();
  const T min_normal = FPBits::min_normal().get_val();
  const T max_subnormal = FPBits::max_subnormal().get_val();
  const T min_subnormal = FPBits::min_subnormal().get_val();

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

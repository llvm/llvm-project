//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "test/UnitTest/SIMDMatcher.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::Sign;
using LIBC_NAMESPACE::testing::within_ulp_tolerance;

template <typename T> void test_infinities() {
  using FPBits = LIBC_NAMESPACE::fputil::FPBits<T>;
  constexpr uint64_t TOLERANCES[] = {0, 1, UINT64_MAX};
  constexpr Sign SIGNS[] = {Sign::POS, Sign::NEG};
  for (uint64_t tolerance : TOLERANCES) {
    for (Sign sign : SIGNS) {
      T inf = FPBits::inf(sign).get_val();
      T max = FPBits::max_normal(sign).get_val();
      EXPECT_TRUE(within_ulp_tolerance(inf, inf, tolerance));
      EXPECT_FALSE(within_ulp_tolerance(inf, max, tolerance));
      EXPECT_FALSE(within_ulp_tolerance(max, inf, tolerance));
    }
    T inf = FPBits::inf().get_val();
    T neg_inf = FPBits::inf(Sign::NEG).get_val();
    EXPECT_FALSE(within_ulp_tolerance(inf, neg_inf, tolerance));
    EXPECT_FALSE(within_ulp_tolerance(neg_inf, inf, tolerance));
  }
}

template <typename T> void test_finite_values() {
  using FPBits = LIBC_NAMESPACE::fputil::FPBits<T>;
  T x = T(1.0);
  T next = FPBits(FPBits(x).uintval() + 1).get_val();
  T next2 = FPBits(FPBits(x).uintval() + 2).get_val();
  EXPECT_TRUE(within_ulp_tolerance(x, x, 0));
  EXPECT_FALSE(within_ulp_tolerance(x, next, 0));
  EXPECT_TRUE(within_ulp_tolerance(x, next, 1));
  EXPECT_TRUE(within_ulp_tolerance(next, x, 1));
  EXPECT_FALSE(within_ulp_tolerance(x, next2, 1));
  EXPECT_TRUE(within_ulp_tolerance(x, next2, 2));
  EXPECT_TRUE(within_ulp_tolerance(-x, -next, 1));
  EXPECT_FALSE(within_ulp_tolerance(-x, -next2, 1));
  EXPECT_FALSE(within_ulp_tolerance(T(0.0), T(-0.0), 0));
  EXPECT_TRUE(within_ulp_tolerance(T(-0.0), T(-0.0), 0));
}

template <typename T> void test_nans() {
  using FPBits = LIBC_NAMESPACE::fputil::FPBits<T>;
  T nan = FPBits::quiet_nan().get_val();
  T other_nan = FPBits::quiet_nan(Sign::NEG, 1).get_val();
  EXPECT_TRUE(within_ulp_tolerance(nan, other_nan, 0));
  EXPECT_FALSE(within_ulp_tolerance(nan, T(1.0), UINT64_MAX));
  EXPECT_FALSE(within_ulp_tolerance(T(1.0), nan, UINT64_MAX));
  EXPECT_FALSE(within_ulp_tolerance(nan, FPBits::inf().get_val(), UINT64_MAX));
}

TEST(LlvmLibcSIMDMatcherTest, FloatInfinities) { test_infinities<float>(); }
TEST(LlvmLibcSIMDMatcherTest, DoubleInfinities) { test_infinities<double>(); }
TEST(LlvmLibcSIMDMatcherTest, FloatFiniteValues) {
  test_finite_values<float>();
}
TEST(LlvmLibcSIMDMatcherTest, DoubleFiniteValues) {
  test_finite_values<double>();
}
TEST(LlvmLibcSIMDMatcherTest, FloatNaNs) { test_nans<float>(); }
TEST(LlvmLibcSIMDMatcherTest, DoubleNaNs) { test_nans<double>(); }

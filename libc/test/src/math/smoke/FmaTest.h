//===-- Utility class to test different flavors of fma --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_MATH_FMATEST_H
#define LLVM_LIBC_TEST_SRC_MATH_FMATEST_H

#include "src/__support/FPUtil/FPBits.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

template <typename T>
class FmaTestTemplate : public LIBC_NAMESPACE::testing::Test {
private:
  using Func = T (*)(T, T, T);
  using FPBits = LIBC_NAMESPACE::fputil::FPBits<T>;
  using StorageType = typename FPBits::StorageType;
  using Sign = LIBC_NAMESPACE::fputil::Sign;

  const T inf = FPBits::inf(Sign::POS).get_val();
  const T neg_inf = FPBits::inf(Sign::NEG).get_val();
  const T zero = FPBits::zero(Sign::POS).get_val();
  const T neg_zero = FPBits::zero(Sign::NEG).get_val();
  const T nan = FPBits::build_quiet_nan().get_val();

public:
  void test_special_numbers(Func func) {
    EXPECT_FP_EQ(func(zero, zero, zero), zero);
    EXPECT_FP_EQ(func(zero, neg_zero, neg_zero), neg_zero);
    EXPECT_FP_EQ(func(inf, inf, zero), inf);
    EXPECT_FP_EQ(func(neg_inf, inf, neg_inf), neg_inf);
    EXPECT_FP_EQ(func(inf, zero, zero), nan);
    EXPECT_FP_EQ(func(inf, neg_inf, inf), nan);
    EXPECT_FP_EQ(func(nan, zero, inf), nan);
    EXPECT_FP_EQ(func(inf, neg_inf, nan), nan);

    // Test underflow rounding up.
    EXPECT_FP_EQ(func(T(0.5), FPBits::min_subnormal().get_val(),
                      FPBits::min_subnormal().get_val()),
                 FPBits(StorageType(2)).get_val());
    // Test underflow rounding down.
    StorageType MIN_NORMAL = FPBits::min_normal().uintval();
    T v = FPBits(MIN_NORMAL + StorageType(1)).get_val();
    EXPECT_FP_EQ(
        func(T(1) / T(MIN_NORMAL << 1), v, FPBits::min_normal().get_val()), v);
    // Test overflow.
    T z = FPBits::max_normal().get_val();
    EXPECT_FP_EQ(func(T(1.75), z, -z), T(0.75) * z);
    // Exact cancellation.
    EXPECT_FP_EQ(func(T(3.0), T(5.0), -T(15.0)), T(0.0));
    EXPECT_FP_EQ(func(T(-3.0), T(5.0), T(15.0)), T(0.0));
  }
};

#endif // LLVM_LIBC_TEST_SRC_MATH_FMATEST_H

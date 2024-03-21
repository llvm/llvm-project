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
#include "test/src/math/RandUtils.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

template <typename T>
class FmaTestTemplate : public LIBC_NAMESPACE::testing::Test {
private:
  using Func = T (*)(T, T, T);
  using FPBits = LIBC_NAMESPACE::fputil::FPBits<T>;
  using StorageType = typename FPBits::StorageType;

  const T min_subnormal = FPBits::min_subnormal(Sign::POS).get_val();
  const T min_normal = FPBits::min_normal(Sign::POS).get_val();
  const T max_normal = FPBits::max_normal(Sign::POS).get_val();
  const T inf = FPBits::inf(Sign::POS).get_val();
  const T neg_inf = FPBits::inf(Sign::NEG).get_val();
  const T zero = FPBits::zero(Sign::POS).get_val();
  const T neg_zero = FPBits::zero(Sign::NEG).get_val();
  const T nan = FPBits::quiet_nan().get_val();

  static constexpr StorageType MAX_NORMAL = FPBits::max_normal().uintval();
  static constexpr StorageType MIN_NORMAL = FPBits::min_normal().uintval();
  static constexpr StorageType MAX_SUBNORMAL =
      FPBits::max_subnormal().uintval();
  static constexpr StorageType MIN_SUBNORMAL =
      FPBits::min_subnormal().uintval();

  StorageType get_random_bit_pattern() {
    StorageType bits{0};
    for (StorageType i = 0; i < sizeof(StorageType) / 2; ++i) {
      bits = (bits << 2) +
             static_cast<uint16_t>(LIBC_NAMESPACE::testutils::rand());
    }
    return bits;
  }

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
    EXPECT_FP_EQ(func(T(0.5), min_subnormal, min_subnormal),
                 FPBits(StorageType(2)).get_val());
    // Test underflow rounding down.
    T v = FPBits(MIN_NORMAL + StorageType(1)).get_val();
    EXPECT_FP_EQ(func(T(1) / T(MIN_NORMAL << 1), v, min_normal), v);
    // Test overflow.
    T z = max_normal;
    EXPECT_FP_EQ(func(T(1.75), z, -z), T(0.75) * z);
    // Exact cancellation.
    EXPECT_FP_EQ(func(T(3.0), T(5.0), -T(15.0)), T(0.0));
    EXPECT_FP_EQ(func(T(-3.0), T(5.0), T(15.0)), T(0.0));
  }

  void test_subnormal_range(Func func) {
    constexpr StorageType COUNT = 100'001;
    constexpr StorageType STEP = (MAX_SUBNORMAL - MIN_SUBNORMAL) / COUNT;
    for (StorageType v = MIN_SUBNORMAL, w = MAX_SUBNORMAL;
         v <= MAX_SUBNORMAL && w >= MIN_SUBNORMAL; v += STEP, w -= STEP) {
      T x = FPBits(get_random_bit_pattern()).get_val(), y = FPBits(v).get_val(),
        z = FPBits(w).get_val();
      mpfr::TernaryInput<T> input{x, y, z};
      ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Fma, input, func(x, y, z),
                                     0.5);
    }
  }

  void test_normal_range(Func func) {
    constexpr StorageType COUNT = 100'001;
    constexpr StorageType STEP = (MAX_NORMAL - MIN_NORMAL) / COUNT;
    for (StorageType v = MIN_NORMAL, w = MAX_NORMAL;
         v <= MAX_NORMAL && w >= MIN_NORMAL; v += STEP, w -= STEP) {
      T x = FPBits(v).get_val(), y = FPBits(w).get_val(),
        z = FPBits(get_random_bit_pattern()).get_val();
      mpfr::TernaryInput<T> input{x, y, z};
      ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Fma, input, func(x, y, z),
                                     0.5);
    }
  }
};

#endif // LLVM_LIBC_TEST_SRC_MATH_FMATEST_H

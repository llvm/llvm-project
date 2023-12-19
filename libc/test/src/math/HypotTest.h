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
#include "utils/MPFRWrapper/MPFRUtils.h"

#include <math.h>

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

template <typename T>
class HypotTestTemplate : public LIBC_NAMESPACE::testing::Test {
private:
  using Func = T (*)(T, T);
  using FPBits = LIBC_NAMESPACE::fputil::FPBits<T>;
  using StorageType = typename FPBits::StorageType;
  const T nan = FPBits::build_quiet_nan(1);
  const T inf = FPBits::inf();
  const T neg_inf = FPBits::neg_inf();
  const T zero = FPBits::zero();
  const T neg_zero = FPBits::neg_zero();
  const T max_normal = FPBits::max_normal();
  const T min_normal = FPBits::min_normal();
  const T max_subnormal = FPBits::max_denormal();
  const T min_subnormal = FPBits::min_denormal();

public:
  void test_special_numbers(Func func) {
    constexpr int N = 13;
    const T SpecialInputs[N] = {inf,           neg_inf,        zero,
                                neg_zero,      max_normal,     min_normal,
                                max_subnormal, min_subnormal,  -max_normal,
                                -min_normal,   -max_subnormal, -min_subnormal};

    EXPECT_FP_EQ(func(inf, nan), inf);
    EXPECT_FP_EQ(func(nan, neg_inf), inf);
    EXPECT_FP_EQ(func(nan, nan), nan);
    EXPECT_FP_EQ(func(nan, zero), nan);
    EXPECT_FP_EQ(func(neg_zero, nan), nan);

    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        mpfr::BinaryInput<T> input{SpecialInputs[i], SpecialInputs[j]};
        EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Hypot, input,
                                       func(SpecialInputs[i], SpecialInputs[j]),
                                       0.5);
      }
    }
  }

  void test_subnormal_range(Func func) {
    constexpr StorageType COUNT = 10'001;
    for (unsigned scale = 0; scale < 4; ++scale) {
      StorageType max_value = FPBits::MAX_SUBNORMAL << scale;
      StorageType step = (max_value - FPBits::MIN_SUBNORMAL) / COUNT;
      for (int signs = 0; signs < 4; ++signs) {
        for (StorageType v = FPBits::MIN_SUBNORMAL, w = max_value;
             v <= max_value && w >= FPBits::MIN_SUBNORMAL;
             v += step, w -= step) {
          T x = T(FPBits(v)), y = T(FPBits(w));
          if (signs % 2 == 1) {
            x = -x;
          }
          if (signs >= 2) {
            y = -y;
          }

          mpfr::BinaryInput<T> input{x, y};
          ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Hypot, input,
                                         func(x, y), 0.5);
        }
      }
    }
  }

  void test_normal_range(Func func) {
    constexpr StorageType COUNT = 10'001;
    constexpr StorageType STEP =
        (StorageType(FPBits::MAX_NORMAL) - StorageType(FPBits::MIN_NORMAL)) /
        COUNT;
    for (int signs = 0; signs < 4; ++signs) {
      for (StorageType v = FPBits::MIN_NORMAL, w = FPBits::MAX_NORMAL;
           v <= FPBits::MAX_NORMAL && w >= FPBits::MIN_NORMAL;
           v += STEP, w -= STEP) {
        T x = T(FPBits(v)), y = T(FPBits(w));
        if (signs % 2 == 1) {
          x = -x;
        }
        if (signs >= 2) {
          y = -y;
        }

        mpfr::BinaryInput<T> input{x, y};
        ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Hypot, input,
                                       func(x, y), 0.5);
      }
    }
  }

  void test_input_list(Func func, int n, const mpfr::BinaryInput<T> *inputs) {
    for (int i = 0; i < n; ++i) {
      ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Hypot, inputs[i],
                                     func(inputs[i].x, inputs[i].y), 0.5);
    }
  }
};

#endif // LLVM_LIBC_TEST_SRC_MATH_HYPOTTEST_H

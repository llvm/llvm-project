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
#include "utils/MPFRWrapper/MPFRUtils.h"
#include "utils/UnitTest/FPMatcher.h"
#include "utils/UnitTest/Test.h"

#include <math.h>

namespace mpfr = __llvm_libc::testing::mpfr;

template <typename T>
class HypotTestTemplate : public __llvm_libc::testing::Test {
private:
  using Func = T (*)(T, T);
  using FPBits = __llvm_libc::fputil::FPBits<T>;
  using UIntType = typename FPBits::UIntType;
  const T nan = T(FPBits::build_nan(1));
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
    constexpr UIntType COUNT = 1000001;
    for (unsigned scale = 0; scale < 4; ++scale) {
      UIntType max_value = FPBits::MAX_SUBNORMAL << scale;
      UIntType step = (max_value - FPBits::MIN_SUBNORMAL) / COUNT;
      for (int signs = 0; signs < 4; ++signs) {
        for (UIntType v = FPBits::MIN_SUBNORMAL, w = max_value;
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
    constexpr UIntType COUNT = 1000001;
    constexpr UIntType STEP = (FPBits::MAX_NORMAL - FPBits::MIN_NORMAL) / COUNT;
    for (int signs = 0; signs < 4; ++signs) {
      for (UIntType v = FPBits::MIN_NORMAL, w = FPBits::MAX_NORMAL;
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

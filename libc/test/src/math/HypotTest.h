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
#include "src/__support/FPUtil/Hypot.h"
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
  const T nan = T(__llvm_libc::fputil::FPBits<T>::build_nan(1));
  const T inf = T(__llvm_libc::fputil::FPBits<T>::inf());
  const T neg_inf = T(__llvm_libc::fputil::FPBits<T>::neg_inf());
  const T zero = T(__llvm_libc::fputil::FPBits<T>::zero());
  const T neg_zero = T(__llvm_libc::fputil::FPBits<T>::neg_zero());

public:
  void test_special_numbers(Func func) {
    EXPECT_FP_EQ(func(inf, nan), inf);
    EXPECT_FP_EQ(func(nan, neg_inf), inf);
    EXPECT_FP_EQ(func(zero, inf), inf);
    EXPECT_FP_EQ(func(neg_inf, neg_zero), inf);

    EXPECT_FP_EQ(func(nan, nan), nan);
    EXPECT_FP_EQ(func(nan, zero), nan);
    EXPECT_FP_EQ(func(neg_zero, nan), nan);

    EXPECT_FP_EQ(func(neg_zero, zero), zero);
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

          T result = func(x, y);
          mpfr::BinaryInput<T> input{x, y};
          ASSERT_MPFR_MATCH(mpfr::Operation::Hypot, input, result, 0.5);
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

        T result = func(x, y);
        mpfr::BinaryInput<T> input{x, y};
        ASSERT_MPFR_MATCH(mpfr::Operation::Hypot, input, result, 0.5);
      }
    }
  }
};

#endif // LLVM_LIBC_TEST_SRC_MATH_HYPOTTEST_H

//===-- Utility class to test different flavors of hypot ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_MATH_HYPOTTEST_H
#define LLVM_LIBC_TEST_SRC_MATH_HYPOTTEST_H

#include "utils/FPUtil/FPBits.h"
#include "utils/FPUtil/Hypot.h"
#include "utils/FPUtil/TestHelpers.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include "utils/UnitTest/Test.h"

#include <math.h>

namespace mpfr = __llvm_libc::testing::mpfr;

template <typename T>
class HypotTestTemplate : public __llvm_libc::testing::Test {
private:
  using Func = T (*)(T, T);
  using FPBits = __llvm_libc::fputil::FPBits<T>;
  using UIntType = typename FPBits::UIntType;
  const T nan = __llvm_libc::fputil::FPBits<T>::buildNaN(1);
  const T inf = __llvm_libc::fputil::FPBits<T>::inf();
  const T negInf = __llvm_libc::fputil::FPBits<T>::negInf();
  const T zero = __llvm_libc::fputil::FPBits<T>::zero();
  const T negZero = __llvm_libc::fputil::FPBits<T>::negZero();

public:
  void testSpecialNumbers(Func func) {
    EXPECT_FP_EQ(func(inf, nan), inf);
    EXPECT_FP_EQ(func(nan, negInf), inf);
    EXPECT_FP_EQ(func(zero, inf), inf);
    EXPECT_FP_EQ(func(negInf, negZero), inf);

    EXPECT_FP_EQ(func(nan, nan), nan);
    EXPECT_FP_EQ(func(nan, zero), nan);
    EXPECT_FP_EQ(func(negZero, nan), nan);

    EXPECT_FP_EQ(func(negZero, zero), zero);
  }

  void testSubnormalRange(Func func) {
    constexpr UIntType count = 1000001;
    constexpr UIntType step =
        (FPBits::maxSubnormal - FPBits::minSubnormal) / count;
    for (UIntType v = FPBits::minSubnormal, w = FPBits::maxSubnormal;
         v <= FPBits::maxSubnormal && w >= FPBits::minSubnormal;
         v += step, w -= step) {
      T x = FPBits(v), y = FPBits(w);
      T result = func(x, y);
      mpfr::BinaryInput<T> input{x, y};
      ASSERT_MPFR_MATCH(mpfr::Operation::Hypot, input, result, 0.5);
    }
  }

  void testNormalRange(Func func) {
    constexpr UIntType count = 1000001;
    constexpr UIntType step = (FPBits::maxNormal - FPBits::minNormal) / count;
    for (UIntType v = FPBits::minNormal, w = FPBits::maxNormal;
         v <= FPBits::maxNormal && w >= FPBits::minNormal;
         v += step, w -= step) {
      T x = FPBits(v), y = FPBits(w);
      T result = func(x, y);
      mpfr::BinaryInput<T> input{x, y};
      ASSERT_MPFR_MATCH(mpfr::Operation::Hypot, input, result, 0.5);
    }
  }
};

#endif // LLVM_LIBC_TEST_SRC_MATH_HYPOTTEST_H

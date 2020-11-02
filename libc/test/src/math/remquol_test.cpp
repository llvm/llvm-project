//===-- Unittests for remquol ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/math.h"
#include "src/math/remquol.h"
#include "utils/FPUtil/BasicOperations.h"
#include "utils/FPUtil/FPBits.h"
#include "utils/FPUtil/TestHelpers.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include "utils/UnitTest/Test.h"

using FPBits = __llvm_libc::fputil::FPBits<long double>;
using UIntType = FPBits::UIntType;

namespace mpfr = __llvm_libc::testing::mpfr;

DECLARE_SPECIAL_CONSTANTS(long double)

TEST(RemquolTest, SpecialNumbers) {
  int exponent;
  long double x, y;

  y = 1.0l;
  x = inf;
  EXPECT_NE(isnan(__llvm_libc::remquol(x, y, &exponent)), 0);
  x = negInf;
  EXPECT_NE(isnan(__llvm_libc::remquol(x, y, &exponent)), 0);

  x = 1.0l;
  y = zero;
  EXPECT_NE(isnan(__llvm_libc::remquol(x, y, &exponent)), 0);
  y = negZero;
  EXPECT_NE(isnan(__llvm_libc::remquol(x, y, &exponent)), 0);

  y = nan;
  x = 1.0l;
  EXPECT_NE(isnan(__llvm_libc::remquol(x, y, &exponent)), 0);

  y = 1.0l;
  x = nan;
  EXPECT_NE(isnan(__llvm_libc::remquol(x, y, &exponent)), 0);

  x = nan;
  y = nan;
  EXPECT_NE(isnan(__llvm_libc::remquol(x, y, &exponent)), 0);

  x = zero;
  y = 1.0l;
  EXPECT_FP_EQ(__llvm_libc::remquol(x, y, &exponent), zero);

  x = negZero;
  y = 1.0l;
  EXPECT_FP_EQ(__llvm_libc::remquol(x, y, &exponent), negZero);
}

TEST(RemquolTest, SubnormalRange) {
  constexpr UIntType count = 1000001;
  constexpr UIntType step =
      (FPBits::maxSubnormal - FPBits::minSubnormal) / count;
  for (UIntType v = FPBits::minSubnormal, w = FPBits::maxSubnormal;
       v <= FPBits::maxSubnormal && w >= FPBits::minSubnormal;
       v += step, w -= step) {
    long double x = FPBits(v), y = FPBits(w);
    mpfr::BinaryOutput<long double> result;
    mpfr::BinaryInput<long double> input{x, y};
    result.f = __llvm_libc::remquol(x, y, &result.i);
    ASSERT_MPFR_MATCH(mpfr::Operation::RemQuo, input, result, 0.0);
  }
}

TEST(RemquolTest, NormalRange) {
  constexpr UIntType count = 1000001;
  constexpr UIntType step = (FPBits::maxNormal - FPBits::minNormal) / count;
  for (UIntType v = FPBits::minNormal, w = FPBits::maxNormal;
       v <= FPBits::maxNormal && w >= FPBits::minNormal; v += step, w -= step) {
    long double x = FPBits(v), y = FPBits(w);
    mpfr::BinaryOutput<long double> result;
    result.f = __llvm_libc::remquol(x, y, &result.i);
    // In normal range on x86 platforms, the implicit 1 bit can be zero making
    // the numbers NaN. Hence we test for them separately.
    if (isnan(x) || isnan(y)) {
      ASSERT_NE(isnan(result.f), 0);
    } else {
      mpfr::BinaryInput<long double> input{x, y};
      ASSERT_MPFR_MATCH(mpfr::Operation::RemQuo, input, result, 0.0);
    }
  }
}

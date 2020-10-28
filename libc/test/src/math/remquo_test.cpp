//===-- Unittests for remquo ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/math.h"
#include "src/math/remquo.h"
#include "utils/FPUtil/BasicOperations.h"
#include "utils/FPUtil/FPBits.h"
#include "utils/FPUtil/TestHelpers.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include "utils/UnitTest/Test.h"

using FPBits = __llvm_libc::fputil::FPBits<double>;
using UIntType = FPBits::UIntType;

namespace mpfr = __llvm_libc::testing::mpfr;

static const double zero = FPBits::zero();
static const double negZero = FPBits::negZero();
static const double nan = FPBits::buildNaN(1);
static const double inf = FPBits::inf();
static const double negInf = FPBits::negInf();

TEST(RemquoTest, SpecialNumbers) {
  int exponent;
  double x, y;

  y = 1.0;
  x = inf;
  EXPECT_NE(isnan(__llvm_libc::remquo(x, y, &exponent)), 0);
  x = negInf;
  EXPECT_NE(isnan(__llvm_libc::remquo(x, y, &exponent)), 0);

  x = 1.0;
  y = zero;
  EXPECT_NE(isnan(__llvm_libc::remquo(x, y, &exponent)), 0);
  y = negZero;
  EXPECT_NE(isnan(__llvm_libc::remquo(x, y, &exponent)), 0);

  y = nan;
  x = 1.0;
  EXPECT_NE(isnan(__llvm_libc::remquo(x, y, &exponent)), 0);

  y = 1.0;
  x = nan;
  EXPECT_NE(isnan(__llvm_libc::remquo(x, y, &exponent)), 0);

  x = nan;
  y = nan;
  EXPECT_NE(isnan(__llvm_libc::remquo(x, y, &exponent)), 0);

  x = zero;
  y = 1.0;
  EXPECT_FP_EQ(__llvm_libc::remquo(x, y, &exponent), zero);

  x = negZero;
  y = 1.0;
  EXPECT_FP_EQ(__llvm_libc::remquo(x, y, &exponent), negZero);
}

TEST(RemquoTest, SubnormalRange) {
  constexpr UIntType count = 1000001;
  constexpr UIntType step =
      (FPBits::maxSubnormal - FPBits::minSubnormal) / count;
  for (UIntType v = FPBits::minSubnormal, w = FPBits::maxSubnormal;
       v <= FPBits::maxSubnormal && w >= FPBits::minSubnormal;
       v += step, w -= step) {
    double x = FPBits(v), y = FPBits(w);
    mpfr::BinaryOutput<double> result;
    mpfr::BinaryInput<double> input{x, y};
    result.f = __llvm_libc::remquo(x, y, &result.i);
    ASSERT_MPFR_MATCH(mpfr::Operation::RemQuo, input, result, 0.0);
  }
}

TEST(RemquoTest, NormalRange) {
  constexpr UIntType count = 1000001;
  constexpr UIntType step = (FPBits::maxNormal - FPBits::minNormal) / count;
  for (UIntType v = FPBits::minNormal, w = FPBits::maxNormal;
       v <= FPBits::maxNormal && w >= FPBits::minNormal; v += step, w -= step) {
    double x = FPBits(v), y = FPBits(w);
    mpfr::BinaryOutput<double> result;
    mpfr::BinaryInput<double> input{x, y};
    result.f = __llvm_libc::remquo(x, y, &result.i);
    ASSERT_MPFR_MATCH(mpfr::Operation::RemQuo, input, result, 0.0);
  }
}

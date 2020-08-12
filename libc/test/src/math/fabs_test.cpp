//===-- Unittests for fabs ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/math.h"
#include "src/math/fabs.h"
#include "utils/FPUtil/FPBits.h"
#include "utils/FPUtil/TestHelpers.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include "utils/UnitTest/Test.h"

using FPBits = __llvm_libc::fputil::FPBits<double>;

static const double zero = FPBits::zero();
static const double negZero = FPBits::negZero();
static const double nan = FPBits::buildNaN(1);
static const double inf = FPBits::inf();
static const double negInf = FPBits::negInf();

namespace mpfr = __llvm_libc::testing::mpfr;

// Zero tolerance; As in, exact match with MPFR result.
static constexpr mpfr::Tolerance tolerance{mpfr::Tolerance::doublePrecision, 0,
                                           0};

TEST(FabsTest, SpecialNumbers) {
  EXPECT_FP_EQ(nan, __llvm_libc::fabs(nan));

  EXPECT_FP_EQ(inf, __llvm_libc::fabs(inf));
  EXPECT_FP_EQ(inf, __llvm_libc::fabs(negInf));

  EXPECT_FP_EQ(zero, __llvm_libc::fabs(zero));
  EXPECT_FP_EQ(zero, __llvm_libc::fabs(negZero));
}

TEST(FabsTest, InDoubleRange) {
  using UIntType = FPBits::UIntType;
  constexpr UIntType count = 10000000;
  constexpr UIntType step = UIntType(-1) / count;
  for (UIntType i = 0, v = 0; i <= count; ++i, v += step) {
    double x = FPBits(v);
    if (isnan(x) || isinf(x))
      continue;
    ASSERT_MPFR_MATCH(mpfr::Operation::Abs, x, __llvm_libc::fabs(x), tolerance);
  }
}

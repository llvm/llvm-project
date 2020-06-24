//===-- Unittests for fabsl -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/math.h"
#include "src/math/fabsl.h"
#include "utils/FPUtil/FPBits.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include "utils/UnitTest/Test.h"

using FPBits = __llvm_libc::fputil::FPBits<long double>;

namespace mpfr = __llvm_libc::testing::mpfr;

// Zero tolerance; As in, exact match with MPFR result.
static constexpr mpfr::Tolerance tolerance{mpfr::Tolerance::floatPrecision, 0,
                                           0};

TEST(FabslTest, SpecialNumbers) {
  EXPECT_TRUE(FPBits::zero() == __llvm_libc::fabsl(FPBits::zero()));
  EXPECT_TRUE(FPBits::zero() == __llvm_libc::fabsl(FPBits::negZero()));

  EXPECT_TRUE(FPBits::inf() == __llvm_libc::fabsl(FPBits::inf()));
  EXPECT_TRUE(FPBits::inf() == __llvm_libc::fabsl(FPBits::negInf()));

  long double nan = FPBits::buildNaN(1);
  ASSERT_TRUE(isnan(nan) != 0);
  ASSERT_TRUE(isnan(__llvm_libc::fabsl(nan)) != 0);
}

TEST(FabslTest, InLongDoubleRange) {
  using UIntType = FPBits::UIntType;
  constexpr UIntType count = 10000000;
  constexpr UIntType step = UIntType(-1) / count;
  for (UIntType i = 0, v = 0; i <= count; ++i, v += step) {
    long double x = FPBits(v);
    if (isnan(x) || isinf(x))
      continue;
    ASSERT_MPFR_MATCH(mpfr::Operation::Abs, x, __llvm_libc::fabsl(x),
                      tolerance);
  }
}

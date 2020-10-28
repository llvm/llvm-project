//===-- Unittests for copysignl -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/math.h"
#include "src/math/copysignl.h"
#include "utils/FPUtil/FPBits.h"
#include "utils/FPUtil/TestHelpers.h"
#include "utils/UnitTest/Test.h"

using FPBits = __llvm_libc::fputil::FPBits<long double>;

static const long double zero = FPBits::zero();
static const long double negZero = FPBits::negZero();
static const long double nan = FPBits::buildNaN(1);
static const long double inf = FPBits::inf();
static const long double negInf = FPBits::negInf();

TEST(CopySinlTest, SpecialNumbers) {
  EXPECT_FP_EQ(nan, __llvm_libc::copysignl(nan, -1.0));
  EXPECT_FP_EQ(nan, __llvm_libc::copysignl(nan, 1.0));

  EXPECT_FP_EQ(negInf, __llvm_libc::copysignl(inf, -1.0));
  EXPECT_FP_EQ(inf, __llvm_libc::copysignl(negInf, 1.0));

  EXPECT_FP_EQ(negZero, __llvm_libc::copysignl(zero, -1.0));
  EXPECT_FP_EQ(zero, __llvm_libc::copysignl(negZero, 1.0));
}

TEST(CopySinlTest, InLongDoubleRange) {
  using UIntType = FPBits::UIntType;
  constexpr UIntType count = 10000000;
  constexpr UIntType step = UIntType(-1) / count;
  for (UIntType i = 0, v = 0; i <= count; ++i, v += step) {
    long double x = FPBits(v);
    if (isnan(x) || isinf(x) || x == 0)
      continue;

    long double res1 = __llvm_libc::copysignl(x, -x);
    ASSERT_FP_EQ(res1, -x);

    long double res2 = __llvm_libc::copysignl(x, x);
    ASSERT_FP_EQ(res2, x);
  }
}

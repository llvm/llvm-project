//===-- Unittests for copysignl
//--------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/math.h"
#include "src/math/copysignl.h"
#include "utils/FPUtil/FPBits.h"
#include "utils/UnitTest/Test.h"

using FPBits = __llvm_libc::fputil::FPBits<long double>;

TEST(CopysignlTest, SpecialNumbers) {
  EXPECT_TRUE(FPBits::negZero() ==
              __llvm_libc::copysignl(FPBits::zero(), -1.0l));
  EXPECT_TRUE(FPBits::zero() ==
              __llvm_libc::copysignl(FPBits::negZero(), 1.0l));

  EXPECT_TRUE(FPBits::negZero() ==
              __llvm_libc::copysignl(FPBits::zero(), -1.0l));
  EXPECT_TRUE(FPBits::zero() ==
              __llvm_libc::copysignl(FPBits::negZero(), 1.0l));

  EXPECT_TRUE(
      FPBits(__llvm_libc::copysignl(FPBits::buildNaN(1), -1.0l)).isNaN());
}

TEST(CopysignlTest, InDoubleRange) {
  using UIntType = FPBits::UIntType;
  constexpr UIntType count = 10000000;
  constexpr UIntType step = UIntType(-1) / count;
  for (UIntType i = 0, v = 0; i <= count; ++i, v += step) {
    long double x = FPBits(v);
    if (isnan(x) || isinf(x) || x == 0)
      continue;

    long double res1 = __llvm_libc::copysignl(x, -x);
    ASSERT_TRUE(res1 == -x);

    long double res2 = __llvm_libc::copysignl(x, x);
    ASSERT_TRUE(res2 == x);
  }
}

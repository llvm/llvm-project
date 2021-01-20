//===-- Unittests for fabsl -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/fabsl.h"
#include "utils/FPUtil/FPBits.h"
#include "utils/FPUtil/TestHelpers.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include "utils/UnitTest/Test.h"
#include <math.h>

using FPBits = __llvm_libc::fputil::FPBits<long double>;

DECLARE_SPECIAL_CONSTANTS(long double)

namespace mpfr = __llvm_libc::testing::mpfr;

TEST(LlvmLibcFabslTest, SpecialNumbers) {
  EXPECT_FP_EQ(aNaN, __llvm_libc::fabsl(aNaN));

  EXPECT_FP_EQ(inf, __llvm_libc::fabsl(inf));
  EXPECT_FP_EQ(inf, __llvm_libc::fabsl(negInf));

  EXPECT_FP_EQ(zero, __llvm_libc::fabsl(zero));
  EXPECT_FP_EQ(zero, __llvm_libc::fabsl(negZero));
}

TEST(LlvmLibcFabslTest, InLongDoubleRange) {
  using UIntType = FPBits::UIntType;
  constexpr UIntType count = 10000000;
  constexpr UIntType step = UIntType(-1) / count;
  for (UIntType i = 0, v = 0; i <= count; ++i, v += step) {
    long double x = FPBits(v);
    if (isnan(x) || isinf(x))
      continue;
    ASSERT_MPFR_MATCH(mpfr::Operation::Abs, x, __llvm_libc::fabsl(x), 0.0);
  }
}

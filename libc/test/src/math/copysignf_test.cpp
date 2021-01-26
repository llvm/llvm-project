//===-- Unittests for copysignf -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/copysignf.h"
#include "utils/FPUtil/FPBits.h"
#include "utils/FPUtil/TestHelpers.h"
#include "utils/UnitTest/Test.h"
#include <math.h>

using FPBits = __llvm_libc::fputil::FPBits<float>;

DECLARE_SPECIAL_CONSTANTS(float)

TEST(LlvmLibcCopySinfTest, SpecialNumbers) {
  EXPECT_FP_EQ(aNaN, __llvm_libc::copysignf(aNaN, -1.0));
  EXPECT_FP_EQ(aNaN, __llvm_libc::copysignf(aNaN, 1.0));

  EXPECT_FP_EQ(negInf, __llvm_libc::copysignf(inf, -1.0));
  EXPECT_FP_EQ(inf, __llvm_libc::copysignf(negInf, 1.0));

  EXPECT_FP_EQ(negZero, __llvm_libc::copysignf(zero, -1.0));
  EXPECT_FP_EQ(zero, __llvm_libc::copysignf(negZero, 1.0));
}

TEST(LlvmLibcCopySinfTest, InFloatRange) {
  using UIntType = FPBits::UIntType;
  constexpr UIntType count = 1000000;
  constexpr UIntType step = UIntType(-1) / count;
  for (UIntType i = 0, v = 0; i <= count; ++i, v += step) {
    float x = FPBits(v);
    if (isnan(x) || isinf(x) || x == 0)
      continue;

    float res1 = __llvm_libc::copysignf(x, -x);
    ASSERT_FP_EQ(res1, -x);

    float res2 = __llvm_libc::copysignf(x, x);
    ASSERT_FP_EQ(res2, x);
  }
}

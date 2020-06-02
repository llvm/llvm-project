//===-- Unittests for copysignf
//--------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/math.h"
#include "src/math/copysignf.h"
#include "utils/FPUtil/BitPatterns.h"
#include "utils/FPUtil/FloatOperations.h"
#include "utils/FPUtil/FloatProperties.h"
#include "utils/UnitTest/Test.h"

using __llvm_libc::fputil::valueAsBits;
using __llvm_libc::fputil::valueFromBits;

using BitPatterns = __llvm_libc::fputil::BitPatterns<float>;
using Properties = __llvm_libc::fputil::FloatProperties<float>;

TEST(CopySignFTest, SpecialNumbers) {
  EXPECT_EQ(BitPatterns::aNegativeQuietNaN,
            valueAsBits(__llvm_libc::copysignf(
                valueFromBits(BitPatterns::aQuietNaN), -1.0f)));
  EXPECT_EQ(BitPatterns::aQuietNaN,
            valueAsBits(__llvm_libc::copysignf(
                valueFromBits(BitPatterns::aNegativeQuietNaN), 1.0f)));

  EXPECT_EQ(BitPatterns::aNegativeSignallingNaN,
            valueAsBits(__llvm_libc::copysignf(
                valueFromBits(BitPatterns::aSignallingNaN), -1.0f)));
  EXPECT_EQ(BitPatterns::aSignallingNaN,
            valueAsBits(__llvm_libc::copysignf(
                valueFromBits(BitPatterns::aNegativeSignallingNaN), 1.0f)));

  EXPECT_EQ(BitPatterns::negInf, valueAsBits(__llvm_libc::copysignf(
                                     valueFromBits(BitPatterns::inf), -1.0f)));
  EXPECT_EQ(BitPatterns::inf, valueAsBits(__llvm_libc::copysignf(
                                  valueFromBits(BitPatterns::negInf), 1.0f)));

  EXPECT_EQ(BitPatterns::negZero,
            valueAsBits(__llvm_libc::copysignf(valueFromBits(BitPatterns::zero),
                                               -1.0f)));
  EXPECT_EQ(BitPatterns::zero, valueAsBits(__llvm_libc::copysignf(
                                   valueFromBits(BitPatterns::negZero), 1.0f)));
}

TEST(CopySignFTest, InDoubleRange) {
  using BitsType = Properties::BitsType;
  constexpr BitsType count = 1000000;
  constexpr BitsType step = UINT32_MAX / count;
  for (BitsType i = 0, v = 0; i <= count; ++i, v += step) {
    float x = valueFromBits(v);
    if (isnan(x) || isinf(x) || x == 0)
      continue;

    float res1 = __llvm_libc::copysignf(x, -x);
    ASSERT_TRUE(res1 == -x);

    float res2 = __llvm_libc::copysignf(x, x);
    ASSERT_TRUE(res2 == x);
  }
}

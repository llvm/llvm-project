//===-- Unittests for sqrtf -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "src/math/sqrtf.h"
#include "utils/FPUtil/FPBits.h"
#include "utils/FPUtil/TestHelpers.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include <math.h>

using FPBits = __llvm_libc::fputil::FPBits<float>;
using UIntType = typename FPBits::UIntType;

namespace mpfr = __llvm_libc::testing::mpfr;

constexpr UIntType HiddenBit =
    UIntType(1) << __llvm_libc::fputil::MantissaWidth<float>::value;

DECLARE_SPECIAL_CONSTANTS(float)

TEST(SqrtfTest, SpecialValues) {
  ASSERT_FP_EQ(nan, __llvm_libc::sqrtf(nan));
  ASSERT_FP_EQ(inf, __llvm_libc::sqrtf(inf));
  ASSERT_FP_EQ(nan, __llvm_libc::sqrtf(negInf));
  ASSERT_FP_EQ(0.0f, __llvm_libc::sqrtf(0.0f));
  ASSERT_FP_EQ(-0.0f, __llvm_libc::sqrtf(-0.0f));
  ASSERT_FP_EQ(nan, __llvm_libc::sqrtf(-1.0f));
  ASSERT_FP_EQ(1.0f, __llvm_libc::sqrtf(1.0f));
  ASSERT_FP_EQ(2.0f, __llvm_libc::sqrtf(4.0f));
  ASSERT_FP_EQ(3.0f, __llvm_libc::sqrtf(9.0f));
}

TEST(SqrtfTest, DenormalValues) {
  for (UIntType mant = 1; mant < HiddenBit; mant <<= 1) {
    FPBits denormal(0.0f);
    denormal.mantissa = mant;

    ASSERT_MPFR_MATCH(mpfr::Operation::Sqrt, float(denormal),
                      __llvm_libc::sqrtf(denormal), 0.5);
  }

  constexpr UIntType count = 1'000'001;
  constexpr UIntType step = HiddenBit / count;
  for (UIntType i = 0, v = 0; i <= count; ++i, v += step) {
    float x = *reinterpret_cast<float *>(&v);
    ASSERT_MPFR_MATCH(mpfr::Operation::Sqrt, x, __llvm_libc::sqrtf(x), 0.5);
  }
}

TEST(SqrtfTest, InFloatRange) {
  constexpr UIntType count = 10'000'001;
  constexpr UIntType step = UIntType(-1) / count;
  for (UIntType i = 0, v = 0; i <= count; ++i, v += step) {
    float x = *reinterpret_cast<float *>(&v);
    if (isnan(x) || (x < 0)) {
      continue;
    }

    ASSERT_MPFR_MATCH(mpfr::Operation::Sqrt, x, __llvm_libc::sqrtf(x), 0.5);
  }
}

//===-- Unittests for sqrt -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "src/math/sqrt.h"
#include "utils/FPUtil/FPBits.h"
#include "utils/FPUtil/TestHelpers.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include <math.h>

using FPBits = __llvm_libc::fputil::FPBits<double>;
using UIntType = typename FPBits::UIntType;

namespace mpfr = __llvm_libc::testing::mpfr;

constexpr UIntType HiddenBit =
    UIntType(1) << __llvm_libc::fputil::MantissaWidth<double>::value;

DECLARE_SPECIAL_CONSTANTS(double)

TEST(LlvmLibcSqrtTest, SpecialValues) {
  ASSERT_FP_EQ(aNaN, __llvm_libc::sqrt(aNaN));
  ASSERT_FP_EQ(inf, __llvm_libc::sqrt(inf));
  ASSERT_FP_EQ(aNaN, __llvm_libc::sqrt(negInf));
  ASSERT_FP_EQ(0.0, __llvm_libc::sqrt(0.0));
  ASSERT_FP_EQ(-0.0, __llvm_libc::sqrt(-0.0));
  ASSERT_FP_EQ(aNaN, __llvm_libc::sqrt(-1.0));
  ASSERT_FP_EQ(1.0, __llvm_libc::sqrt(1.0));
  ASSERT_FP_EQ(2.0, __llvm_libc::sqrt(4.0));
  ASSERT_FP_EQ(3.0, __llvm_libc::sqrt(9.0));
}

TEST(LlvmLibcSqrtTest, DenormalValues) {
  for (UIntType mant = 1; mant < HiddenBit; mant <<= 1) {
    FPBits denormal(0.0);
    denormal.mantissa = mant;

    ASSERT_MPFR_MATCH(mpfr::Operation::Sqrt, double(denormal),
                      __llvm_libc::sqrt(denormal), 0.5);
  }

  constexpr UIntType count = 1'000'001;
  constexpr UIntType step = HiddenBit / count;
  for (UIntType i = 0, v = 0; i <= count; ++i, v += step) {
    double x = *reinterpret_cast<double *>(&v);
    ASSERT_MPFR_MATCH(mpfr::Operation::Sqrt, x, __llvm_libc::sqrt(x), 0.5);
  }
}

TEST(LlvmLibcSqrtTest, InDoubleRange) {
  constexpr UIntType count = 10'000'001;
  constexpr UIntType step = UIntType(-1) / count;
  for (UIntType i = 0, v = 0; i <= count; ++i, v += step) {
    double x = *reinterpret_cast<double *>(&v);
    if (isnan(x) || (x < 0)) {
      continue;
    }

    ASSERT_MPFR_MATCH(mpfr::Operation::Sqrt, x, __llvm_libc::sqrt(x), 0.5);
  }
}

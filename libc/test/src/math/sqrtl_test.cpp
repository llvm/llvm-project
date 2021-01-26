//===-- Unittests for sqrtl ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "src/math/sqrtl.h"
#include "utils/FPUtil/FPBits.h"
#include "utils/FPUtil/TestHelpers.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include <math.h>

using FPBits = __llvm_libc::fputil::FPBits<long double>;
using UIntType = typename FPBits::UIntType;

namespace mpfr = __llvm_libc::testing::mpfr;

constexpr UIntType HiddenBit =
    UIntType(1) << __llvm_libc::fputil::MantissaWidth<long double>::value;

DECLARE_SPECIAL_CONSTANTS(long double)

TEST(LlvmLibcSqrtlTest, SpecialValues) {
  ASSERT_FP_EQ(aNaN, __llvm_libc::sqrtl(aNaN));
  ASSERT_FP_EQ(inf, __llvm_libc::sqrtl(inf));
  ASSERT_FP_EQ(aNaN, __llvm_libc::sqrtl(negInf));
  ASSERT_FP_EQ(0.0L, __llvm_libc::sqrtl(0.0L));
  ASSERT_FP_EQ(-0.0L, __llvm_libc::sqrtl(-0.0L));
  ASSERT_FP_EQ(aNaN, __llvm_libc::sqrtl(-1.0L));
  ASSERT_FP_EQ(1.0L, __llvm_libc::sqrtl(1.0L));
  ASSERT_FP_EQ(2.0L, __llvm_libc::sqrtl(4.0L));
  ASSERT_FP_EQ(3.0L, __llvm_libc::sqrtl(9.0L));
}

TEST(LlvmLibcSqrtlTest, DenormalValues) {
  for (UIntType mant = 1; mant < HiddenBit; mant <<= 1) {
    FPBits denormal(0.0L);
    denormal.mantissa = mant;

    ASSERT_MPFR_MATCH(mpfr::Operation::Sqrt, static_cast<long double>(denormal),
                      __llvm_libc::sqrtl(denormal), 0.5);
  }

  constexpr UIntType count = 1'000'001;
  constexpr UIntType step = HiddenBit / count;
  for (UIntType i = 0, v = 0; i <= count; ++i, v += step) {
    long double x = *reinterpret_cast<long double *>(&v);
    ASSERT_MPFR_MATCH(mpfr::Operation::Sqrt, x, __llvm_libc::sqrtl(x), 0.5);
  }
}

TEST(LlvmLibcSqrtlTest, InLongDoubleRange) {
  constexpr UIntType count = 10'000'001;
  constexpr UIntType step = UIntType(-1) / count;
  for (UIntType i = 0, v = 0; i <= count; ++i, v += step) {
    long double x = *reinterpret_cast<long double *>(&v);
    if (isnan(x) || (x < 0)) {
      continue;
    }

    ASSERT_MPFR_MATCH(mpfr::Operation::Sqrt, x, __llvm_libc::sqrtl(x), 0.5);
  }
}

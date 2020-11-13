//===-- Unittests for hypotf ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/math.h"
#include "src/math/hypotf.h"
#include "utils/FPUtil/FPBits.h"
#include "utils/FPUtil/TestHelpers.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include "utils/UnitTest/Test.h"

using FPBits = __llvm_libc::fputil::FPBits<float>;
using UIntType = FPBits::UIntType;

namespace mpfr = __llvm_libc::testing::mpfr;

DECLARE_SPECIAL_CONSTANTS(float)

TEST(HypotfTest, SpecialNumbers) {
  EXPECT_FP_EQ(__llvm_libc::hypotf(inf, nan), inf);
  EXPECT_FP_EQ(__llvm_libc::hypotf(nan, negInf), inf);
  EXPECT_FP_EQ(__llvm_libc::hypotf(zero, inf), inf);
  EXPECT_FP_EQ(__llvm_libc::hypotf(negInf, negZero), inf);

  EXPECT_FP_EQ(__llvm_libc::hypotf(nan, nan), nan);
  EXPECT_FP_EQ(__llvm_libc::hypotf(nan, zero), nan);
  EXPECT_FP_EQ(__llvm_libc::hypotf(negZero, nan), nan);

  EXPECT_FP_EQ(__llvm_libc::hypotf(negZero, zero), zero);
}

TEST(HypotfTest, SubnormalRange) {
  constexpr UIntType count = 1000001;
  constexpr UIntType step =
      (FPBits::maxSubnormal - FPBits::minSubnormal) / count;
  for (UIntType v = FPBits::minSubnormal, w = FPBits::maxSubnormal;
       v <= FPBits::maxSubnormal && w >= FPBits::minSubnormal;
       v += step, w -= step) {
    float x = FPBits(v), y = FPBits(w);
    float result = __llvm_libc::hypotf(x, y);
    mpfr::BinaryInput<float> input{x, y};
    ASSERT_MPFR_MATCH(mpfr::Operation::Hypot, input, result, 0.5);
  }
}

TEST(HypotfTest, NormalRange) {
  constexpr UIntType count = 1000001;
  constexpr UIntType step = (FPBits::maxNormal - FPBits::minNormal) / count;
  for (UIntType v = FPBits::minNormal, w = FPBits::maxNormal;
       v <= FPBits::maxNormal && w >= FPBits::minNormal; v += step, w -= step) {
    float x = FPBits(v), y = FPBits(w);
    float result = __llvm_libc::hypotf(x, y);
    ;
    mpfr::BinaryInput<float> input{x, y};
    ASSERT_MPFR_MATCH(mpfr::Operation::Hypot, input, result, 0.5);
  }
}

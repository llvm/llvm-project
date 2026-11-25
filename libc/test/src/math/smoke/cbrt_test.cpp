//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Smoke tests for cbrt.
///
//===----------------------------------------------------------------------===//

#include "hdr/fenv_macros.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/math/cbrt.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcCbrtTest = LIBC_NAMESPACE::testing::FPTest<double>;

TEST_F(LlvmLibcCbrtTest, SpecialNumbers) {
  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::cbrt(sNaN), FE_INVALID);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::cbrt(aNaN));
  EXPECT_FP_EQ_ALL_ROUNDING(inf, LIBC_NAMESPACE::cbrt(inf));
  EXPECT_FP_EQ_ALL_ROUNDING(neg_inf, LIBC_NAMESPACE::cbrt(neg_inf));
  EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::cbrt(zero));
  EXPECT_FP_EQ_ALL_ROUNDING(neg_zero, LIBC_NAMESPACE::cbrt(neg_zero));
  EXPECT_FP_EQ_ALL_ROUNDING(1.0, LIBC_NAMESPACE::cbrt(1.0));
  EXPECT_FP_EQ_ALL_ROUNDING(-1.0, LIBC_NAMESPACE::cbrt(-1.0));
  EXPECT_FP_EQ_ALL_ROUNDING(2.0, LIBC_NAMESPACE::cbrt(8.0));
  EXPECT_FP_EQ_ALL_ROUNDING(-2.0, LIBC_NAMESPACE::cbrt(-8.0));
  EXPECT_FP_EQ_ALL_ROUNDING(3.0, LIBC_NAMESPACE::cbrt(27.0));
  EXPECT_FP_EQ_ALL_ROUNDING(-3.0, LIBC_NAMESPACE::cbrt(-27.0));
  EXPECT_FP_EQ_ALL_ROUNDING(5.0, LIBC_NAMESPACE::cbrt(125.0));
  EXPECT_FP_EQ_ALL_ROUNDING(-5.0, LIBC_NAMESPACE::cbrt(-125.0));
  EXPECT_FP_EQ_ALL_ROUNDING(0x1.0p42, LIBC_NAMESPACE::cbrt(0x1.0p126));
  EXPECT_FP_EQ_ALL_ROUNDING(-0x1.0p42, LIBC_NAMESPACE::cbrt(-0x1.0p126));
  EXPECT_FP_EQ_ALL_ROUNDING(0x1.0p341, LIBC_NAMESPACE::cbrt(0x1.0p1023));
  EXPECT_FP_EQ_ALL_ROUNDING(-0x1.0p341, LIBC_NAMESPACE::cbrt(-0x1.0p1023));
  EXPECT_FP_EQ(-0x1.0p-340, LIBC_NAMESPACE::cbrt(-0x1.fffffffffffffp-1021));
  EXPECT_FP_EQ(2.0, LIBC_NAMESPACE::cbrt(0x1.fffffffffffffp2));

#ifndef LIBC_MATH_HAS_SKIP_ACCURATE_PASS
  // Exact cubes must not raise FE_INEXACT, and must preserve a pre-existing
  // FE_INEXACT flag across all rounding modes.
  constexpr double EXACT_CUBES[][2] = {
      {1.0, 1.0},
      {8.0, 2.0},
      {27.0, 3.0},
      {0x1.577f9b5ep+1, 0x1.63cp+0},
      {0x1.ca54bee7202cp+1, 0x1.87a6p+0},
  };
  for (const auto &tc : EXACT_CUBES) {
    double x = LIBC_NAMESPACE::testing::opaque(tc[0]);
    double expected = tc[1];
    EXPECT_FP_EQ_WITH_EXCEPTION_ALL_ROUNDING(expected, LIBC_NAMESPACE::cbrt(x),
                                             0);
    EXPECT_FP_EQ_WITH_EXCEPTION_ALL_ROUNDING(-expected,
                                             LIBC_NAMESPACE::cbrt(-x), 0);

    for (auto rm : ROUNDING_MODES) {
      LIBC_NAMESPACE::fputil::testing::ForceRoundingMode r(rm);
      if (!r.success)
        continue;
      LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
      LIBC_NAMESPACE::fputil::raise_except_if_required(FE_INEXACT);
      EXPECT_FP_EQ(expected, LIBC_NAMESPACE::cbrt(x));
      EXPECT_FP_EXCEPTION(FE_INEXACT);
    }
  }

  // Non-exact neighbor rounding to an exact cube (1.0) must still raise
  // FE_INEXACT.
  EXPECT_FP_EQ_WITH_EXCEPTION_ROUNDING_NEAREST(
      1.0, LIBC_NAMESPACE::cbrt(0x1.0000000000001p+0), FE_INEXACT);
#endif // !LIBC_MATH_HAS_SKIP_ACCURATE_PASS
}

#ifdef LIBC_TEST_FTZ_DAZ

using namespace LIBC_NAMESPACE::testing;

TEST_F(LlvmLibcCbrtTest, FTZMode) {
  ModifyMXCSR mxcsr(FTZ);

  EXPECT_FP_EQ(0x1.0p-358, LIBC_NAMESPACE::cbrt(min_denormal));
  EXPECT_FP_EQ(0x1.428a2f98d728ap-341, LIBC_NAMESPACE::cbrt(max_denormal));
}

TEST_F(LlvmLibcCbrtTest, DAZMode) {
  ModifyMXCSR mxcsr(DAZ);

  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::cbrt(min_denormal));
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::cbrt(max_denormal));
}

TEST_F(LlvmLibcCbrtTest, FTZDAZMode) {
  ModifyMXCSR mxcsr(FTZ | DAZ);

  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::cbrt(min_denormal));
  EXPECT_FP_EQ(0.0, LIBC_NAMESPACE::cbrt(max_denormal));
}

#endif

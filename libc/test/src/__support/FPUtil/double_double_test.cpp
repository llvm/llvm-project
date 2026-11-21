//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for LIBC_DO_NOT_OPTIMIZE and double-double utilities.
///
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/double_double.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h"
#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/RoundingModeUtils.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::fputil::DoubleDouble;
using LIBC_NAMESPACE::fputil::testing::ForceRoundingMode;
using LIBC_NAMESPACE::fputil::testing::RoundingMode;

namespace {

#if LIBC_HAS_CONSTANT_EVALUATION
constexpr float constexpr_barrier_check_f(float x) {
  LIBC_DO_NOT_OPTIMIZE(x);
  LIBC_DO_NOT_OPTIMIZE_ROUNDING(x);
  return x + 1.0f;
}
constexpr double constexpr_barrier_check_d(double x) {
  LIBC_DO_NOT_OPTIMIZE(x);
  LIBC_DO_NOT_OPTIMIZE_ROUNDING(x);
  return x + 1.0;
}
static_assert(constexpr_barrier_check_f(2.5f) == 3.5f);
static_assert(constexpr_barrier_check_d(2.5) == 3.5);
#endif

} // namespace

class LlvmLibcDoubleDoubleTest : public LIBC_NAMESPACE::testing::FEnvSafeTest {
};

TEST_F(LlvmLibcDoubleDoubleTest, DoNotOptimizeFloatAndDouble) {
  float f = 1.5f;
  LIBC_DO_NOT_OPTIMIZE(f);
  LIBC_DO_NOT_OPTIMIZE_ROUNDING(f);
  EXPECT_FP_EQ(f, 1.5f);

  double d = -2.25;
  LIBC_DO_NOT_OPTIMIZE(d);
  LIBC_DO_NOT_OPTIMIZE_ROUNDING(d);
  EXPECT_FP_EQ(d, -2.25);
}

#if !defined(LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY) &&                       \
    !defined(LIBC_COMPILER_IS_MSVC)
TEST_F(LlvmLibcDoubleDoubleTest, DoNotOptimizeBlocksConstantFolding) {
  {
    ForceRoundingMode r(RoundingMode::Upward);
    if (r.success) {
      float one_f = 1.0f;
      LIBC_DO_NOT_OPTIMIZE(one_f);
      EXPECT_FP_EQ(one_f + 0x1.0p-50f, 0x1.000002p0f);

      double one_d = 1.0;
      LIBC_DO_NOT_OPTIMIZE_ROUNDING(one_d);
      EXPECT_FP_EQ(one_d + 0x1.0p-100, 0x1.0000000000001p0);
    }
  }
  {
    ForceRoundingMode r(RoundingMode::Downward);
    if (r.success) {
      float neg_one_f = -1.0f;
      LIBC_DO_NOT_OPTIMIZE(neg_one_f);
      EXPECT_FP_EQ(neg_one_f - 0x1.0p-50f, -0x1.000002p0f);

      double neg_one_d = -1.0;
      LIBC_DO_NOT_OPTIMIZE_ROUNDING(neg_one_d);
      EXPECT_FP_EQ(neg_one_d - 0x1.0p-100, -0x1.0000000000001p0);
    }
  }
}
#endif

TEST_F(LlvmLibcDoubleDoubleTest, ExactMultDirectedRounding) {
  if constexpr (LIBC_NAMESPACE::fputil::TargetHasFmaInstruction<
                    double>::VALUE) {
    // Exact product: (1 + 2^-52)^2 = (1 + 2^-51) + 2^-104
    //                              = (1 + 3*2^-52) + (-2^-52 + 2^-104).
    auto check_square = [](RoundingMode mode, double expected_hi,
                           double expected_lo) {
      ForceRoundingMode r(mode);
      if (!r.success)
        return;
      double a = 1.0 + 0x1.0p-52;
      LIBC_DO_NOT_OPTIMIZE(a);
      DoubleDouble prod = LIBC_NAMESPACE::fputil::exact_mult(a, a);
      EXPECT_FP_EQ(prod.hi, expected_hi);
      EXPECT_FP_EQ(prod.lo, expected_lo);
    };

    check_square(RoundingMode::Nearest, 1.0 + 0x1.0p-51, 0x1.0p-104);
#ifndef LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
    check_square(RoundingMode::Upward, 1.0 + 0x1.8p-51,
                 -0x1.0p-52 + 0x1.0p-104);
    check_square(RoundingMode::Downward, 1.0 + 0x1.0p-51, 0x1.0p-104);
    check_square(RoundingMode::TowardZero, 1.0 + 0x1.0p-51, 0x1.0p-104);
#endif

    // Four-way exact_mult + exact_add<false> cancellation pattern (as in
    // sincos, b/586440175):
    //   p0 = (1 + 2^-52) * (1 + 2^-52)   =  1 + 2^-51 + 2^-104
    //   p1 = (1 + 2^-51) * -(1 + 2^-51)  = -1 - 2^-50 - 2^-102
    //   p0 + p1 = -(2^-51 + 1.5 * 2^-103) (exact midpoint in ulp(2^-51))
    //   p2 = (1 + 2^-50) * (1 + 2^-50)   =  1 + 2^-49 + 2^-100
    //   p3 = (1 + 2^-49) * -(1 + 2^-49)  = -1 - 2^-48 - 2^-98
    //   p2 + p3 = -(2^-49 + 3 * 2^-100)  (exact binary64:
    //   -0x1.0000000000006p-49)
    auto check_four_way = [](RoundingMode mode, double expected_s0,
                             double expected_s1) {
      ForceRoundingMode r(mode);
      if (!r.success)
        return;

      double a0 = 1.0 + 0x1.0p-52;
      double b0 = 1.0 + 0x1.0p-52;
      double a1 = 1.0 + 0x1.0p-51;
      double b1 = -(1.0 + 0x1.0p-51);
      double a2 = 1.0 + 0x1.0p-50;
      double b2 = 1.0 + 0x1.0p-50;
      double a3 = 1.0 + 0x1.0p-49;
      double b3 = -(1.0 + 0x1.0p-49);
      LIBC_DO_NOT_OPTIMIZE(a0);
      LIBC_DO_NOT_OPTIMIZE(b0);
      LIBC_DO_NOT_OPTIMIZE(a1);
      LIBC_DO_NOT_OPTIMIZE(b1);
      LIBC_DO_NOT_OPTIMIZE(a2);
      LIBC_DO_NOT_OPTIMIZE(b2);
      LIBC_DO_NOT_OPTIMIZE(a3);
      LIBC_DO_NOT_OPTIMIZE(b3);

      DoubleDouble p0 = LIBC_NAMESPACE::fputil::exact_mult(a0, b0);
      DoubleDouble p1 = LIBC_NAMESPACE::fputil::exact_mult(a1, b1);
      DoubleDouble p2 = LIBC_NAMESPACE::fputil::exact_mult(a2, b2);
      DoubleDouble p3 = LIBC_NAMESPACE::fputil::exact_mult(a3, b3);

      DoubleDouble s0 = LIBC_NAMESPACE::fputil::exact_add<false>(p0.hi, p1.hi);
      DoubleDouble s1 = LIBC_NAMESPACE::fputil::exact_add<false>(p2.hi, p3.hi);
      s0.lo += p0.lo + p1.lo;
      s1.lo += p2.lo + p3.lo;

      EXPECT_FP_EQ(s0.hi + s0.lo, expected_s0);
      EXPECT_FP_EQ(s1.hi + s1.lo, expected_s1);
    };

    check_four_way(RoundingMode::Nearest, -0x1.0000000000002p-51,
                   -0x1.0000000000006p-49);
#ifndef LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
    check_four_way(RoundingMode::Upward, -0x1.0000000000001p-51,
                   -0x1.0000000000006p-49);
    check_four_way(RoundingMode::Downward, -0x1.0000000000002p-51,
                   -0x1.0000000000006p-49);
    check_four_way(RoundingMode::TowardZero, -0x1.0000000000001p-51,
                   -0x1.0000000000006p-49);
#endif
  }
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the unit tests for statically-rounded implementation of
/// static_rounding::exp(x)
///
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "hdr/stdint_proxy.h"
#include "shared/static_rounding_math.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/math/exp.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/RoundingModeUtils.h"
#include "test/UnitTest/Test.h"

using LlvmLibcExpStaticRoundingTest = LIBC_NAMESPACE::testing::FPTest<double>;
using RoundingMode = LIBC_NAMESPACE::fputil::testing::RoundingMode;
using ForceRoundingMode = LIBC_NAMESPACE::fputil::testing::ForceRoundingMode;
using LIBC_NAMESPACE::testing::tlog;

namespace static_rounding = LIBC_NAMESPACE::shared::math::static_rounding;
namespace math = LIBC_NAMESPACE::math;

TEST_F(LlvmLibcExpStaticRoundingTest, SpecialNumbers) {
  constexpr double VALUES[] = {aNaN,     inf,  neg_inf, -0x1.0p20,
                               0x1.0p20, zero, neg_zero};

  for (auto rounding : ROUNDING_MODES) {
    const int fenv_rounding = get_fe_rounding(rounding);

    // Statically rounded exp doesn't raise exceptions
    for (auto x : VALUES) {
      EXPECT_FP_EQ_ROUNDING_MODE(
          math::exp(x), static_rounding::exp(x, fenv_rounding), rounding);
    }
  }
}

TEST_F(LlvmLibcExpStaticRoundingTest, TrickyInputs) {
  constexpr uint64_t VALUES[] = {
      0x3FD79289C6E6A5C0,
      0x3FD05DE80A173EA0, // 0x1.05de80a173eap-2
      0xbf1eb7a4cb841fcc, // -0x1.eb7a4cb841fccp-14
      0xbf19a61fb925970d,
      0x3fda7b764e2cf47a, // 0x1.a7b764e2cf47ap-2
      0xc04757852a4b93aa, // -0x1.757852a4b93aap+5
      0x4044c19e5712e377, // x=0x1.4c19e5712e377p+5
      0xbf19a61fb925970d, // x=-0x1.9a61fb925970dp-14
      0xc039a74cdab36c28, // x=-0x1.9a74cdab36c28p+4
      0xc085b3e4e2e3bba9, // x=-0x1.5b3e4e2e3bba9p+9
      0xc086960d591aec34, // x=-0x1.6960d591aec34p+9
      0xc086232c09d58d91, // x=-0x1.6232c09d58d91p+9
      0xc0874910d52d3051, // x=-0x1.74910d52d3051p9
      0xc0867a172ceb0990, // x=-0x1.67a172ceb099p+9
  };

  for (auto rounding : ROUNDING_MODES) {
    const int fenv_rounding = get_fe_rounding(rounding);

    // Statically rounded exp doesn't raise exceptions
    for (auto val : VALUES) {
      double x = FPBits(val).get_val();
      EXPECT_FP_EQ_ROUNDING_MODE(
          math::exp(x), static_rounding::exp(x, fenv_rounding), rounding);
    }
  }
}

// TODO: double has a very big range. Define test functions that allows for
// finding max ULPs over a range. EXPECT_* just isn't enough for this.
TEST_F(LlvmLibcExpStaticRoundingTest, InDoubleRange) {
  constexpr uint64_t COUNT = 1'231;
  constexpr uint64_t START = FPBits(0.25).uintval();
  constexpr uint64_t STOP = FPBits(4.0).uintval();
  constexpr uint64_t STEP = (STOP - START) / COUNT;

  auto test = [&](RoundingMode rounding) {
    const int fenv_rounding = get_fe_rounding(rounding);

    uint64_t fails = 0;
    uint64_t count = 0;
    uint64_t cc = 0;
    double mx, mr = 0.0;
    double tol = 0.5;

    for (uint64_t i = 0, v = START; i <= COUNT; ++i, v += STEP) {
      double x = FPBits(v).get_val();
      if (FPBits(v).is_nan() || FPBits(v).is_inf() || x < 0.0)
        continue;
      double result = math::exp(x);
      ++cc;
      if (FPBits(result).is_nan() || FPBits(result).is_inf())
        continue;

      ++count;
      // ASSERT_MPFR_MATCH(mpfr::Operation::Log, x, result, 0.5);
      // if (!TEST_MPFR_MATCH_ROUNDING_SILENTLY(mpfr::Operation::Exp, x, result,
      //                                        TOLERANCE + 0.5, rounding_mode))
      //                                        {
      //   ++fails;
      //   while (!TEST_MPFR_MATCH_ROUNDING_SILENTLY(mpfr::Operation::Exp, x,
      //                                             result, tol,
      //                                             rounding_mode)) {
      //     mx = x;
      //     mr = result;

      //     if (tol > 1000.0)
      //       break;

      //     tol *= 2.0;
      //   }
      // }
    }
    if (fails) {
      tlog << " Statically rounded exp failed: " << fails << "/" << count << "/"
           << cc << " tests.\n";
      tlog << "   Max ULPs is at most: " << static_cast<uint64_t>(tol) << ".\n";
      // EXPECT_MPFR_MATCH(mpfr::Operation::Exp, mx, mr, 0.5, rounding_mode);
    }
  };

  tlog << " Test Rounding To Nearest...\n";
  test(RoundingMode::Nearest);

  tlog << " Test Rounding Downward...\n";
  test(RoundingMode::Downward);

  tlog << " Test Rounding Upward...\n";
  test(RoundingMode::Upward);

  tlog << " Test Rounding Toward Zero...\n";
  test(RoundingMode::TowardZero);
}

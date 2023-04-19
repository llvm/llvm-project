//===-- Unittests for log10 -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/errno/libc_errno.h"
#include "src/math/log10.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include <math.h>

#include <errno.h>
#include <stdint.h>

namespace mpfr = __llvm_libc::testing::mpfr;

DECLARE_SPECIAL_CONSTANTS(double)

TEST(LlvmLibcLog10Test, SpecialNumbers) {
  EXPECT_FP_EQ(aNaN, __llvm_libc::log10(aNaN));
  EXPECT_FP_EQ(inf, __llvm_libc::log10(inf));
  EXPECT_TRUE(FPBits(__llvm_libc::log10(neg_inf)).is_nan());
  EXPECT_FP_EQ(neg_inf, __llvm_libc::log10(0.0));
  EXPECT_FP_EQ(neg_inf, __llvm_libc::log10(-0.0));
  EXPECT_TRUE(FPBits(__llvm_libc::log10(-1.0)).is_nan());
  EXPECT_FP_EQ(zero, __llvm_libc::log10(1.0));
}

TEST(LlvmLibcLog10Test, TrickyInputs) {
  constexpr int N = 27;
  constexpr uint64_t INPUTS[N] = {
      0x3ff0000000000000, // x = 1.0
      0x4024000000000000, // x = 10.0
      0x4059000000000000, // x = 10^2
      0x408f400000000000, // x = 10^3
      0x40c3880000000000, // x = 10^4
      0x40f86a0000000000, // x = 10^5
      0x412e848000000000, // x = 10^6
      0x416312d000000000, // x = 10^7
      0x4197d78400000000, // x = 10^8
      0x41cdcd6500000000, // x = 10^9
      0x4202a05f20000000, // x = 10^10
      0x42374876e8000000, // x = 10^11
      0x426d1a94a2000000, // x = 10^12
      0x42a2309ce5400000, // x = 10^13
      0x42d6bcc41e900000, // x = 10^14
      0x430c6bf526340000, // x = 10^15
      0x4341c37937e08000, // x = 10^16
      0x4376345785d8a000, // x = 10^17
      0x43abc16d674ec800, // x = 10^18
      0x43e158e460913d00, // x = 10^19
      0x4415af1d78b58c40, // x = 10^20
      0x444b1ae4d6e2ef50, // x = 10^21
      0x4480f0cf064dd592, // x = 10^22
      0x3fefffffffef06ad, 0x3fefde0f22c7d0eb,
      0x225e7812faadb32f, 0x3fee1076964c2903,
  };
  for (int i = 0; i < N; ++i) {
    double x = double(FPBits(INPUTS[i]));
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Log10, x,
                                   __llvm_libc::log10(x), 0.5);
  }
}

TEST(LlvmLibcLog10Test, InDoubleRange) {
  constexpr uint64_t COUNT = 1234561;
  constexpr uint64_t STEP = 0x3FF0'0000'0000'0000ULL / COUNT;

  auto test = [&](mpfr::RoundingMode rounding_mode) {
    mpfr::ForceRoundingMode __r(rounding_mode);
    uint64_t fails = 0;
    uint64_t count = 0;
    uint64_t cc = 0;
    double mx, mr = 0.0;
    double tol = 0.5;

    for (uint64_t i = 0, v = 0; i <= COUNT; ++i, v += STEP) {
      double x = FPBits(v).get_val();
      if (isnan(x) || isinf(x) || x < 0.0)
        continue;
      libc_errno = 0;
      double result = __llvm_libc::log10(x);
      ++cc;
      if (isnan(result))
        continue;

      ++count;
      // ASSERT_MPFR_MATCH(mpfr::Operation::Log10, x, result, 0.5);
      if (!EXPECT_MPFR_MATCH_ROUNDING_SILENTLY(mpfr::Operation::Log10, x,
                                               result, 0.5, rounding_mode)) {
        ++fails;
        while (!EXPECT_MPFR_MATCH_ROUNDING_SILENTLY(
            mpfr::Operation::Log10, x, result, tol, rounding_mode)) {
          mx = x;
          mr = result;
          tol *= 2.0;
        }
      }
    }
    if (fails) {
      EXPECT_MPFR_MATCH(mpfr::Operation::Log10, mx, mr, 0.5, rounding_mode);
    }
  };

  test(mpfr::RoundingMode::Nearest);

  test(mpfr::RoundingMode::Downward);

  test(mpfr::RoundingMode::Upward);

  test(mpfr::RoundingMode::TowardZero);
}

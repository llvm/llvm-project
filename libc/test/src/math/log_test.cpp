//===-- Unittests for log -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/errno/libc_errno.h"
#include "src/math/log.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include <math.h>

#include <errno.h>
#include <stdint.h>

using LlvmLibcLogTest = LIBC_NAMESPACE::testing::FPTest<double>;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;
using LIBC_NAMESPACE::testing::tlog;

TEST_F(LlvmLibcLogTest, SpecialNumbers) {
  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::log(aNaN));
  EXPECT_FP_EQ(inf, LIBC_NAMESPACE::log(inf));
  EXPECT_FP_IS_NAN_WITH_EXCEPTION(LIBC_NAMESPACE::log(neg_inf), FE_INVALID);
  EXPECT_FP_EQ_WITH_EXCEPTION(neg_inf, LIBC_NAMESPACE::log(0.0), FE_DIVBYZERO);
  EXPECT_FP_EQ_WITH_EXCEPTION(neg_inf, LIBC_NAMESPACE::log(-0.0), FE_DIVBYZERO);
  EXPECT_FP_IS_NAN_WITH_EXCEPTION(LIBC_NAMESPACE::log(-1.0), FE_INVALID);
  EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::log(1.0));
}

TEST_F(LlvmLibcLogTest, TrickyInputs) {
  constexpr int N = 30;
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
      0x3fefffffffef06ad, 0x3fefde0f22c7d0eb, 0x225e7812faadb32f,
      0x3fee1076964c2903, 0x3fdfe93fff7fceb0, 0x3ff012631ad8df10,
      0x3fefbfdaa448ed98,
  };
  for (int i = 0; i < N; ++i) {
    double x = FPBits(INPUTS[i]).get_val();
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Log, x,
                                   LIBC_NAMESPACE::log(x), 0.5);
  }
}

TEST_F(LlvmLibcLogTest, AllExponents) {
  double x = 0x1.0p-1074;
  for (int i = -1074; i < 1024; ++i, x *= 2.0) {
    ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Log, x,
                                   LIBC_NAMESPACE::log(x), 0.5);
  }
}

TEST_F(LlvmLibcLogTest, InDoubleRange) {
  constexpr uint64_t COUNT = 234561;
  constexpr uint64_t START = 0x3FD0'0000'0000'0000ULL; // 0.25
  constexpr uint64_t STOP = 0x4010'0000'0000'0000ULL;  // 4.0
  // constexpr uint64_t START = 0x3FF0'0000'0000'0000ULL;  // 1.0
  // constexpr uint64_t STOP = 0x4000'0000'0000'0000ULL;   // 2.0
  constexpr uint64_t STEP = (STOP - START) / COUNT;

  auto test = [&](mpfr::RoundingMode rounding_mode) {
    mpfr::ForceRoundingMode __r(rounding_mode);
    if (!__r.success)
      return;

    uint64_t fails = 0;
    uint64_t count = 0;
    uint64_t cc = 0;
    double mx, mr = 0.0;
    double tol = 0.5;

    for (uint64_t i = 0, v = START; i <= COUNT; ++i, v += STEP) {
      double x = FPBits(v).get_val();
      if (isnan(x) || isinf(x) || x < 0.0)
        continue;
      libc_errno = 0;
      double result = LIBC_NAMESPACE::log(x);
      ++cc;
      if (isnan(result) || isinf(result))
        continue;

      ++count;
      // ASSERT_MPFR_MATCH(mpfr::Operation::Log, x, result, 0.5);
      if (!TEST_MPFR_MATCH_ROUNDING_SILENTLY(mpfr::Operation::Log, x, result,
                                             0.5, rounding_mode)) {
        ++fails;
        while (!TEST_MPFR_MATCH_ROUNDING_SILENTLY(mpfr::Operation::Log, x,
                                                  result, tol, rounding_mode)) {
          mx = x;
          mr = result;
          tol *= 2.0;
        }
      }
    }
    tlog << " Log failed: " << fails << "/" << count << "/" << cc
         << " tests.\n";
    tlog << "   Max ULPs is at most: " << static_cast<uint64_t>(tol) << ".\n";
    if (fails) {
      EXPECT_MPFR_MATCH(mpfr::Operation::Log, mx, mr, 0.5, rounding_mode);
    }
  };

  tlog << " Test Rounding To Nearest...\n";
  test(mpfr::RoundingMode::Nearest);

  tlog << " Test Rounding Downward...\n";
  test(mpfr::RoundingMode::Downward);

  tlog << " Test Rounding Upward...\n";
  test(mpfr::RoundingMode::Upward);

  tlog << " Test Rounding Toward Zero...\n";
  test(mpfr::RoundingMode::TowardZero);
}

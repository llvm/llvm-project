//===-- Unittests for 10^x ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/errno/libc_errno.h"
#include "src/math/exp10.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include <math.h>

#include <errno.h>
#include <stdint.h>

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;
using LIBC_NAMESPACE::testing::tlog;

DECLARE_SPECIAL_CONSTANTS(double)

TEST(LlvmLibcExp10Test, SpecialNumbers) {
  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::exp10(aNaN));
  EXPECT_FP_EQ(inf, LIBC_NAMESPACE::exp10(inf));
  EXPECT_FP_EQ_ALL_ROUNDING(zero, LIBC_NAMESPACE::exp10(neg_inf));
  EXPECT_FP_EQ_WITH_EXCEPTION(zero, LIBC_NAMESPACE::exp10(-0x1.0p20),
                              FE_UNDERFLOW);
  EXPECT_FP_EQ_WITH_EXCEPTION(inf, LIBC_NAMESPACE::exp10(0x1.0p20),
                              FE_OVERFLOW);
  EXPECT_FP_EQ_ALL_ROUNDING(1.0, LIBC_NAMESPACE::exp10(0.0));
  EXPECT_FP_EQ_ALL_ROUNDING(1.0, LIBC_NAMESPACE::exp10(-0.0));
}

TEST(LlvmLibcExp10Test, TrickyInputs) {
  constexpr int N = 41;
  constexpr uint64_t INPUTS[N] = {
      0x40033093317082F8, 0x3FD79289C6E6A5C0,
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
      0xc08ff80000000000, // x=-0x1.ff8p+9
      0xbc971547652b82fe, // x=-0x1.71547652b82fep-54
      0x0000000000000000, // x = 0
      0x3ff0000000000000, // x = 1
      0x4000000000000000, // x = 2
      0x4008000000000000, // x = 3
      0x4010000000000000, // x = 4
      0x4014000000000000, // x = 5
      0x4018000000000000, // x = 6
      0x401c000000000000, // x = 7
      0x4020000000000000, // x = 8
      0x4022000000000000, // x = 9
      0x4024000000000000, // x = 10
      0x4026000000000000, // x = 11
      0x4028000000000000, // x = 12
      0x402a000000000000, // x = 13
      0x402c000000000000, // x = 14
      0x402e000000000000, // x = 15
      0x4030000000000000, // x = 16
      0x4031000000000000, // x = 17
      0x4032000000000000, // x = 18
      0x4033000000000000, // x = 19
      0x4034000000000000, // x = 20
      0x4035000000000000, // x = 21
      0x4036000000000000, // x = 22
      0x4037000000000000, // x = 23
  };
  for (int i = 0; i < N; ++i) {
    double x = double(FPBits(INPUTS[i]));
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp10, x,
                                   LIBC_NAMESPACE::exp10(x), 0.5);
  }
}

TEST(LlvmLibcExp10Test, InDoubleRange) {
  constexpr uint64_t COUNT = 1'231;
  uint64_t START = LIBC_NAMESPACE::fputil::FPBits<double>(0.25).uintval();
  uint64_t STOP = LIBC_NAMESPACE::fputil::FPBits<double>(4.0).uintval();
  uint64_t STEP = (STOP - START) / COUNT;

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
      double result = LIBC_NAMESPACE::exp10(x);
      ++cc;
      if (isnan(result) || isinf(result))
        continue;

      ++count;

      if (!TEST_MPFR_MATCH_ROUNDING_SILENTLY(mpfr::Operation::Exp10, x, result,
                                             0.5, rounding_mode)) {
        ++fails;
        while (!TEST_MPFR_MATCH_ROUNDING_SILENTLY(mpfr::Operation::Exp10, x,
                                                  result, tol, rounding_mode)) {
          mx = x;
          mr = result;

          if (tol > 1000.0)
            break;

          tol *= 2.0;
        }
      }
    }
    tlog << " Exp10 failed: " << fails << "/" << count << "/" << cc
         << " tests.\n";
    tlog << "   Max ULPs is at most: " << static_cast<uint64_t>(tol) << ".\n";
    if (fails) {
      EXPECT_MPFR_MATCH(mpfr::Operation::Exp10, mx, mr, 0.5, rounding_mode);
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

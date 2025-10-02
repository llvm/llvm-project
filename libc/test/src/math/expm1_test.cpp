//===-- Unittests for e^x - 1 ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/expm1.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

#include "hdr/stdint_proxy.h"

using LlvmLibcExpm1Test = LIBC_NAMESPACE::testing::FPTest<double>;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;
using LIBC_NAMESPACE::testing::tlog;

TEST_F(LlvmLibcExpm1Test, TrickyInputs) {
  constexpr double INPUTS[] = {
      0x1.71547652b82fep-54, 0x1.465655f122ff6p-49, 0x1.bc8ee6b28659ap-46,
      0x1.8442b169f672dp-14, 0x1.9a61fb925970dp-14, 0x1.eb7a4cb841fccp-14,
      0x1.05de80a173eap-2,   0x1.79289c6e6a5cp-2,   0x1.a7b764e2cf47ap-2,
      0x1.b4f0cfb15ca0fp+3,  0x1.9a74cdab36c28p+4,  0x1.2b708872320ddp+5,
      0x1.4c19e5712e377p+5,  0x1.757852a4b93aap+5,  0x1.77f74111e0894p+6,
      0x1.a6c3780bbf824p+6,  0x1.e3d57e4c557f6p+6,  0x1.f07560077985ap+6,
      0x1.1f0da93354198p+7,  0x1.71018579c0758p+7,  0x1.204684c1167e9p+8,
      0x1.5b3e4e2e3bba9p+9,  0x1.6232c09d58d91p+9,  0x1.67a172ceb099p+9,
      0x1.6960d591aec34p+9,  0x1.74910d52d3051p+9,  0x1.ff8p+9,
  };
  constexpr int N = sizeof(INPUTS) / sizeof(INPUTS[0]);
  for (int i = 0; i < N; ++i) {
    double x = INPUTS[i];
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Expm1, x,
                                   LIBC_NAMESPACE::expm1(x), 0.5);
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Expm1, -x,
                                   LIBC_NAMESPACE::expm1(-x), 0.5);
  }
}

TEST_F(LlvmLibcExpm1Test, InDoubleRange) {
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
      if (FPBits(v).is_nan() || FPBits(v).is_inf() || x < 0.0)
        continue;
      double result = LIBC_NAMESPACE::expm1(x);
      ++cc;
      if (FPBits(result).is_nan() || FPBits(result).is_inf())
        continue;

      ++count;

      if (!TEST_MPFR_MATCH_ROUNDING_SILENTLY(mpfr::Operation::Expm1, x, result,
                                             0.5, rounding_mode)) {
        ++fails;
        while (!TEST_MPFR_MATCH_ROUNDING_SILENTLY(mpfr::Operation::Expm1, x,
                                                  result, tol, rounding_mode)) {
          mx = x;
          mr = result;

          if (tol > 1000.0)
            break;

          tol *= 2.0;
        }
      }
    }
    if (fails) {
      tlog << " Expm1 failed: " << fails << "/" << count << "/" << cc
           << " tests.\n";
      tlog << "   Max ULPs is at most: " << static_cast<uint64_t>(tol) << ".\n";
      EXPECT_MPFR_MATCH(mpfr::Operation::Expm1, mx, mr, 0.5, rounding_mode);
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

//===-- Unittests for e^x - 1 ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/errno/libc_errno.h"
#include "src/math/expm1.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include <math.h>

#include <errno.h>
#include <stdint.h>

using LlvmLibcExpm1Test = LIBC_NAMESPACE::testing::FPTest<double>;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;
using LIBC_NAMESPACE::testing::tlog;

TEST_F(LlvmLibcExpm1Test, TrickyInputs) {
  constexpr int N = 21;
  constexpr uint64_t INPUTS[N] = {
      0x3FD79289C6E6A5C0, // x=0x1.79289c6e6a5cp-2
      0x3FD05DE80A173EA0, // x=0x1.05de80a173eap-2
      0xbf1eb7a4cb841fcc, // x=-0x1.eb7a4cb841fccp-14
      0xbf19a61fb925970d, // x=-0x1.9a61fb925970dp-14
      0x3fda7b764e2cf47a, // x=0x1.a7b764e2cf47ap-2
      0xc04757852a4b93aa, // x=-0x1.757852a4b93aap+5
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
      0xbce465655f122ff6, // x=-0x1.465655f122ff6p-49
      0x3d1bc8ee6b28659a, // x=0x1.bc8ee6b28659ap-46
      0x3f18442b169f672d, // x=0x1.8442b169f672dp-14
      0xc02b4f0cfb15ca0f, // x=-0x1.b4f0cfb15ca0fp+3
      0xc042b708872320dd, // x=-0x1.2b708872320ddp+5
  };
  for (int i = 0; i < N; ++i) {
    double x = FPBits(INPUTS[i]).get_val();
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Expm1, x,
                                   LIBC_NAMESPACE::expm1(x), 0.5);
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
      if (isnan(x) || isinf(x) || x < 0.0)
        continue;
      libc_errno = 0;
      double result = LIBC_NAMESPACE::expm1(x);
      ++cc;
      if (isnan(result) || isinf(result))
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
    tlog << " Expm1 failed: " << fails << "/" << count << "/" << cc
         << " tests.\n";
    tlog << "   Max ULPs is at most: " << static_cast<uint64_t>(tol) << ".\n";
    if (fails) {
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

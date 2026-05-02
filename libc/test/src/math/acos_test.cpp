//===-- Unittests for acos ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/macros/optimization.h"
#include "src/math/acos.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

#ifdef LIBC_MATH_HAS_SKIP_ACCURATE_PASS
#define TOLERANCE 8
#else
#define TOLERANCE 0
#endif // LIBC_MATH_HAS_SKIP_ACCURATE_PASS

using LlvmLibcAcosTest = LIBC_NAMESPACE::testing::FPTest<double>;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

using LIBC_NAMESPACE::testing::tlog;

TEST_F(LlvmLibcAcosTest, InDoubleRange) {
  constexpr uint64_t COUNT = 123'451;
  uint64_t START = FPBits(0x1.0p-60).uintval();
  uint64_t STOP = FPBits(1.0).uintval();
  uint64_t STEP = (STOP - START) / COUNT;

  auto test = [&](mpfr::RoundingMode rounding_mode) {
    mpfr::ForceRoundingMode __r(rounding_mode);
    if (!__r.success)
      return;

    uint64_t fails = 0;
    uint64_t count = 0;
    uint64_t cc = 0;
    double mx = 0.0, mr = 0.0;
    double tol = 0.5;

    for (uint64_t i = 0, v = START; i <= COUNT; ++i, v += STEP) {
      double x = FPBits(v).get_val();
      if (FPBits(v).is_inf_or_nan())
        continue;
      double result = LIBC_NAMESPACE::acos(x);
      ++cc;
      if (FPBits(result).is_inf_or_nan())
        continue;

      ++count;

      if (!TEST_MPFR_MATCH_ROUNDING_SILENTLY(mpfr::Operation::Acos, x, result,
                                             TOLERANCE + 0.5, rounding_mode)) {
        ++fails;
        while (!TEST_MPFR_MATCH_ROUNDING_SILENTLY(mpfr::Operation::Acos, x,
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
      tlog << " Acos failed: " << fails << "/" << count << "/" << cc
           << " tests.\n";
      tlog << "   Max ULPs is at most: " << static_cast<uint64_t>(tol) << ".\n";
      EXPECT_MPFR_MATCH(mpfr::Operation::Acos, mx, mr, 0.5, rounding_mode);
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

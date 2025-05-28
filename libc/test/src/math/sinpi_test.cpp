//===-- Exhaustive test for sinpif16 --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "src/math/sinpi.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

#include <iostream>

using LlvmLibcSinpiTest = LIBC_NAMESPACE::testing::FPTest<double>;
using namespace std;
namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

using LIBC_NAMESPACE::testing::tlog;

TEST_F(LlvmLibcSinpiTest, InDoubleRange) {
  constexpr uint64_t COUNT = 1'234'51;
  uint64_t START = LIBC_NAMESPACE::fputil::FPBits<double>(0x1.0p-50).uintval();
  uint64_t STOP = LIBC_NAMESPACE::fputil::FPBits<double>(0x1.0p200).uintval();
  uint64_t STEP = (STOP - START) / COUNT;

  auto test = [&](mpfr::RoundingMode rounding_mode) {
    mpfr::ForceRoundingMode __r(rounding_mode);
    if (!__r.success)
      return;

    uint64_t fails = 0;
    uint64_t count = 0;
    uint64_t cc = 0;
    double mx, mr = 0.0;
    double tol = 2.0;

    for (uint64_t i = 0, v = START; i <= COUNT; ++i, v += STEP) {
      double x = FPBits(v).get_val();
      if (FPBits(v).is_nan() || FPBits(v).is_inf())
        continue;
      LIBC_NAMESPACE::libc_errno = 0;
      double result = LIBC_NAMESPACE::sinpi(x);
      std::cout << "result: " << result << std::endl;
      ++cc;
      if (FPBits(result).is_nan() || FPBits(result).is_inf())
        continue;

      ++count;

      if (!TEST_MPFR_MATCH_ROUNDING_SILENTLY(mpfr::Operation::Sinpi, x, result,
                                             2.0, rounding_mode)) {
        ++fails;
        while (!TEST_MPFR_MATCH_ROUNDING_SILENTLY(mpfr::Operation::Sinpi, x,
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
      tlog << " Sin failed: " << fails << "/" << count << "/" << cc
           << " tests.\n";
      tlog << "   Max ULPs is at most: " << static_cast<uint64_t>(tol) << ".\n";
      EXPECT_MPFR_MATCH(mpfr::Operation::Sinpi, mx, mr, 2.0, rounding_mode);
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

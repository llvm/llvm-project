//===-- Unittests for cbrt ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/cbrt.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

using LlvmLibcCbrtTest = LIBC_NAMESPACE::testing::FPTest<double>;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

using LIBC_NAMESPACE::testing::tlog;

TEST_F(LlvmLibcCbrtTest, InDoubleRange) {
  constexpr uint64_t COUNT = 123'451;
  uint64_t START = FPBits(1.0).uintval();
  uint64_t STOP = FPBits(8.0).uintval();
  uint64_t STEP = (STOP - START) / COUNT;

  auto test = [&](mpfr::RoundingMode rounding_mode) {
    mpfr::ForceRoundingMode force_rounding(rounding_mode);
    if (!force_rounding.success)
      return;

    uint64_t fails = 0;
    uint64_t tested = 0;
    uint64_t total = 0;
    double worst_input, worst_output = 0.0;
    double ulp = 0.5;

    for (uint64_t i = 0, v = START; i <= COUNT; ++i, v += STEP) {
      double x = FPBits(v).get_val();
      if (FPBits(x).is_inf_or_nan())
        continue;

      double result = LIBC_NAMESPACE::cbrt(x);
      ++total;
      if (FPBits(result).is_inf_or_nan())
        continue;

      ++tested;

      if (!TEST_MPFR_MATCH_ROUNDING_SILENTLY(mpfr::Operation::Cbrt, x, result,
                                             0.5, rounding_mode)) {
        ++fails;
        while (!TEST_MPFR_MATCH_ROUNDING_SILENTLY(mpfr::Operation::Cbrt, x,
                                                  result, ulp, rounding_mode)) {
          worst_input = x;
          worst_output = result;

          if (ulp > 1000.0)
            break;

          ulp *= 2.0;
        }
      }
    }
    if (fails) {
      tlog << " Cbrt failed: " << fails << "/" << tested << "/" << total
           << " tests.\n";
      tlog << "   Max ULPs is at most: " << static_cast<uint64_t>(ulp) << ".\n";
      EXPECT_MPFR_MATCH(mpfr::Operation::Cbrt, worst_input, worst_output, 0.5,
                        rounding_mode);
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

TEST_F(LlvmLibcCbrtTest, SpecialValues) {
  constexpr double INPUTS[] = {
      0x1.4f61672324c8p-1028, 0x1.00152f57068b7p-1, 0x1.006509cda9886p-1,
      0x1.018369b92e523p-1,   0x1.10af932ef2bf9p-1, 0x1.1a41117939fdbp-1,
      0x1.2ae8076520d9ap-1,   0x1.a202bfc89ddffp-1, 0x1.a6bb8c803147bp-1,
      0x1.000197b499b1bp+0,   0x1.00065ed266c6cp+0, 0x1.d4306c202c4c2p+0,
      0x1.8fd409efe4851p+1,   0x1.95fd0eb31cc4p+1,  0x1.7cef1d276e335p+2,
      0x1.94910c4fc98p+2,     0x1.a0cc1327bb4c4p+2, 0x1.e7d6ebed549c4p+2,
  };
  for (double v : INPUTS) {
    double x = FPBits(v).get_val();
    ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Cbrt, x,
                                   LIBC_NAMESPACE::cbrt(x), 0.5);
    ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Cbrt, -x,
                                   LIBC_NAMESPACE::cbrt(-x), 0.5);
  }
}

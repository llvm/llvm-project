//===-- Unittests for sin -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/math/sin.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

using LlvmLibcSinTest = LIBC_NAMESPACE::testing::FPTest<double>;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

using LIBC_NAMESPACE::testing::tlog;

TEST_F(LlvmLibcSinTest, TrickyInputs) {
  constexpr double INPUTS[] = {
      0x1.940c877fb7dacp-7,    0x1.fffffffffdb6p24,    0x1.fd4da4ef37075p29,
      0x1.b951f1572eba5p+31,   0x1.55202aefde314p+31,  0x1.85fc0f04c0128p101,
      0x1.7776c2343ba4ep101,   0x1.678309fa50d58p110,  0x1.fffffffffef4ep199,
      -0x1.ab514bfc61c76p+7,   -0x1.f7898d5a756ddp+2,  -0x1.f42fb19b5b9b2p-6,
      0x1.5f09cad750ab1p+3,    -0x1.14823229799c2p+7,  -0x1.0285070f9f1bcp-5,
      0x1.23f40dccdef72p+0,    0x1.43cf16358c9d7p+0,   0x1.addf3b9722265p+0,
      0x1.48ff1782ca91dp+8,    0x1.a211877de55dbp+4,   0x1.dcbfda0c7559ep+8,
      0x1.1ffb509f3db15p+5,    0x1.2345d1e090529p+5,   0x1.ae945054939c2p+10,
      0x1.2e566149bf5fdp+9,    0x1.be886d9c2324dp+6,   -0x1.119471e9216cdp+10,
      -0x1.aaf85537ea4c7p+3,   0x1.cb996c60f437ep+9,   0x1.c96e28eb679f8p+5,
      -0x1.a5eece87e8606p+4,   0x1.e31b55306f22cp+2,   0x1.ae78d360afa15p+0,
      0x1.1685973506319p+3,    0x1.4f2b874135d27p+4,   0x1.ae945054939c2p+10,
      0x1.3eec5912ea7cdp+331,  0x1.dcbfda0c7559ep+8,   0x1.a65d441ea6dcep+4,
      0x1.e639103a05997p+2,    0x1.13114266f9764p+4,   -0x1.3eec5912ea7cdp+331,
      0x1.08087e9aad90bp+887,  0x1.2b5fe88a9d8d5p+903, -0x1.a880417b7b119p+1023,
      -0x1.6deb37da81129p+205, 0x1.08087e9aad90bp+887, 0x1.f6d7518808571p+1023,
      -0x1.8bb5847d49973p+845, 0x1.f08b14e1c4d0fp+890, 0x1.6ac5b262ca1ffp+849,
      0x1.e0000000001c2p-20,
  };
  constexpr int N = sizeof(INPUTS) / sizeof(INPUTS[0]);

  for (int i = 0; i < N; ++i) {
    double x = INPUTS[i];
    ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Sin, x,
                                   LIBC_NAMESPACE::sin(x), 0.5);
  }
}

TEST_F(LlvmLibcSinTest, InDoubleRange) {
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
    double tol = 0.5;

    for (uint64_t i = 0, v = START; i <= COUNT; ++i, v += STEP) {
      double x = FPBits(v).get_val();
      if (isnan(x) || isinf(x))
        continue;
      LIBC_NAMESPACE::libc_errno = 0;
      double result = LIBC_NAMESPACE::sin(x);
      ++cc;
      if (isnan(result) || isinf(result))
        continue;

      ++count;

      if (!TEST_MPFR_MATCH_ROUNDING_SILENTLY(mpfr::Operation::Sin, x, result,
                                             0.5, rounding_mode)) {
        ++fails;
        while (!TEST_MPFR_MATCH_ROUNDING_SILENTLY(mpfr::Operation::Sin, x,
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
      EXPECT_MPFR_MATCH(mpfr::Operation::Sin, mx, mr, 0.5, rounding_mode);
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

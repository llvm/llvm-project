//===-- Unittests for tgamma ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/math/tgamma.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

using LlvmLibcTgammaTest = LIBC_NAMESPACE::testing::FPTest<double>;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

TEST_F(LlvmLibcTgammaTest, ExtremelySmallInputs) {
  constexpr double INPUTS[] = {
      0x1.fffffffffffffp-54, // largest magnitude in the branch (~2^-53)
      0x1.5555555555555p-54, // off-power-of-two
      0x1.0000000000001p-54, // just above an exact power of two
      0x1.0p-54,             // exact power of two (division is exact here)
      0x1.0p-60,
      0x1.0p-100,
      0x1.0p-300,
      0x1.0p-600,
      0x1.0p-900,
      0x1.0p-1000,
  };

  for (size_t i = 0; i < sizeof(INPUTS) / sizeof(INPUTS[0]); i++) {
    double x = INPUTS[i];
    EXPECT_MPFR_MATCH(mpfr::Operation::Tgamma, x, LIBC_NAMESPACE::tgamma(x),
                      1.0);
    EXPECT_MPFR_MATCH(mpfr::Operation::Tgamma, -x, LIBC_NAMESPACE::tgamma(-x),
                      1.0);
  }
}

TEST_F(LlvmLibcTgammaTest, PositiveIntegers) {
  // 171 is the maximum integer input for which
  // tgamma returns a finite output.
  constexpr int N = 171;
  for (int i = 1; i <= N; i++) {
    double x = static_cast<double>(i);
    EXPECT_MPFR_MATCH(mpfr::Operation::Tgamma, x, LIBC_NAMESPACE::tgamma(x),
                      0.5);
  }
}

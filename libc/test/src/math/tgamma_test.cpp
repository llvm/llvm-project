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

TEST_F(LlvmLibcTgammaTest, PositiveIntegers) {
  // 171 is the maximum integer input for which
  // tgamma returns a finite output.
  constexpr int N = 171;
  for (int i = 1; i <= N; i++) {
    double x = static_cast<double>(i);
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Tgamma, x,
                                   LIBC_NAMESPACE::tgamma(x), 0.5);
  }
}

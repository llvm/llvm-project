//===-- Unittests for tan -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/tan.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

#include <math.h>

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

DECLARE_SPECIAL_CONSTANTS(double)

TEST(LlvmLibctanTest, Range) {
  static constexpr double _2pi = 6.283185307179586;
  constexpr UIntType COUNT = 100'000;
  constexpr UIntType STEP = UIntType(-1) / COUNT;
  for (UIntType i = 0, v = 0; i <= COUNT; ++i, v += STEP) {
    double x = double(FPBits(v));
    // TODO: Expand the range of testing after range reduction is implemented.
    if (isnan(x) || isinf(x) || x > _2pi || x < -_2pi)
      continue;

    ASSERT_MPFR_MATCH(mpfr::Operation::Tan, x, LIBC_NAMESPACE::tan(x), 1.0);
  }
}

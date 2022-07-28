//===-- Unittests for expxf -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/math/generic/expxf.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include "utils/UnitTest/FPMatcher.h"
#include "utils/UnitTest/Test.h"
#include <math.h>

#include <errno.h>
#include <stdint.h>

namespace mpfr = __llvm_libc::testing::mpfr;

DECLARE_SPECIAL_CONSTANTS(float)

TEST(LlvmLibcExpxfTest, InFloatRange) {
  constexpr uint32_t COUNT = 1000000;
  constexpr uint32_t STEP = UINT32_MAX / COUNT;
  auto fx = [](float x) -> float {
    auto result = __llvm_libc::exp_eval<-1>(x);
    return static_cast<float>(2 * result.mult_exp * result.r +
                              2 * result.mult_exp);
  };
  for (uint32_t i = 0, v = 0; i <= COUNT; ++i, v += STEP) {
    float x = float(FPBits(v));
    if (isnan(x) || isinf(x) || x < -70 || x > 70 || fabsf(x) < 0x1.0p-10)
      continue;

    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Exp, x, fx(x), 0.5);
  }
}

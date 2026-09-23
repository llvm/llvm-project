//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains smoke tests for static_rounding::exp(x)
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/math_macros.h"
#include "hdr/stdint_proxy.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/math/exp.h"
#include "src/__support/math/exp_integer_eval.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcExpStaticRoundingTest = LIBC_NAMESPACE::testing::FPTest<double>;

namespace static_rounding = LIBC_NAMESPACE::shared::math::static_rounding;
namespace math = LIBC_NAMESPACE::math;

TEST_F(LlvmLibcExpStaticRoundingTest, SpecialNumbers) {
  using LIBC_NAMESPACE::fputil::testing::get_fe_rounding;

  constexpr double VALUES[] = {sNaN,      aNaN,     inf,  neg_inf,
                               -0x1.0p20, 0x1.0p20, zero, neg_zero};

  for (auto rounding : ROUNDING_MODES) {
    const int fenv_rounding = get_fe_rounding(rounding);

    for (auto x : VALUES) {
      EXPECT_FP_EQ_ROUNDING_MODE(
          math::exp(x), static_rounding::exp(x, fenv_rounding), rounding);
      // Statically rounded exp doesn't raise exceptions, but the baseline
      // exp may raise overflow exception.
      // So, we won't check for that here.
    }
  }
}

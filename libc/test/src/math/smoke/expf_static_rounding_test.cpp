//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains smoke tests for static_rounding::expf(x)
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/math_macros.h"
#include "hdr/stdint_proxy.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/math/expf.h"
#include "src/__support/math/expf_integer_eval.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcExpfStaticRoundingTest = LIBC_NAMESPACE::testing::FPTest<float>;

namespace static_rounding = LIBC_NAMESPACE::shared::math::static_rounding;
namespace math = LIBC_NAMESPACE::math;

TEST_F(LlvmLibcExpfStaticRoundingTest, SpecialNumbers) {
  using LIBC_NAMESPACE::fputil::testing::get_fe_rounding;

  constexpr float VALUES[] = {sNaN, aNaN, inf, neg_inf, 0.0f, -0.0f};

  for (auto rounding : ROUNDING_MODES) {
    const int fenv_rounding = get_fe_rounding(rounding);

    for (auto x : VALUES) {
      EXPECT_FP_EQ_ROUNDING_MODE(
          math::expf(x), static_rounding::expf(x, fenv_rounding), rounding);
      // Statically rounded expf doesn't raise exceptions, but the baseline
      // expf may raise overflow exception.
      // So, we won't check for that here.
    }
  }
}

TEST_F(LlvmLibcExpfStaticRoundingTest, Overflow) {
  using LIBC_NAMESPACE::fputil::testing::get_fe_rounding;

  constexpr float VALUES[] = {FPBits(0x7f7fffffU).get_val(),
                              FPBits(0x42cffff8U).get_val(),
                              FPBits(0x42d00008U).get_val()};

  for (auto rounding : ROUNDING_MODES) {
    const int fenv_rounding = get_fe_rounding(rounding);

    for (auto x : VALUES) {
      EXPECT_FP_EQ_ROUNDING_MODE(
          math::expf(x), static_rounding::expf(x, fenv_rounding), rounding);
      // The same reason above, in the SpecialNumbers smoke test suite
    }
  }
}

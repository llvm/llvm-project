//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the unit tests for statically-rounded implementation of
/// static_rounding::expf(x)
///
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "hdr/stdint_proxy.h"
#include "shared/static_rounding_math.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/math/expf.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/RoundingModeUtils.h"
#include "test/UnitTest/Test.h"

using LlvmLibcExpfStaticRoundingTest = LIBC_NAMESPACE::testing::FPTest<float>;
using RoundingMode = LIBC_NAMESPACE::fputil::testing::RoundingMode;

namespace static_rounding = LIBC_NAMESPACE::shared::math::static_rounding;
namespace math = LIBC_NAMESPACE::math;

TEST_F(LlvmLibcExpfStaticRoundingTest, SpecialNumbers) {
  using LIBC_NAMESPACE::fputil::testing::get_fe_rounding;

  for (auto rounding : ROUNDING_MODES) {
    const int fenv_rounding = get_fe_rounding(rounding);
    EXPECT_FP_EQ_ROUNDING_MODE(
        math::expf(aNaN), static_rounding::expf(aNaN, fenv_rounding), rounding);
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ROUNDING_MODE(
        math::expf(inf), static_rounding::expf(inf, fenv_rounding), rounding);
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ROUNDING_MODE(math::expf(neg_inf),
                               static_rounding::expf(neg_inf, fenv_rounding),
                               rounding);
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ROUNDING_MODE(
        math::expf(0.0f), static_rounding::expf(0.0f, fenv_rounding), rounding);
    EXPECT_MATH_ERRNO(0);

    EXPECT_FP_EQ_ROUNDING_MODE(math::expf(-0.0f),
                               static_rounding::expf(-0.0f, fenv_rounding),
                               rounding);
    EXPECT_MATH_ERRNO(0);
  }
}

TEST_F(LlvmLibcExpfStaticRoundingTest, Overflow) {
  constexpr float VALUES[] = {FPBits(0x7f7fffffU).get_val(),
                              FPBits(0x42cffff8U).get_val(),
                              FPBits(0x42d00008U).get_val()};

  for (auto rounding : ROUNDING_MODES) {
    const int fenv_rounding = get_fe_rounding(rounding);

    // Statically rounded expf doesn't raise exceptions

    for (auto x : VALUES) {
      EXPECT_FP_EQ_ROUNDING_MODE(
          math::expf(x), static_rounding::expf(x, fenv_rounding), rounding);
    }
  }
}

TEST_F(LlvmLibcExpfStaticRoundingTest, Underflow) {
  using LIBC_NAMESPACE::fputil::testing::get_fe_rounding;

  constexpr float VALUES[] = {FPBits(0xff7fffffU).get_val(),
                              FPBits(0xc2cffff8U).get_val(),
                              FPBits(0xc2d00008U).get_val()};

  for (auto rounding : ROUNDING_MODES) {
    const int fenv_rounding = get_fe_rounding(rounding);

    // Statically rounded expf doesn't raise exceptions
    for (auto x : VALUES) {
      EXPECT_FP_EQ_ROUNDING_MODE(
          math::expf(x), static_rounding::expf(x, fenv_rounding), rounding);
    }
  }
}

// Test with inputs which are the borders of underflow/overflow but still
// produce valid results without setting errno.
TEST_F(LlvmLibcExpfStaticRoundingTest, Borderline) {
  using LIBC_NAMESPACE::fputil::testing::get_fe_rounding;

  constexpr float VALUES[] = {
      FPBits(0x42affff8U).get_val(), FPBits(0x42b00008U).get_val(),
      FPBits(0xc2affff8U).get_val(), FPBits(0xc2b00008U).get_val(),
      FPBits(0xc236bd8cU).get_val()};

  for (auto rounding : ROUNDING_MODES) {
    const int fenv_rounding = get_fe_rounding(rounding);

    for (auto x : VALUES) {
      EXPECT_FP_EQ_ROUNDING_MODE(
          math::expf(x), static_rounding::expf(x, fenv_rounding), rounding);
    }
  }
}

TEST_F(LlvmLibcExpfStaticRoundingTest, InFloatRange) {
  using LIBC_NAMESPACE::fputil::testing::get_fe_rounding;

  constexpr uint32_t COUNT = 1'231;
  constexpr uint32_t STEP = UINT32_MAX / COUNT;
  for (auto rounding : ROUNDING_MODES) {
    const int fenv_rounding = get_fe_rounding(rounding);

    for (uint32_t i = 0, v = 0; i <= COUNT; ++i, v += STEP) {
      float x = FPBits(v).get_val();
      if (FPBits(v).is_nan() || FPBits(v).is_inf())
        continue;
      libc_errno = 0;
      EXPECT_FP_EQ_ROUNDING_MODE(
          math::expf(x), static_rounding::expf(x, fenv_rounding), rounding);
    }
  }
}

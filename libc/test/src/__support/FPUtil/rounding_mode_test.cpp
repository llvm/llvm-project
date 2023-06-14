//===-- Unittests for the quick rounding mode checks ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/rounding_mode.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

#include <fenv.h>

using __llvm_libc::testing::mpfr::ForceRoundingMode;
using __llvm_libc::testing::mpfr::RoundingMode;

TEST(LlvmLibcFEnvImplTest, QuickRoundingUpTest) {
  using __llvm_libc::fputil::fenv_is_round_up;
  {
    ForceRoundingMode __r(RoundingMode::Upward);
    if (__r.success)
      ASSERT_TRUE(fenv_is_round_up());
  }
  {
    ForceRoundingMode __r(RoundingMode::Downward);
    if (__r.success)
      ASSERT_FALSE(fenv_is_round_up());
  }
  {
    ForceRoundingMode __r(RoundingMode::Nearest);
    if (__r.success)
      ASSERT_FALSE(fenv_is_round_up());
  }
  {
    ForceRoundingMode __r(RoundingMode::TowardZero);
    if (__r.success)
      ASSERT_FALSE(fenv_is_round_up());
  }
}

TEST(LlvmLibcFEnvImplTest, QuickRoundingDownTest) {
  using __llvm_libc::fputil::fenv_is_round_down;
  {
    ForceRoundingMode __r(RoundingMode::Upward);
    if (__r.success)
      ASSERT_FALSE(fenv_is_round_down());
  }
  {
    ForceRoundingMode __r(RoundingMode::Downward);
    if (__r.success)
      ASSERT_TRUE(fenv_is_round_down());
  }
  {
    ForceRoundingMode __r(RoundingMode::Nearest);
    if (__r.success)
      ASSERT_FALSE(fenv_is_round_down());
  }
  {
    ForceRoundingMode __r(RoundingMode::TowardZero);
    if (__r.success)
      ASSERT_FALSE(fenv_is_round_down());
  }
}

TEST(LlvmLibcFEnvImplTest, QuickRoundingNearestTest) {
  using __llvm_libc::fputil::fenv_is_round_to_nearest;
  {
    ForceRoundingMode __r(RoundingMode::Upward);
    if (__r.success)
      ASSERT_FALSE(fenv_is_round_to_nearest());
  }
  {
    ForceRoundingMode __r(RoundingMode::Downward);
    if (__r.success)
      ASSERT_FALSE(fenv_is_round_to_nearest());
  }
  {
    ForceRoundingMode __r(RoundingMode::Nearest);
    if (__r.success)
      ASSERT_TRUE(fenv_is_round_to_nearest());
  }
  {
    ForceRoundingMode __r(RoundingMode::TowardZero);
    if (__r.success)
      ASSERT_FALSE(fenv_is_round_to_nearest());
  }
}

TEST(LlvmLibcFEnvImplTest, QuickRoundingTowardZeroTest) {
  using __llvm_libc::fputil::fenv_is_round_to_zero;
  {
    ForceRoundingMode __r(RoundingMode::Upward);
    if (__r.success)
      ASSERT_FALSE(fenv_is_round_to_zero());
  }
  {
    ForceRoundingMode __r(RoundingMode::Downward);
    if (__r.success)
      ASSERT_FALSE(fenv_is_round_to_zero());
  }
  {
    ForceRoundingMode __r(RoundingMode::Nearest);
    if (__r.success)
      ASSERT_FALSE(fenv_is_round_to_zero());
  }
  {
    ForceRoundingMode __r(RoundingMode::TowardZero);
    if (__r.success)
      ASSERT_TRUE(fenv_is_round_to_zero());
  }
}

TEST(LlvmLibcFEnvImplTest, QuickGetRoundTest) {
  using __llvm_libc::fputil::quick_get_round;
  {
    ForceRoundingMode __r(RoundingMode::Upward);
    if (__r.success)
      ASSERT_EQ(quick_get_round(), FE_UPWARD);
  }
  {
    ForceRoundingMode __r(RoundingMode::Downward);
    if (__r.success)
      ASSERT_EQ(quick_get_round(), FE_DOWNWARD);
  }
  {
    ForceRoundingMode __r(RoundingMode::Nearest);
    if (__r.success)
      ASSERT_EQ(quick_get_round(), FE_TONEAREST);
  }
  {
    ForceRoundingMode __r(RoundingMode::TowardZero);
    if (__r.success)
      ASSERT_EQ(quick_get_round(), FE_TOWARDZERO);
  }
}

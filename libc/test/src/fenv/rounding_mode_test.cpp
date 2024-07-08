//===-- Unittests for fegetround and fesetround ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/fenv/fegetround.h"
#include "src/fenv/fesetround.h"

#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/Test.h"

#include "src/__support/FPUtil/rounding_mode.h"
#include "src/__support/macros/properties/architectures.h"

#include "hdr/fenv_macros.h"

using LlvmLibcRoundingModeTest = LIBC_NAMESPACE::testing::FEnvSafeTest;

#pragma fenv_access(on)

TEST_F(LlvmLibcRoundingModeTest, SetAndGet) {
  struct ResetDefaultRoundingMode {
    int original = LIBC_NAMESPACE::fegetround();
    ~ResetDefaultRoundingMode() { LIBC_NAMESPACE::fesetround(original); }
  } reset;

  int s = LIBC_NAMESPACE::fesetround(FE_UPWARD);
  EXPECT_EQ(s, 0);
  int rm = LIBC_NAMESPACE::fegetround();
  EXPECT_EQ(rm, FE_UPWARD);
  EXPECT_TRUE(LIBC_NAMESPACE::fputil::fenv_is_round_up());

  s = LIBC_NAMESPACE::fesetround(FE_DOWNWARD);
  EXPECT_EQ(s, 0);
  rm = LIBC_NAMESPACE::fegetround();
  EXPECT_EQ(rm, FE_DOWNWARD);
  EXPECT_TRUE(LIBC_NAMESPACE::fputil::fenv_is_round_down());

  s = LIBC_NAMESPACE::fesetround(FE_TOWARDZERO);
  EXPECT_EQ(s, 0);
  rm = LIBC_NAMESPACE::fegetround();
  EXPECT_EQ(rm, FE_TOWARDZERO);
  EXPECT_TRUE(LIBC_NAMESPACE::fputil::fenv_is_round_to_zero());

  s = LIBC_NAMESPACE::fesetround(FE_TONEAREST);
  EXPECT_EQ(s, 0);
  rm = LIBC_NAMESPACE::fegetround();
  EXPECT_EQ(rm, FE_TONEAREST);
  EXPECT_TRUE(LIBC_NAMESPACE::fputil::fenv_is_round_to_nearest());
}

#ifdef LIBC_TARGET_ARCH_IS_AMDGPU

// These values are extensions from the following documentation:
// https://llvm.org/docs/AMDGPUUsage.html#amdgpu-rounding-mode-enumeration-values-table.
// This will set the f64/f16 rounding mode to nearest while modifying f32 only.
enum RoundingF32 : int {
  ROUND_F32_TONEAREST = 1,
  ROUND_F32_UPWARD = 11,
  ROUND_F32_DOWNWARD = 14,
  ROUND_F32_TOWARDZERO = 17,
};

TEST_F(LlvmLibcRoundingModeTest, AMDGPUExtensionF32) {
  struct ResetDefaultRoundingMode {
    int original = LIBC_NAMESPACE::fegetround();
    ~ResetDefaultRoundingMode() { LIBC_NAMESPACE::fesetround(original); }
  } reset;

  int s = LIBC_NAMESPACE::fesetround(ROUND_F32_UPWARD);
  EXPECT_EQ(s, 0);
  int rm = LIBC_NAMESPACE::fegetround();
  EXPECT_TRUE(LIBC_NAMESPACE::fputil::fenv_is_round_up());

  s = LIBC_NAMESPACE::fesetround(ROUND_F32_DOWNWARD);
  EXPECT_EQ(s, 0);
  rm = LIBC_NAMESPACE::fegetround();
  EXPECT_TRUE(LIBC_NAMESPACE::fputil::fenv_is_round_down());

  s = LIBC_NAMESPACE::fesetround(ROUND_F32_TOWARDZERO);
  EXPECT_EQ(s, 0);
  rm = LIBC_NAMESPACE::fegetround();
  EXPECT_TRUE(LIBC_NAMESPACE::fputil::fenv_is_round_to_zero());

  s = LIBC_NAMESPACE::fesetround(ROUND_F32_TONEAREST);
  EXPECT_EQ(s, 0);
  rm = LIBC_NAMESPACE::fegetround();
  EXPECT_TRUE(LIBC_NAMESPACE::fputil::fenv_is_round_to_nearest());
}

// TODO: Check to verify that the f64 rounding mode is unaffected. This requires
//       updating the floating point utils to support doubles.
#endif

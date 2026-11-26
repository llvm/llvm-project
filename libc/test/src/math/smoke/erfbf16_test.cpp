//===-- erfbf16_test.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file contains smoke tests for the erfbf16 function.
//
//===-- Smoke tests for erfbf16 -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "hdr/stdint_proxy.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/bfloat16.h"
#include "src/math/erfbf16.h"
#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcErfbf16SmokeTest = LIBC_NAMESPACE::testing::FPTest<bfloat16>;
using BFBits = LIBC_NAMESPACE::fputil::FPBits<bfloat16>;

static bfloat16 from_bits(uint16_t bits) { return BFBits(bits).get_val(); }

TEST_F(LlvmLibcErfbf16SmokeTest, ZerosAndInfinities) {
  EXPECT_FP_EQ_ALL_ROUNDING(from_bits(0x0000U),
                            LIBC_NAMESPACE::erfbf16(from_bits(0x0000U)));
  EXPECT_FP_EQ_ALL_ROUNDING(from_bits(0x8000U),
                            LIBC_NAMESPACE::erfbf16(from_bits(0x8000U)));
  EXPECT_FP_EQ_ALL_ROUNDING(from_bits(0x3f80U),
                            LIBC_NAMESPACE::erfbf16(from_bits(0x7f80U)));
  EXPECT_FP_EQ_ALL_ROUNDING(from_bits(0xbf80U),
                            LIBC_NAMESPACE::erfbf16(from_bits(0xff80U)));
}

TEST_F(LlvmLibcErfbf16SmokeTest, NaNs) {
  const bfloat16 qnan = from_bits(0x7fc1U);
  LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
  const bfloat16 quiet_result = LIBC_NAMESPACE::erfbf16(qnan);
  EXPECT_TRUE(BFBits(quiet_result).is_nan());
  EXPECT_FP_EXCEPTION(0);

  const bfloat16 snan = from_bits(0x7f81U);
  LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
  const bfloat16 signaling_result = LIBC_NAMESPACE::erfbf16(snan);
  EXPECT_TRUE(BFBits(signaling_result).is_nan());
  EXPECT_FP_EXCEPTION(FE_INVALID);
}

TEST_F(LlvmLibcErfbf16SmokeTest, RepresentativeValuesNearest) {
  // All expectations here are for round-to-nearest.
  using RoundingMode = LIBC_NAMESPACE::fputil::testing::RoundingMode;
  LIBC_NAMESPACE::fputil::testing::ForceRoundingMode rounding(
      RoundingMode::Nearest);
  if (!rounding.success)
    return;

  EXPECT_EQ(BFBits(LIBC_NAMESPACE::erfbf16(from_bits(0x3f00U))).uintval(),
            static_cast<uint16_t>(0x3f05U)); // erf(0.5)
  EXPECT_EQ(BFBits(LIBC_NAMESPACE::erfbf16(from_bits(0x3f80U))).uintval(),
            static_cast<uint16_t>(0x3f58U)); // erf(1.0)
  EXPECT_EQ(BFBits(LIBC_NAMESPACE::erfbf16(from_bits(0x4000U))).uintval(),
            static_cast<uint16_t>(0x3f7fU)); // erf(2.0)
  EXPECT_EQ(BFBits(LIBC_NAMESPACE::erfbf16(from_bits(0x400dU))).uintval(),
            static_cast<uint16_t>(0x3f80U)); // erf(2.203125)
}

TEST_F(LlvmLibcErfbf16SmokeTest, InexactForNonzeroFinite) {
  const bfloat16 one = from_bits(0x3f80U);
  LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
  const bfloat16 result = LIBC_NAMESPACE::erfbf16(one);
  EXPECT_EQ(BFBits(result).uintval(), static_cast<uint16_t>(0x3f58U));
  EXPECT_FP_EXCEPTION(FE_INEXACT);
}

using LlvmLibcErfbf16UnderflowTest = LIBC_NAMESPACE::testing::FEnvSafeTest;

TEST_F(LlvmLibcErfbf16UnderflowTest, SubnormalUnderflowAndErrno) {
  using Bits = LIBC_NAMESPACE::fputil::FPBits<bfloat16>;
  using namespace LIBC_NAMESPACE::fputil::testing;

  constexpr uint16_t INPUTS[] = {0x0001U, 0x0040U};
  constexpr uint16_t SIGNS[] = {0x0000U, 0x8000U};

#ifdef LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
  constexpr RoundingMode MODES[] = {RoundingMode::Nearest};
#else
  constexpr RoundingMode MODES[] = {RoundingMode::Nearest, RoundingMode::Upward,
                                    RoundingMode::Downward,
                                    RoundingMode::TowardZero};
#endif

  for (RoundingMode mode : MODES) {
    ForceRoundingMode rounding(mode);
    if (!rounding.success)
      continue;

    for (uint16_t bits : INPUTS) {
      for (uint16_t sign : SIGNS) {
        bfloat16 x = Bits(static_cast<uint16_t>(bits | sign)).get_val();

        LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
        LIBC_NAMESPACE::libc_errno = 0;

        bfloat16 result = LIBC_NAMESPACE::erfbf16(x);

        // The rounded result remains subnormal.
        EXPECT_EQ(Bits(result).uintval() & 0x7f80U, 0U);
        EXPECT_FP_EXCEPTION(FE_UNDERFLOW | FE_INEXACT);
        EXPECT_MATH_ERRNO(ERANGE);
      }
    }
  }
}

TEST_F(LlvmLibcErfbf16UnderflowTest, SubnormalInputNormalResult) {
  using Bits = LIBC_NAMESPACE::fputil::FPBits<bfloat16>;
  using namespace LIBC_NAMESPACE::fputil::testing;

  constexpr uint16_t INPUTS[] = {0x007fU, 0x807fU};

#ifdef LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY
  constexpr RoundingMode MODES[] = {RoundingMode::Nearest};
#else
  constexpr RoundingMode MODES[] = {RoundingMode::Nearest, RoundingMode::Upward,
                                    RoundingMode::Downward,
                                    RoundingMode::TowardZero};
#endif

  for (RoundingMode mode : MODES) {
    ForceRoundingMode rounding(mode);
    if (!rounding.success)
      continue;

    for (uint16_t bits : INPUTS) {
      bfloat16 x = Bits(bits).get_val();

      LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
      LIBC_NAMESPACE::libc_errno = 0;

      bfloat16 result = LIBC_NAMESPACE::erfbf16(x);

      // A subnormal input can produce a normal erf result.
      EXPECT_NE(Bits(result).uintval() & 0x7f80U, 0U);
      EXPECT_EQ(LIBC_NAMESPACE::fputil::test_except(FE_UNDERFLOW), 0);
      EXPECT_MATH_ERRNO(0);
    }
  }
}

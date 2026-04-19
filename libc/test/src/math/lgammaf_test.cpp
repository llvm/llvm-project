//===-- Unittests for lgammaf ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/stdint_proxy.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/libc_errno.h"
#include "src/math/lgammaf.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

using LlvmLibcLgammafTest = LIBC_NAMESPACE::testing::FPTest<float>;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

// Hard-to-round cases collected from prior exhaustive tests.
// Each is a midpoint-case where small precision drift in the implementation
// tips the double to float rounding direction.
TEST_F(LlvmLibcLgammafTest, HardCases) {
  static constexpr uint32_t INPUTS[] = {
      0x77ac5740u, // 0x1.58ace8p+112 (huge positive, Stirling path)
      0x87acf970u, // -0x1.f59f2ep-16  (small negative, reflection path)
      0xc0afda0bu, // -0x1.5fb416p+1   (negative middle, reflection through M3)
      0x3fc0737cu, // 0x1.80e6f8p+0    (medium near lgamma minimum)
      0x3fc07be9u, // 0x1.80f7d2p+0    (same)
      0x3fd0b2bfu, // 0x1.a1657ep+0    (same)
      0xb3000824u, // -0x1.001048p-25  (tiny negative)
      0x30f00a14u, // 0x1.e01428p-30   (tiny positive)
      0x00800000u, // 0x1p-126         (min normal positive)
      0x80800000u, // -0x1p-126        (min normal negative)
      0x3f800000u, // 0x1p+0           (lgamma(1) = 0 exactly)
      0x40000000u, // 0x1p+1           (lgamma(2) = 0 exactly)
  };
  for (uint32_t v : INPUTS) {
    float x = FPBits(v).get_val();
    libc_errno = 0;
    float result = LIBC_NAMESPACE::lgammaf(x);
    if (FPBits(result).is_nan() || FPBits(result).is_inf() || libc_errno != 0)
      continue;
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Lgamma, x,
                                   LIBC_NAMESPACE::lgammaf(x), 0.5);
  }
}

TEST_F(LlvmLibcLgammafTest, PositiveRange) {
  constexpr uint32_t COUNT = 100'000;
  // [min normal, max_normal].
  constexpr uint32_t POS_START = 0x0080'0000U;
  constexpr uint32_t POS_STOP = 0x7f7f'ffffU;
  constexpr uint32_t STEP = (POS_STOP - POS_START) / COUNT;
  for (uint32_t i = 0, v = POS_START; i <= COUNT; ++i, v += STEP) {
    float x = FPBits(v).get_val();
    libc_errno = 0;
    float result = LIBC_NAMESPACE::lgammaf(x);
    if (FPBits(result).is_nan() || FPBits(result).is_inf() || libc_errno != 0)
      continue;
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Lgamma, x,
                                   LIBC_NAMESPACE::lgammaf(x), 0.5);
  }
}

TEST_F(LlvmLibcLgammafTest, NegativeRange) {
  constexpr uint32_t COUNT = 100'000;
  //-max_normal, -min normal].
  constexpr uint32_t NEG_START = 0x8080'0000U;
  constexpr uint32_t NEG_STOP = 0xff7f'ffffU;
  constexpr uint32_t STEP = (NEG_STOP - NEG_START) / COUNT;
  for (uint32_t i = 0, v = NEG_START; i <= COUNT; ++i, v += STEP) {
    float x = FPBits(v).get_val();
    if (FPBits(v).is_nan() || FPBits(v).is_inf())
      continue;
    libc_errno = 0;
    float result = LIBC_NAMESPACE::lgammaf(x);
    if (FPBits(result).is_nan() || FPBits(result).is_inf() || libc_errno != 0)
      continue;
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Lgamma, x,
                                   LIBC_NAMESPACE::lgammaf(x), 0.5);
  }
}

TEST_F(LlvmLibcLgammafTest, SmallRange) {
  constexpr float LO = -4.0f;
  constexpr float HI = 4.0f;
  constexpr uint32_t COUNT = 10'000;
  for (uint32_t i = 0; i <= COUNT; ++i) {
    float x =
        LO + (HI - LO) * static_cast<float>(i) / static_cast<float>(COUNT);
    libc_errno = 0;
    float result = LIBC_NAMESPACE::lgammaf(x);
    if (FPBits(result).is_nan() || FPBits(result).is_inf() || libc_errno != 0)
      continue;
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Lgamma, x,
                                   LIBC_NAMESPACE::lgammaf(x), 0.5);
  }
}

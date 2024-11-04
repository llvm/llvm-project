//===-- Unittests for cosf ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/errno/libc_errno.h"
#include "src/math/cosf.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "test/src/math/sdcomp26094.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include <math.h>

#include <errno.h>
#include <stdint.h>

using LIBC_NAMESPACE::testing::SDCOMP26094_VALUES;
using LlvmLibcCosfTest = LIBC_NAMESPACE::testing::FPTest<float>;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

TEST_F(LlvmLibcCosfTest, SpecialNumbers) {
  libc_errno = 0;

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::cosf(aNaN));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::cosf(0.0f));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(1.0f, LIBC_NAMESPACE::cosf(-0.0f));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::cosf(inf));
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::cosf(neg_inf));
  EXPECT_MATH_ERRNO(EDOM);
}

TEST_F(LlvmLibcCosfTest, InFloatRange) {
  constexpr uint32_t COUNT = 100'000;
  constexpr uint32_t STEP = UINT32_MAX / COUNT;
  for (uint32_t i = 0, v = 0; i <= COUNT; ++i, v += STEP) {
    float x = float(FPBits(v));
    if (isnan(x) || isinf(x))
      continue;
    ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Cos, x,
                                   LIBC_NAMESPACE::cosf(x), 0.5);
  }
}

TEST_F(LlvmLibcCosfTest, SpecificBitPatterns) {
  constexpr int N = 42;
  constexpr uint32_t INPUTS[N] = {
      0x3f06'0a92U, // x = pi/6
      0x3f3a'dc51U, // x = 0x1.75b8a2p-1f
      0x3f49'0fdbU, // x = pi/4
      0x3f86'0a92U, // x = pi/3
      0x3fa7'832aU, // x = 0x1.4f0654p+0f
      0x3fc9'0fdbU, // x = pi/2
      0x4017'1973U, // x = 0x1.2e32e6p+1f
      0x4049'0fdbU, // x = pi
      0x4096'cbe4U, // x = 0x1.2d97c8p+2f
      0x40c9'0fdbU, // x = 2*pi
      0x433b'7490U, // x = 0x1.76e92p+7f
      0x437c'e5f1U, // x = 0x1.f9cbe2p+7f
      0x4619'9998U, // x = 0x1.33333p+13f
      0x474d'246fU, // x = 0x1.9a48dep+15f
      0x4afd'ece4U, // x = 0x1.fbd9c8p+22f
      0x4c23'32e9U, // x = 0x1.4665d2p+25f
      0x50a3'e87fU, // x = 0x1.47d0fep+34f
      0x5239'47f6U, // x = 0x1.728fecp+37f
      0x53b1'46a6U, // x = 0x1.628d4cp+40f
      0x5532'5019U, // x = 0x1.64a032p+43f
      0x55ca'fb2aU, // x = 0x1.95f654p+44f
      0x588e'f060U, // x = 0x1.1de0cp+50f
      0x5922'aa80U, // x = 0x1.4555p+51f
      0x5aa4'542cU, // x = 0x1.48a858p+54f
      0x5c07'bcd0U, // x = 0x1.0f79ap+57f
      0x5ebc'fddeU, // x = 0x1.79fbbcp+62f
      0x5f18'b878U, // x = 0x1.3170fp+63f
      0x5fa6'eba7U, // x = 0x1.4dd74ep+64f
      0x6115'cb11U, // x = 0x1.2b9622p+67f
      0x61a4'0b40U, // x = 0x1.48168p+68f
      0x6386'134eU, // x = 0x1.0c269cp+72f
      0x6589'8498U, // x = 0x1.13093p+76f
      0x6600'0001U, // x = 0x1.000002p+77f
      0x664e'46e4U, // x = 0x1.9c8dc8p+77f
      0x66b0'14aaU, // x = 0x1.602954p+78f
      0x67a9'242bU, // x = 0x1.524856p+80f
      0x6a19'76f1U, // x = 0x1.32ede2p+85f
      0x6c55'da58U, // x = 0x1.abb4bp+89f
      0x6f79'be45U, // x = 0x1.f37c8ap+95f
      0x7276'69d4U, // x = 0x1.ecd3a8p+101f
      0x7758'4625U, // x = 0x1.b08c4ap+111f
      0x7bee'f5efU, // x = 0x1.ddebdep+120f
  };

  for (int i = 0; i < N; ++i) {
    float x = float(FPBits(INPUTS[i]));
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Cos, x,
                                   LIBC_NAMESPACE::cosf(x), 0.5);
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Cos, -x,
                                   LIBC_NAMESPACE::cosf(-x), 0.5);
  }
}

// SDCOMP-26094: check cosf in the cases for which the range reducer
// returns values furthest beyond its nominal upper bound of pi/4.
TEST_F(LlvmLibcCosfTest, SDCOMP_26094) {
  for (uint32_t v : SDCOMP26094_VALUES) {
    float x = float(FPBits(v));
    ASSERT_MPFR_MATCH(mpfr::Operation::Cos, x, LIBC_NAMESPACE::cosf(x), 0.5);
  }
}

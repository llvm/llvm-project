//===-- Exhaustive test for fmabf16 ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/bfloat16.h"
#include "src/math/fmabf16.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

using LlvmLibcFmaBf16Test = LIBC_NAMESPACE::testing::FPTest<bfloat16>;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

// range: [0, inf]
static constexpr uint16_t POS_START = 0x0000U;
static constexpr uint16_t POS_STOP = 0x7f80U;

// range: [-0, -inf]
static constexpr uint16_t NEG_START = 0x8000U;
static constexpr uint16_t NEG_STOP = 0xff80U;

// subnormal range (positive)
static constexpr uint16_t SUBNORM_POS_START = 0x0001U;
static constexpr uint16_t SUBNORM_POS_STOP = 0x007FU;

// subnormal range (negative)
static constexpr uint16_t SUBNORM_NEG_START = 0x8001U;
static constexpr uint16_t SUBNORM_NEG_STOP = 0x807FU;


TEST_F(LlvmLibcFmaBf16Test, SpecialNumbers) {
  const bfloat16 z_values[] = {zero,
                               neg_zero,
                               inf,
                               neg_inf,
                               min_normal,
                               max_normal};
  const bfloat16 x_values[] = {zero,
                               neg_zero,
                               inf,
                               neg_inf,
                               min_normal,
                               max_normal};
  const bfloat16 y_values[] = {zero,
                               neg_zero,
                               inf,
                               neg_inf,
                               min_normal,
                               max_normal};

  for (bfloat16 x : x_values) {
    for (bfloat16 y : y_values) {
      for (bfloat16 z : z_values) {
        mpfr::TernaryInput<bfloat16> input{x, y, z};

        EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Fma, input,
                                       LIBC_NAMESPACE::fmabf16(x, y, z), 0.5);
      }
      bfloat16 neg_xy = -(x * y);
      mpfr::TernaryInput<bfloat16> input{x, y, neg_xy};
      EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Fma, input,
                                     LIBC_NAMESPACE::fmabf16(x, y, neg_xy),
                                     0.5);
    }
  }
}

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
#include <vector>

using LlvmLibcFmaBf16Test = LIBC_NAMESPACE::testing::FPTest<bfloat16>;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

// range: [0, inf]
static constexpr uint16_t POS_START = 0x0000U;
static constexpr uint16_t POS_STOP = 0x7f80U;

// range: [-0, -inf]
static constexpr uint16_t NEG_START = 0x8000U;
static constexpr uint16_t NEG_STOP = 0xff80U; //-inf

TEST_F(LlvmLibcFmaBf16Test, PositiveRange) {

  
    for (uint16_t v1 = POS_START; v1 <= POS_STOP; ++v1) {
      for (uint16_t v2 = POS_START; v2 <= POS_STOP; v2 += 256) {

        bfloat16 x = FPBits(v1).get_val();
        bfloat16 y = FPBits(v2).get_val();
        bfloat16 z = FPBits(POS_START).get_val();
        
        mpfr::TernaryInput<bfloat16> input{x, y, z};

        EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Fma, input,
                                       LIBC_NAMESPACE::fmabf16(x, y, z), 0.5);
      }
    }
  
}

TEST_F(LlvmLibcFmaBf16Test, NegativeRange) {
  for (uint16_t v1 = NEG_START; v1 <= NEG_STOP; ++v1) {
      for (uint16_t v2 = NEG_START; v2 <= NEG_STOP; v2 += 256) {

        bfloat16 x = FPBits(v1).get_val();
        bfloat16 y = FPBits(v2).get_val();
        bfloat16 z = FPBits(NEG_START).get_val();

        mpfr::TernaryInput<bfloat16> input{x, y, z};

        EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Fma, input,
                                       LIBC_NAMESPACE::fmabf16(x, y, z), 0.5);
      }
    }
  
}

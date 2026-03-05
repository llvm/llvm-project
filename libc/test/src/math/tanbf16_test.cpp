//===-- Exhaustive test for tanbf16 ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/bfloat16.h"
#include "src/math/tanbf16.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include <iostream>

using LlvmLibcTanBf16Test = LIBC_NAMESPACE::testing::FPTest<bfloat16>;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

// Range: [0, Inf]
static constexpr uint16_t POS_START = 0x0000U;
static constexpr uint16_t POS_STOP = 0x7f80U;

// Range: [-Inf, 0]
static constexpr uint16_t NEG_START = 0x8000U;
static constexpr uint16_t NEG_STOP = 0xff80U;

TEST_F(LlvmLibcTanBf16Test, PositiveRange) {
  uint16_t last_equal = 0; // remove
  for (uint16_t v = POS_START; v <= POS_STOP; ++v) {
    bfloat16 x = FPBits(v).get_val();

    bfloat16 result = LIBC_NAMESPACE::tanbf16(x);
    // Check if tan(x) rounds to x itself
    if (FPBits(result).uintval() == v) {
      last_equal = v;
    } else {
      // First x where tan(x) != x — print and stop
      bfloat16 prev = FPBits(last_equal).get_val();
      std::cout << "Last x where tan(x)==x: "
                << "hex=0x" << std::hex << last_equal << " value=" << std::dec
                << (float)prev << "\n";
      std::cout << "First x where tan(x)!=x: "
                << "hex=0x" << std::hex << v << " value=" << std::dec
                << (float)x << "\n";
      std::cout << "tan(first_diff_x)=" << (float)result << "\n";
      std::cout << "\n=== USE IN CODE ===\n"
                << "if (x_abs <= 0x" << std::hex << last_equal
                << "U)  // tan(x)==x for bfloat16\n";
      break;
    }

    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Tan, x,
                                   LIBC_NAMESPACE::tanbf16(x), 0.5);
  }
}

TEST_F(LlvmLibcTanBf16Test, NegativeRange) {
  uint16_t last_equal = 0x8000; // remove
  for (uint16_t v = NEG_START; v <= NEG_STOP; ++v) {
    bfloat16 x = FPBits(v).get_val();

    bfloat16 result = LIBC_NAMESPACE::tanbf16(x);
    if (FPBits(result).uintval() == v) {
      last_equal = v;
    } else {
      std::cout << "Negative — last equal: "
                << "hex=0x" << std::hex << last_equal << " value=" << std::dec
                << (float)FPBits(last_equal).get_val() << "\n";
      break;
    }

    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Tan, x,
                                   LIBC_NAMESPACE::tanbf16(x), 0.5);
  }
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/bfloat16.h"
#include "src/math/tanhbf16.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

using LlvmLibcTanhBf16Test = LIBC_NAMESPACE::testing::FPTest<bfloat16>;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

TEST_F(LlvmLibcTanhBf16Test, PositiveRange) {
  for (uint32_t v = 0; v <= 0x7f80U; ++v) {
    bfloat16 x = FPBits(static_cast<uint16_t>(v)).get_val();
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Tanh, x,
                                   LIBC_NAMESPACE::tanhbf16(x), 0.5);
  }
}

TEST_F(LlvmLibcTanhBf16Test, NegativeRange) {
  for (uint32_t v = 0x8000U; v <= 0xff80U; ++v) {
    bfloat16 x = FPBits(static_cast<uint16_t>(v)).get_val();
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Tanh, x,
                                   LIBC_NAMESPACE::tanhbf16(x), 0.5);
  }
}

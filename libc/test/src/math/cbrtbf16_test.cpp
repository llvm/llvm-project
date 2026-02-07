//===-- Exhaustive test for cbrtbf16 --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/bfloat16.h"
#include "src/math/cbrtbf16.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

using LlvmLibcCbrtbf16Test = LIBC_NAMESPACE::testing::FPTest<bfloat16>;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

// for the exhaustive test, we test with every 16-bit combination
// and skip NaN cases. v is uint32_t to prevent integer overflow and
// wraparound to 0.
TEST_F(LlvmLibcCbrtbf16Test, Exhaustive) {
  for (uint32_t v = 0x00000; v < 0x10000; ++v) {
    bfloat16 x =
        LIBC_NAMESPACE::fputil::FPBits<bfloat16>(static_cast<uint16_t>(v))
            .get_val();

    LIBC_NAMESPACE::fputil::FPBits<bfloat16> bits(x); // NaN checking
    if (bits.is_nan())
      continue;

    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Cbrt, x,
                                   LIBC_NAMESPACE::cbrtbf16(x), 0.5);
  }
}
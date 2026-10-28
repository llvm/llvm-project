//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Tests every valid positive bfloat16 input against MPFR across all
/// supported floating-point rounding modes.
///
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/bfloat16.h"
#include "src/math/acoshbf16.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

using LlvmLibcAcoshBf16Test = LIBC_NAMESPACE::testing::FPTest<bfloat16>;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

// Domain: [1, +Inf]
// The first 31 inputs above 1 exercise the polynomial fast path.
static constexpr uint16_t START = 0x3f80U;
static constexpr uint16_t STOP = 0x7f80U;

TEST_F(LlvmLibcAcoshBf16Test, PositiveRange) {
  for (uint16_t bits = START; bits <= STOP; bits++) {
    bfloat16 x = FPBits(bits).get_val();
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Acosh, x,
                                   LIBC_NAMESPACE::acoshbf16(x), 0.5);
  }
}

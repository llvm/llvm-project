//===-- Unittests for acoshf16 --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/stdint_proxy.h"
#include "src/math/acoshf16.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

using LlvmLibcAcoshf16Test = LIBC_NAMESPACE::testing::FPTest<float16>;
namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

static constexpr uint16_t START = 0x3c00U;
static constexpr uint16_t STOP = 0x7c00;

TEST_F(LlvmLibcAcoshf16Test, PositiveRange) {
  for (uint16_t v = START; v <= STOP; ++v) {
    float16 x = FPBits(v).get_val();
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Acosh, x,
                                   LIBC_NAMESPACE::acoshf16(x), 0.5);
  }
}

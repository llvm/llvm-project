//===-- Unittests for acoshf16 ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions. 
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/cast.h"
#include "src/errno/libc_errno.h"
#include "src/math/acoshf16.h"
#include "src/__support/FPUtil/FPBits.h"
#include "test/UnitTest/FPMatcher.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include <stdint.h>

using LlvmLibcAcoshf16Test = LIBC_NAMESPACE::testing::FPTest<float16>;
namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

static constexpr uint16_t START = 0x3c00U;
static constexpr uint16_t STOP  = 0x7bffU;

TEST_F(LlvmLibcAcoshf16Test, SpecialNumbers) {
  LIBC_NAMESPACE::libc_errno = 0;

  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::acoshf16(aNaN));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_WITH_EXCEPTION(aNaN, LIBC_NAMESPACE::acoshf16(sNaN), FE_INVALID);
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(float16(0.0f),
                            LIBC_NAMESPACE::acoshf16(float16(1.0f)));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::acoshf16(float16(0.5f)));
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_EQ_ALL_ROUNDING(inf, LIBC_NAMESPACE::acoshf16(inf));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::acoshf16(neg_inf));
  EXPECT_MATH_ERRNO(EDOM);
}

TEST_F(LlvmLibcAcoshf16Test, PositiveRange) {
  for (uint16_t v = START; v <= STOP; ++v) {
    float16 x = FPBits(v).get_val();

    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Acosh, x,
                                   LIBC_NAMESPACE::acoshf16(x), 3.0);
  }
}

TEST_F(LlvmLibcAcoshf16Test, SpecificBitPatterns) {
  constexpr int N = 12;
  constexpr uint16_t INPUTS[N] = {
      0x3C00, // x = 1.0
      0x3C01, // x = just above 1.0 (minimally larger than 1)
      0x3E00, // x = 1.5
      0x4200, // x = 3.0
      0x4500, // x = 5.0
      0x4900, // x = 10.0
      0x51FF, // x = ~47.94
      0x5CB0, // x = ~300.0
      0x643F, // x = ~1087.6
      0x77FF, // x = just below next exponent interval (max for exponent 0x1D)
      0x7801, // x = just above previous value (min for exponent 0x1E)
      0x7BFF  // x = 65504.0 (max finite half)
  };
  for (int i = 0; i < N; ++i) {
    float16 x = FPBits(INPUTS[i]).get_val();
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Acosh, x,
                                   LIBC_NAMESPACE::acoshf16(x), 0.5);
  }
}

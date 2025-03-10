//===-- Unittests for acoshf16 ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/errno/libc_errno.h"
#include "src/math/acoshf16.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

using LlvmLibcAcoshf16Test = LIBC_NAMESPACE::testing::FPTest<float16>;
namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

TEST_F(LlvmLibcAcoshf16Test, SpecialNumbers) {
  LIBC_NAMESPACE::libc_errno = 0;

  // NaN input
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::acoshf16(aNaN));
  EXPECT_MATH_ERRNO(0);

  // acoshf16(1.0) = 0
  EXPECT_FP_EQ_ALL_ROUNDING(float16(0.0f), LIBC_NAMESPACE::acoshf16(float16(1.0f)));
  EXPECT_MATH_ERRNO(0);

  // Domain error (x < 1)
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::acoshf16(float16(0.5f)));
  EXPECT_MATH_ERRNO(EDOM);

  // acoshf16(+inf) = inf
  EXPECT_FP_EQ_ALL_ROUNDING(inf, LIBC_NAMESPACE::acoshf16(inf));
  EXPECT_MATH_ERRNO(0);

  // acoshf16(x) domain error (negative infinity)
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN, LIBC_NAMESPACE::acoshf16(neg_inf));
  EXPECT_MATH_ERRNO(EDOM);
}

TEST_F(LlvmLibcAcoshf16Test, InFloat16Range) {
  constexpr uint16_t START = 0x3c00U; // 1.0
  constexpr uint16_t STOP = 0x7bffU;  // Largest finite float16 value

  for (uint16_t bits = START; bits <= STOP; ++bits) {
    float16 x = FPBits(bits).get_val();
    ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Acosh, float(x),
                                   float(LIBC_NAMESPACE::acoshf16(x)), 0.5);
  }
}

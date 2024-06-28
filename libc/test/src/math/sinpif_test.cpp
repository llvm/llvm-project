//===-- Unittests for sinpif ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/math_macros.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/errno/libc_errno.h"
#include "src/math/sinpif.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "test/src/math/sdcomp26094.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

#include <errno.h>
#include <stdint.h>

using LlvmLibcSinpifTest = LIBC_NAMESPACE::testing::FPTest<float>;

using LIBC_NAMESPACE::testing::SDCOMP26094_VALUES;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

TEST_F(LlvmLibcSinpifTest, SpecialNumbers) {
  LIBC_NAMESPACE::libc_errno = 0;

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::sinpif(aNaN));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(0.0f, LIBC_NAMESPACE::sinpif(0.0f));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(-0.0f, LIBC_NAMESPACE::sinpif(-0.0f));
  EXPECT_MATH_ERRNO(0);

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::sinpif(inf));
  EXPECT_MATH_ERRNO(EDOM);

  EXPECT_FP_EQ(aNaN, LIBC_NAMESPACE::sinpif(neg_inf));
  EXPECT_MATH_ERRNO(EDOM);
}

// For small values, sin(x) is x.
TEST_F(LlvmLibcSinpifTest, SmallValues) {
  float x = FPBits(0x1780'0000U).get_val();
  EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Sinpi, x,
                                 LIBC_NAMESPACE::sinpif(x), 0.5);

  x = FPBits(0x0040'0000U).get_val();
  EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Sinpi, x,
                                 LIBC_NAMESPACE::sinpif(x), 0.5);
}

// SDCOMP-26094: check sinpif in the cases for which the range reducer
// returns values furthest beyond its nominal upper bound of pi/4.
TEST_F(LlvmLibcSinpifTest, SDCOMP_26094) {
  for (uint32_t v : SDCOMP26094_VALUES) {
    float x = FPBits((v)).get_val();
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Sinpi, x,
                                   LIBC_NAMESPACE::sinpif(x), 0.5);
  }
}

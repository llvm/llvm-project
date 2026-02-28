//===-- Unittests for acospif (MPFR) --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/acospif.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

using LlvmLibcAcospifTest = LIBC_NAMESPACE::testing::FPTest<float>;

TEST_F(LlvmLibcAcospifTest, InFloatRange) {
  constexpr uint32_t COUNT = 100'000;
  // Test in [-1, 1]
  const uint32_t STEP = 0x3F800000 / COUNT; // step through [0, 1.0f] in bits
  for (uint32_t i = 0, v = 0; i <= COUNT; ++i, v += STEP) {
    float x = FPBits(v).get_val();
    ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Acospi, x,
                                   LIBC_NAMESPACE::acospif(x), 0.5);
    ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Acospi, -x,
                                   LIBC_NAMESPACE::acospif(-x), 0.5);
  }
}

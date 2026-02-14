//===-- Unittests for asinpi (MPFR) ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/asinpi.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

using LlvmLibcAsinpiTest = LIBC_NAMESPACE::testing::FPTest<double>;

TEST_F(LlvmLibcAsinpiTest, InDoubleRange) {
  constexpr uint64_t COUNT = 100'000;
  // Step through [0, 1.0] in double bits
  constexpr uint64_t ONE_BITS = 0x3FF0000000000000ULL;
  const uint64_t STEP = ONE_BITS / COUNT;
  for (uint64_t i = 0, v = 0; i <= COUNT; ++i, v += STEP) {
    double x = FPBits(v).get_val();
    ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Asinpi, x,
                                   LIBC_NAMESPACE::asinpi(x), 0.5);
    ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Asinpi, -x,
                                   LIBC_NAMESPACE::asinpi(-x), 0.5);
  }
}

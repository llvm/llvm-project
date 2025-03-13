//===-- Exhaustive test for cospi -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "src/math/cospi.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

#include <iostream>

using LlvmLibcCospiTest = LIBC_NAMESPACE::testing::FPTest<double>;
using namespace std;
namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

static constexpr double POS_START = 0;
static constexpr double POS_STOP = 200;

TEST_F(LlvmLibcCospiTest, PositiveRange) {
  for (double v = POS_START; v <= POS_STOP; ++v) {
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Cospi, v,
                                   LIBC_NAMESPACE::cospi(v), 0.5);
    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Cospi, -v,
                                   LIBC_NAMESPACE::cospi(-v), 0.5);
  }
}

//===-- Exhaustive test for sinpif16 --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "src/math/sinpi.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

#include <iostream>

using LlvmLibcSinpiTest = LIBC_NAMESPACE::testing::FPTest<double>;
using namespace std;
namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

static constexpr uint16_t POS_START = 0x0000000000000000U;
static constexpr uint16_t POS_STOP = 0x7FF0000000000000U;

static constexpr uint16_t NEG_START = 0x8000000000000000U;
static constexpr uint16_t NEG_STOP = 0xFFF0000000000000;

TEST_F(LlvmLibcSinpiTest, PositiveRange) {
  for (uint64_t v = POS_START; v <= POS_STOP; ++v) {
    double x = FPBits(v).get_val();

    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Sinpi, x,
                                   LIBC_NAMESPACE::sinpi(x), 0.5);
  }
}

TEST_F(LlvmLibcSinpiTest, NegativeRange) {
  for (uint64_t v = NEG_START; v <= NEG_STOP; ++v) {
    double x = FPBits(v).get_val();

    EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Sinpi, x,
                                   LIBC_NAMESPACE::sinpi(x), 0.5);
  }
}

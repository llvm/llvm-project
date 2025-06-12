//===-- Exhaustive tests for fbfloat16 function ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/bfloat16.h"

#include "utils/MPFRWrapper/MPCommon.h"

#include "src/math/fbfloat16.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcFBfloat16ExhaustiveTest =
    LIBC_NAMESPACE::testing::FPTest<bfloat16>;

// range: [0, inf]
static constexpr uint16_t POS_START = 0x0000U;
static constexpr uint16_t POS_STOP = 0x7f80U;

// range: [-0, -inf]
static constexpr uint16_t NEG_START = 0x8000U;
static constexpr uint16_t NEG_STOP = 0xff80;


using MPFRNumber = LIBC_NAMESPACE::testing::mpfr::MPFRNumber;

TEST_F(LlvmLibcFBfloat16ExhaustiveTest, PositiveRange) {
  for (uint16_t bits = POS_START; bits <= POS_STOP; bits++) {
    bfloat16 bf16_num{bits};
    MPFRNumber mpfr_num{bf16_num};
    ASSERT_FP_EQ(mpfr_num.as<float>(), bf16_num.as_float());
  }
}

TEST_F(LlvmLibcFBfloat16ExhaustiveTest, NegativeRange) {
  for (uint16_t bits = NEG_START; bits <= NEG_STOP; bits++) {
    bfloat16 bf16_num{bits};
    MPFRNumber mpfr_num{bf16_num};
    ASSERT_FP_EQ(mpfr_num.as<float>(), bf16_num.as_float());
  }
}

//===-- Unit tests for bfloat16 type --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/bfloat16.h"

#include "utils/MPFRWrapper/MPCommon.h"

#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcBfloat16ExhaustiveTest =
    LIBC_NAMESPACE::testing::FPTest<bfloat16>;

// range: [0, inf]
static constexpr uint16_t POS_START = 0x0000U;
static constexpr uint16_t POS_STOP = 0x7f80U;

// range: [-0, -inf]
static constexpr uint16_t NEG_START = 0x8000U;
static constexpr uint16_t NEG_STOP = 0xff80;


using MPFRNumber = LIBC_NAMESPACE::testing::mpfr::MPFRNumber;

// TODO: better naming?
TEST_F(LlvmLibcBfloat16ExhaustiveTest, PositiveRange) {
  for (uint16_t bits = POS_START; bits <= POS_STOP; bits++) {
    bfloat16 bf16_num{bits};
    MPFRNumber mpfr_num{bf16_num};
    
    // bfloat16 to float
    float mpfr_float = mpfr_num.as<float>();
    ASSERT_FP_EQ(mpfr_float, bf16_num.as_float());

    // float to bfloat16
    bfloat16 bf16_from_float{mpfr_float};
    MPFRNumber mpfr_num_2{mpfr_float};
    bfloat16 mpfr_bfloat = mpfr_num_2.as<bfloat16>();
    ASSERT_FP_EQ(mpfr_bfloat, bf16_from_float);
  }
}

TEST_F(LlvmLibcBfloat16ExhaustiveTest, NegativeRange) {
  for (uint16_t bits = NEG_START; bits <= NEG_STOP; bits++) {
    bfloat16 bf16_num{bits};
    MPFRNumber mpfr_num{bf16_num};
    
    // bfloat16 to float
    float mpfr_float = mpfr_num.as<float>();
    ASSERT_FP_EQ(mpfr_float, bf16_num.as_float());

    // float to bfloat16
    bfloat16 bf16_from_float{mpfr_float};
    MPFRNumber mpfr_num_2{mpfr_float};
    bfloat16 mpfr_bfloat = mpfr_num_2.as<bfloat16>();
    ASSERT_FP_EQ(mpfr_bfloat, bf16_from_float);
  }
}

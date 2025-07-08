//===-- Unit tests for bfloat16 type --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/bfloat16.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPCommon.h"

using LlvmLibcBfloat16ToFloatTest = LIBC_NAMESPACE::testing::FPTest<bfloat16>;

// range: [0, inf]
static constexpr uint16_t POS_START = 0x0000U;
static constexpr uint16_t POS_STOP = 0x7f80U;

// range: [-0, -inf]
static constexpr uint16_t NEG_START = 0x8000U;
static constexpr uint16_t NEG_STOP = 0xff80;

using MPFRNumber = LIBC_NAMESPACE::testing::mpfr::MPFRNumber;

TEST_F(LlvmLibcBfloat16ToFloatTest, PositiveRange) {
  for (uint16_t bits = POS_START; bits <= POS_STOP; bits++) {
    bfloat16 bf16_num{bits};
    MPFRNumber mpfr_num{bf16_num};

    // bfloat16 to float
    float mpfr_float = mpfr_num.as<float>();
    EXPECT_FP_EQ_ALL_ROUNDING(mpfr_float, static_cast<float>(bf16_num));

    // float to bfloat16
    bfloat16 bf16_from_float{mpfr_float};
    MPFRNumber mpfr_num_2{mpfr_float};
    bfloat16 mpfr_bfloat = mpfr_num_2.as<bfloat16>();
    EXPECT_FP_EQ_ALL_ROUNDING(mpfr_bfloat, bf16_from_float);
  }
}

TEST_F(LlvmLibcBfloat16ToFloatTest, NegativeRange) {
  for (uint16_t bits = NEG_START; bits <= NEG_STOP; bits++) {
    bfloat16 bf16_num{bits};
    MPFRNumber mpfr_num{bf16_num};

    // bfloat16 to float
    float mpfr_float = mpfr_num.as<float>();
    EXPECT_FP_EQ_ALL_ROUNDING(mpfr_float, static_cast<float>(bf16_num));

    // float to bfloat16
    bfloat16 bf16_from_float{mpfr_float};
    MPFRNumber mpfr_num_2{mpfr_float};
    bfloat16 mpfr_bfloat = mpfr_num_2.as<bfloat16>();
    EXPECT_FP_EQ_ALL_ROUNDING(mpfr_bfloat, bf16_from_float);
  }
}

TEST_F(LlvmLibcBfloat16ToFloatTest, SpecialIntegers) {
  auto test_for_int = [&](const int i) {
    bfloat16 mpfr_bfloat = MPFRNumber(i).as<bfloat16>();
    bfloat16 libc_bfloat{i};
    EXPECT_FP_EQ_ALL_ROUNDING(mpfr_bfloat, libc_bfloat);
  };

  constexpr int RANGE = 100'000;
  for (int i = -RANGE; i <= RANGE; i++) {
    test_for_int(i);
  }
}

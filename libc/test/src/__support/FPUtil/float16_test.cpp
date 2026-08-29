//===-- Unit tests for float16 type ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/float16.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPCommon.h"

using Float16 = LIBC_NAMESPACE::fputil::Float16;
using LlvmLibcFloat16ConversionTest = LIBC_NAMESPACE::testing::FPTest<Float16>;

// range: [0, inf]
static constexpr uint16_t POS_START = 0x0000U;
static constexpr uint16_t POS_STOP = 0x7c00U;

// range: [-0, -inf]
static constexpr uint16_t NEG_START = 0x8000U;
static constexpr uint16_t NEG_STOP = 0xfc00U;

using MPFRNumber = LIBC_NAMESPACE::testing::mpfr::MPFRNumber;

TEST_F(LlvmLibcFloat16ConversionTest, ToFloatPositiveRange) {
  for (uint16_t bits = POS_START; bits <= POS_STOP; bits++) {
    Float16 f16_num{bits};
    MPFRNumber mpfr_num{f16_num};

    // float16 to float
    float mpfr_float = mpfr_num.as<float>();
    EXPECT_FP_EQ_ALL_ROUNDING(mpfr_float, static_cast<float>(f16_num));

    // float to float16
    Float16 f16_from_float{mpfr_float};
    MPFRNumber mpfr_num_2{mpfr_float};
    Float16 mpfr_f16 = mpfr_num_2.as<Float16>();
    EXPECT_FP_EQ_ALL_ROUNDING(mpfr_f16, f16_from_float);
  }
}

TEST_F(LlvmLibcFloat16ConversionTest, ToFloatNegativeRange) {
  for (uint16_t bits = NEG_START; bits <= NEG_STOP; bits++) {
    Float16 f16_num{bits};
    MPFRNumber mpfr_num{f16_num};

    // float16 to float
    float mpfr_float = mpfr_num.as<float>();
    EXPECT_FP_EQ_ALL_ROUNDING(mpfr_float, static_cast<float>(f16_num));

    // float to float16
    Float16 f16_from_float{mpfr_float};
    MPFRNumber mpfr_num_2{mpfr_float};
    Float16 mpfr_f16 = mpfr_num_2.as<Float16>();
    EXPECT_FP_EQ_ALL_ROUNDING(mpfr_f16, f16_from_float);
  }
}

TEST_F(LlvmLibcFloat16ConversionTest, FromInteger) {
  constexpr int RANGE = 1'234;
  for (int i = -RANGE; i <= RANGE; i++) {
    Float16 mpfr_f16 = MPFRNumber(i).as<Float16>();
    Float16 libc_f16{i};
    EXPECT_FP_EQ_ALL_ROUNDING(mpfr_f16, libc_f16);
  }
}

TEST_F(LlvmLibcFloat16ConversionTest, CompoundAssignmentOperators) {
  constexpr Float16 VAL[] = {zero,          neg_zero,       inf,
                             neg_inf,       min_normal,     max_normal,
                             Float16(1.0f), Float16(-1.0f), Float16(2.0f),
                             Float16(3.0f)};
  // *=
  for (const Float16 &x : VAL) {
    for (const Float16 &y : VAL) {
      Float16 a = x, b = y;
      MPFRNumber mpfr_a{a}, mpfr_b{b};
      MPFRNumber mpfr_c = mpfr_a.mul(mpfr_b);
      Float16 mpfr_f16 = mpfr_c.as<Float16>();
      a *= b;
      Float16 libc_f16 = a;
      EXPECT_FP_EQ_ALL_ROUNDING(mpfr_f16, libc_f16);
    }
  }
  // /=
  for (const Float16 &x : VAL) {
    for (const Float16 &y : VAL) {
      Float16 a = x, b = y;
      MPFRNumber mpfr_a{a}, mpfr_b{b};
      MPFRNumber mpfr_c = mpfr_a.div(mpfr_b);
      Float16 mpfr_f16 = mpfr_c.as<Float16>();
      a /= b;
      Float16 libc_f16 = a;
      EXPECT_FP_EQ_ALL_ROUNDING(mpfr_f16, libc_f16);
    }
  }
  // +=
  for (const Float16 &x : VAL) {
    for (const Float16 &y : VAL) {
      Float16 a = x, b = y;
      MPFRNumber mpfr_a{a}, mpfr_b{b};
      MPFRNumber mpfr_c = mpfr_a.add(mpfr_b);
      Float16 mpfr_f16 = mpfr_c.as<Float16>();
      a += b;
      Float16 libc_f16 = a;
      EXPECT_FP_EQ_ALL_ROUNDING(mpfr_f16, libc_f16);
    }
  }
  // -=
  for (const Float16 &x : VAL) {
    for (const Float16 &y : VAL) {
      Float16 a = x, b = y;
      MPFRNumber mpfr_a{a}, mpfr_b{b};
      MPFRNumber mpfr_c = mpfr_a.sub(mpfr_b);
      Float16 mpfr_f16 = mpfr_c.as<Float16>();
      a -= b;
      Float16 libc_f16 = a;
      EXPECT_FP_EQ_ALL_ROUNDING(mpfr_f16, libc_f16);
    }
  }
}

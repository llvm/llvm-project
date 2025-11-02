//===-- Unittests for powf16 ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "src/__support/CPP/bit.h"
#include "src/math/powf16.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

using LlvmLibcPowF16Test = LIBC_NAMESPACE::testing::FPTest<float16>;
using FPBits = LIBC_NAMESPACE::fputil::FPBits<float16>;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

static constexpr float16 SELECTED_VALS[] = {
    0.83984375f16, 1.414f16, 0.0625f16, 2.5f16,
    3.140625f16,   15.5f16,  2.f16,     3.25f16};

// Test tricky inputs for selected x values against all possible y values.
TEST_F(LlvmLibcPowF16Test, TrickyInput_SelectedX_AllY) {
  for (float16 x_base : SELECTED_VALS) {
    // Only test non-negative x_base
    if (FPBits(x_base).is_neg())
      continue;

    // Loop through normal and subnormal values only (0x0001 to 0x7BFF)
    for (uint16_t y_u = 1; y_u <= 0x7BFFU; ++y_u) {
      float16 y_base = FPBits(y_u).get_val();

      // Case 1: (+x, +y) - Standard positive case
      mpfr::BinaryInput<float16> input1{x_base, y_base};
      EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Pow, input1,
                                     LIBC_NAMESPACE::powf16(x_base, y_base),
                                     0.5);

      // Case 2: (+x, -y) - Always valid for positive x
      float16 y_neg = -y_base;
      mpfr::BinaryInput<float16> input2{x_base, y_neg};
      EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Pow, input2,
                                     LIBC_NAMESPACE::powf16(x_base, y_neg),
                                     0.5);
    }

    // Case 3: (-x, +y) - Only test with positive integer y values
    for (int y_int = 1; y_int <= 2048; ++y_int) {
      float16 y_val = static_cast<float16>(y_int);
      float16 x_neg = -x_base;
      mpfr::BinaryInput<float16> input{x_neg, y_val};
      EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Pow, input,
                                     LIBC_NAMESPACE::powf16(x_neg, y_val), 0.5);
    }

    // Case 4: (-x, -y) - Only test with negative integer y values
    for (int y_int = -2048; y_int < 0; ++y_int) {
      float16 y_val = static_cast<float16>(y_int);
      float16 x_neg = -x_base;
      mpfr::BinaryInput<float16> input{x_neg, y_val};
      EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Pow, input,
                                     LIBC_NAMESPACE::powf16(x_neg, y_val), 0.5);
    }
  }
}

// Test tricky inputs for selected y values against all possible x values.
TEST_F(LlvmLibcPowF16Test, TrickyInput_SelectedY_AllX) {
  for (float16 y_base : SELECTED_VALS) {
    // Only test non-negative y_base
    if (FPBits(y_base).is_neg())
      continue;

    // Loop through normal and subnormal values only (0x0001 to 0x7BFF)
    for (uint16_t x_u = 1; x_u <= 0x7BFFU; ++x_u) {
      float16 x_base = FPBits(x_u).get_val();

      // Case 1: (+x, +y) - Standard positive case
      mpfr::BinaryInput<float16> input1{x_base, y_base};
      EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Pow, input1,
                                     LIBC_NAMESPACE::powf16(x_base, y_base),
                                     0.5);

      // Case 2: (+x, -y) - Always valid for positive x
      float16 y_neg = -y_base;
      mpfr::BinaryInput<float16> input2{x_base, y_neg};
      EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Pow, input2,
                                     LIBC_NAMESPACE::powf16(x_base, y_neg),
                                     0.5);
    }

    // Case 3: (-x, +y) - Only test with positive integer x values
    for (int x_int = 1; x_int <= 2048; ++x_int) {
      float16 x_val = static_cast<float16>(x_int);
      float16 x_neg = -x_val;
      mpfr::BinaryInput<float16> input{x_neg, y_base};
      EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Pow, input,
                                     LIBC_NAMESPACE::powf16(x_neg, y_base),
                                     0.5);
    }

    // Case 4: (-x, -y) - Only test with negative integer x values
    for (int x_int = 1; x_int <= 2048; ++x_int) {
      float16 x_val = static_cast<float16>(x_int);
      float16 x_neg = -x_val;
      float16 y_neg = -y_base;
      mpfr::BinaryInput<float16> input{x_neg, y_neg};
      EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Pow, input,
                                     LIBC_NAMESPACE::powf16(x_neg, y_neg), 0.5);
    }
  }
}

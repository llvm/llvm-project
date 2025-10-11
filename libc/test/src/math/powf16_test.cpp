//===-- Unittests for powf16 ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/powf16.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

using LlvmLibcPowF16Test = LIBC_NAMESPACE::testing::FPTest<float16>;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

static constexpr float16 SELECTED_VALS[] = {
    0.5f16, 0.83984375f16, 1.0f16, 2.0f16, 3.0f16, 3.140625f16, 15.5f16,
};

// Test selected x values against all possible y values.
TEST_F(LlvmLibcPowF16Test, SelectedX_AllY) {
  for (size_t i = 0; i < sizeof(SELECTED_VALS) / sizeof(SELECTED_VALS[0]);
       ++i) {
    float16 x = SELECTED_VALS[i];
    for (uint16_t y_u = 0; y_u <= 0x7c00U; ++y_u) {
      float16 y = FPBits(y_u).get_val();

      mpfr::BinaryInput<float16> input{x, y};
      EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Pow, input,
                                     LIBC_NAMESPACE::powf16(x, y), 0.5);
    }
  }
}

// Test selected y values against all possible x values.
TEST_F(LlvmLibcPowF16Test, SelectedY_AllX) {
  for (size_t i = 0; i < sizeof(SELECTED_VALS) / sizeof(SELECTED_VALS[0]);
       ++i) {
    float16 y = SELECTED_VALS[i];
    for (uint16_t x_u = 0; x_u <= 0x7c00U; ++x_u) {
      float16 x = FPBits(x_u).get_val();
      mpfr::BinaryInput<float16> input{x, y};
      EXPECT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Pow, input,
                                     LIBC_NAMESPACE::powf16(x, y), 0.5);
    }
  }
}

//===-- Exhaustive test for sinf ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "exhaustive_test.h"
#include "src/__support/math/sinf_float_eval.h"
#include "src/math/sinf.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

using LlvmLibcSinfExhaustiveTest =
    LlvmLibcUnaryOpExhaustiveMathTest<float, mpfr::Operation::Sin,
                                      LIBC_NAMESPACE::sinf>;

// Range: [0, Inf];
static constexpr uint32_t POS_START = 0x0000'0000U;
static constexpr uint32_t POS_STOP = 0x7f80'0000U;

TEST_F(LlvmLibcSinfExhaustiveTest, PositiveRange) {
  test_full_range_all_roundings(POS_START, POS_STOP);
}

// Range: [-Inf, 0];
static constexpr uint32_t NEG_START = 0xb000'0000U;
static constexpr uint32_t NEG_STOP = 0xff80'0000U;

TEST_F(LlvmLibcSinfExhaustiveTest, NegativeRange) {
  test_full_range_all_roundings(NEG_START, NEG_STOP);
}

// Preserve the float implementation's 3.5 ULP MPFR bound in
// round-to-nearest mode.
static float sinf_float_eval(float x) {
  return LIBC_NAMESPACE::math::float_eval::sinf(x);
}

using LlvmLibcSinfFloatExhaustiveTest =
    LlvmLibcUnaryOpExhaustiveMathTest<float, mpfr::Operation::Sin,
                                      sinf_float_eval, 3>;

TEST_F(LlvmLibcSinfFloatExhaustiveTest, PositiveRange) {
  test_full_range(mpfr::RoundingMode::Nearest, POS_START, POS_STOP);
}

TEST_F(LlvmLibcSinfFloatExhaustiveTest, NegativeRange) {
  test_full_range(mpfr::RoundingMode::Nearest, 0x8000'0000U, NEG_STOP);
}

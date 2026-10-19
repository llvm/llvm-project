//===-- Exhaustive test for exp10f ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/math/exp10f_float_eval.h"
#include "src/math/exp10f.h"
#include "test/src/math/exhaustive/exhaustive_test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

// Range: [0, Inf];
static constexpr uint32_t POS_START = 0x0000'0000U;
static constexpr uint32_t POS_STOP = 0x7f80'0000U;

// Range: [-Inf, 0];
static constexpr uint32_t NEG_START = 0x8000'0000U;
static constexpr uint32_t NEG_STOP = 0xff80'0000U;

using LlvmLibcExp10fExhaustiveTest =
    LlvmLibcUnaryOpExhaustiveMathTest<float, mpfr::Operation::Exp10,
                                      LIBC_NAMESPACE::exp10f>;

TEST_F(LlvmLibcExp10fExhaustiveTest, PositiveRange) {
  test_full_range_all_roundings(POS_START, POS_STOP);
}

TEST_F(LlvmLibcExp10fExhaustiveTest, NegativeRange) {
  test_full_range_all_roundings(NEG_START, NEG_STOP);
}

// Float-eval implementation: tested against the correctly rounded double
// precision version for round-to-nearest with 1 ULP bound.
static float exp10f_float_eval(float x) {
  return LIBC_NAMESPACE::math::float_eval::exp10f(x);
}

using LlvmLibcExp10fFloatExhaustiveTest =
    LlvmLibcUnaryOpAgainstBaselineExhaustiveMathTest<
        float, LIBC_NAMESPACE::exp10f, exp10f_float_eval, 1>;

TEST_F(LlvmLibcExp10fFloatExhaustiveTest, PositiveRange) {
  test_full_range(mpfr::RoundingMode::Nearest, POS_START, POS_STOP);
}

TEST_F(LlvmLibcExp10fFloatExhaustiveTest, NegativeRange) {
  test_full_range(mpfr::RoundingMode::Nearest, NEG_START, NEG_STOP);
}

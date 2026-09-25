//===-- Exhaustive test for expf ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/math/expf_double_eval.h"
#include "src/__support/math/expf_float_eval.h"
#include "src/__support/math/expf_integer_eval.h"
#include "src/math/expf.h"
#include "test/src/math/exhaustive/exhaustive_test.h"
#include "test/src/math/exhaustive/exhaustive_test_static_rounding.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

// Range: [0, Inf];
static constexpr uint32_t POS_START = 0x0000'0000U;
static constexpr uint32_t POS_STOP = 0x7f80'0000U;

// Range: [-Inf, 0];
static constexpr uint32_t NEG_START = 0x8000'0000U;
static constexpr uint32_t NEG_STOP = 0xff80'0000U;

using LlvmLibcExpfExhaustiveTest =
    LlvmLibcUnaryOpExhaustiveMathTest<float, mpfr::Operation::Exp,
                                      LIBC_NAMESPACE::expf>;

TEST_F(LlvmLibcExpfExhaustiveTest, PositiveRange) {
  test_full_range_all_roundings(POS_START, POS_STOP);
}

TEST_F(LlvmLibcExpfExhaustiveTest, NegativeRange) {
  test_full_range_all_roundings(NEG_START, NEG_STOP);
}

// Float-eval implementation: tested against the correctly rounded double
// precision version for round-to-nearest with 1 ULP bound.
static float expf_float_eval(float x) {
  return LIBC_NAMESPACE::math::float_eval::expf(x);
}

using LlvmLibcExpfFloatExhaustiveTest =
    LlvmLibcUnaryOpAgainstBaselineExhaustiveMathTest<
        float, LIBC_NAMESPACE::expf, expf_float_eval, 1>;

TEST_F(LlvmLibcExpfFloatExhaustiveTest, PositiveRange) {
  test_full_range(mpfr::RoundingMode::Nearest, POS_START, POS_STOP);
}

TEST_F(LlvmLibcExpfFloatExhaustiveTest, NegativeRange) {
  test_full_range(mpfr::RoundingMode::Nearest, NEG_START, NEG_STOP);
}

// Statically rounded implementation: tested against double_eval across all
// roundings.
using LlvmLibcExpfStaticRoundingExhaustiveTest =
    LlvmLibcStaticallyRoundedUnaryOpExhaustiveMathTest<
        float, LIBC_NAMESPACE::math::double_eval::expf,
        LIBC_NAMESPACE::shared::math::static_rounding::expf>;

TEST_F(LlvmLibcExpfStaticRoundingExhaustiveTest, PositiveRange) {
  test_full_range_all_roundings(POS_START, POS_STOP);
}

TEST_F(LlvmLibcExpfStaticRoundingExhaustiveTest, NegativeRange) {
  test_full_range_all_roundings(NEG_START, NEG_STOP);
}

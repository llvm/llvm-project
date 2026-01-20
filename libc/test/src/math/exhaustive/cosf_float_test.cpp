//===-- Exhaustive test for cosf - float-only -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "exhaustive_test.h"
#include "src/__support/math/sincosf_float_eval.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

float cosf_fast(float x) {
  return LIBC_NAMESPACE::math::sincosf_float_eval::sincosf_eval<
      /*IS_SIN*/ false>(x);
}

using LlvmLibcCosfExhaustiveTest =
    LlvmLibcUnaryOpExhaustiveMathTest<float, mpfr::Operation::Cos, cosf_fast,
                                      3>;

// Range: [0, Inf];
static constexpr uint32_t POS_START = 0x0000'0000U;
static constexpr uint32_t POS_STOP = 0x7f80'0000U;

TEST_F(LlvmLibcCosfExhaustiveTest, PostiveRange) {
  std::cout << "-- Testing for FE_TONEAREST in range [0x" << std::hex
            << POS_START << ", 0x" << POS_STOP << ") --" << std::dec
            << std::endl;
  test_full_range(mpfr::RoundingMode::Nearest, POS_START, POS_STOP);
}

// Range: [-Inf, 0];
static constexpr uint32_t NEG_START = 0x8000'0000U;
static constexpr uint32_t NEG_STOP = 0xff80'0000U;

TEST_F(LlvmLibcCosfExhaustiveTest, NegativeRange) {
  std::cout << "-- Testing for FE_TONEAREST in range [0x" << std::hex
            << NEG_START << ", 0x" << NEG_STOP << ") --" << std::dec
            << std::endl;
  test_full_range(mpfr::RoundingMode::Nearest, NEG_START, NEG_STOP);
}

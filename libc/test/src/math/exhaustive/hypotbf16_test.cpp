//===-- Exhaustive test for hypotbf16 -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "exhaustive_test.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/Hypot.h"
#include "src/__support/FPUtil/bfloat16.h"
#include "src/math/hypotbf16.h"
#include "test/UnitTest/FPMatcher.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

struct Hypotbf16Checker : public virtual LIBC_NAMESPACE::testing::Test {
  using FloatType = bfloat16;
  using FPBits = LIBC_NAMESPACE::fputil::FPBits<bfloat16>;
  using StorageType = typename FPBits::StorageType;

  uint64_t check(uint16_t x_start, uint16_t x_stop, uint16_t y_start,
                 uint16_t y_stop, mpfr::RoundingMode rounding) {
    mpfr::ForceRoundingMode r(rounding);
    if (!r.success)
      return true;
    uint16_t xbits = x_start;
    uint64_t failed = 0;
    do {
      bfloat16 x = FPBits(xbits).get_val();
      uint16_t ybits = y_start;
      do {
        bfloat16 y = FPBits(ybits).get_val();
        mpfr::BinaryInput<bfloat16> input{x, y};
        bool correct = TEST_MPFR_MATCH_ROUNDING_SILENTLY(
            mpfr::Operation::Hypot, input, LIBC_NAMESPACE::hypotbf16(x, y), 0.5,
            rounding);
        failed += (!correct);
      } while (ybits++ < y_stop);
    } while (xbits++ < x_stop);
    return failed;
  }
};

using LlvmLibcHypotBf16ExhaustiveTest =
    LlvmLibcExhaustiveMathTest<Hypotbf16Checker, 1 << 2>;

// range: [0, inf]
static constexpr uint16_t POS_START = 0x0000U;
static constexpr uint16_t POS_STOP = 0x7f80U;

// range: [-0, -inf]
static constexpr uint16_t NEG_START = 0x8000U;
static constexpr uint16_t NEG_STOP = 0xff80U;

TEST_F(LlvmLibcHypotBf16ExhaustiveTest, PositiveRange) {
  test_full_range_all_roundings(POS_START, POS_STOP, POS_START, POS_STOP);
}

TEST_F(LlvmLibcHypotBf16ExhaustiveTest, NegativeRange) {
  test_full_range_all_roundings(NEG_START, NEG_STOP, NEG_START, NEG_STOP);
}

TEST_F(LlvmLibcHypotBf16ExhaustiveTest, PositiveNegativeRange) {
  test_full_range_all_roundings(POS_START, POS_STOP, NEG_START, NEG_STOP);
}

TEST_F(LlvmLibcHypotBf16ExhaustiveTest, NegativePositiveRange) {
  test_full_range_all_roundings(NEG_START, NEG_STOP, POS_START, POS_STOP);
}

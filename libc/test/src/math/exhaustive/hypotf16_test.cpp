//===-- Exhaustive test for hypotf16 --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "exhaustive_test.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/Hypot.h"
#include "src/math/hypotf16.h"
#include "test/UnitTest/FPMatcher.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

// Range of both inputs: [0, inf]
static constexpr uint16_t START = 0x0000U;
static constexpr uint16_t STOP = 0x7C00U;

struct Hypotf16Checker : public virtual LIBC_NAMESPACE::testing::Test {
  using FloatType = float16;
  using FPBits = LIBC_NAMESPACE::fputil::FPBits<float16>;
  using StorageType = typename FPBits::StorageType;

  uint64_t check(uint16_t start, uint16_t stop, mpfr::RoundingMode rounding) {
    mpfr::ForceRoundingMode r(rounding);
    if (!r.success)
      return true;
    uint16_t xbits = start;
    uint64_t failed = 0;
    do {
      float16 x = FPBits(xbits).get_val();
      uint16_t ybits = xbits;
      do {
        float16 y = FPBits(ybits).get_val();
        bool correct = TEST_FP_EQ(LIBC_NAMESPACE::fputil::hypot(x, y),
                                  LIBC_NAMESPACE::hypotf16(x, y));
        // Using MPFR will be much slower.
        // mpfr::BinaryInput<float16> input{x, y};
        // bool correct = TEST_MPFR_MATCH_ROUNDING_SILENTLY(
        //  mpfr::Operation::Hypot, input, LIBC_NAMESPACE::hypotf16(x, y),
        // 0.5,
        //  rounding);
        failed += (!correct);
      } while (ybits++ < STOP);
    } while (xbits++ < stop);
    return failed;
  }
};

using LlvmLibcHypotf16ExhaustiveTest =
    LlvmLibcExhaustiveMathTest<Hypotf16Checker>;

TEST_F(LlvmLibcHypotf16ExhaustiveTest, PositiveRange) {
  test_full_range_all_roundings(START, STOP);
}

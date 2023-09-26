//===-- Exhaustive test for hypotf ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "exhaustive_test.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/Hypot.h"
#include "src/math/hypotf.h"
#include "test/UnitTest/FPMatcher.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

struct HypotfChecker : public virtual LIBC_NAMESPACE::testing::Test {
  using FloatType = float;
  using FPBits = LIBC_NAMESPACE::fputil::FPBits<float>;
  using UIntType = typename FPBits::UIntType;

  uint64_t check(uint32_t start, uint32_t stop, mpfr::RoundingMode rounding) {
    // Range of the second input: [2^37, 2^48).
    constexpr uint32_t Y_START = (37U + 127U) << 23;
    constexpr uint32_t Y_STOP = (48U + 127U) << 23;

    mpfr::ForceRoundingMode r(rounding);
    if (!r.success)
      return true;
    uint32_t xbits = start;
    uint64_t failed = 0;
    do {
      float x = float(FPBits(xbits));
      uint32_t ybits = Y_START;
      do {
        float y = float(FPBits(ybits));
        bool correct = TEST_FP_EQ(LIBC_NAMESPACE::fputil::hypot(x, y),
                                  LIBC_NAMESPACE::hypotf(x, y));
        // Using MPFR will be much slower.
        // mpfr::BinaryInput<float> input{x, y};
        // bool correct = TEST_MPFR_MATCH_ROUNDING_SILENTLY(
        //     mpfr::Operation::Hypot, input, LIBC_NAMESPACE::hypotf(x, y), 0.5,
        //     rounding);
        failed += (!correct);
      } while (ybits++ < Y_STOP);
    } while (xbits++ < stop);
    return failed;
  }
};

using LlvmLibcHypotfExhaustiveTest = LlvmLibcExhaustiveMathTest<HypotfChecker>;

// Range of the first input: [2^23, 2^24);
static constexpr uint32_t START = (23U + 127U) << 23;
static constexpr uint32_t STOP = ((23U + 127U) << 23) + 1;

TEST_F(LlvmLibcHypotfExhaustiveTest, PositiveRange) {
  test_full_range_all_roundings(START, STOP);
}

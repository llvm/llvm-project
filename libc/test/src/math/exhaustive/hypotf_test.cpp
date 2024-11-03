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

using FPBits = __llvm_libc::fputil::FPBits<float>;

namespace mpfr = __llvm_libc::testing::mpfr;

struct LlvmLibcHypotfExhaustiveTest : public LlvmLibcExhaustiveTest<uint32_t> {
  bool check(uint32_t start, uint32_t stop,
             mpfr::RoundingMode rounding) override {
    // Range of the second input: [2^37, 2^48).
    constexpr uint32_t Y_START = (37U + 127U) << 23;
    constexpr uint32_t Y_STOP = (48U + 127U) << 23;

    mpfr::ForceRoundingMode r(rounding);
    uint32_t xbits = start;
    bool result = true;
    do {
      float x = float(FPBits(xbits));
      uint32_t ybits = Y_START;
      do {
        float y = float(FPBits(ybits));
        result &= EXPECT_FP_EQ(__llvm_libc::fputil::hypot(x, y),
                               __llvm_libc::hypotf(x, y));
        // Using MPFR will be much slower.
        // mpfr::BinaryInput<float> input{x, y};
        // EXPECT_MPFR_MATCH(mpfr::Operation::Hypot, input,
        // __llvm_libc::hypotf(x, y), 0.5,
        //                   rounding);
      } while (ybits++ < Y_STOP);
    } while (xbits++ < stop);
    return result;
  }
};

// Range of the first input: [2^23, 2^24);
static constexpr uint32_t START = (23U + 127U) << 23;
static constexpr uint32_t STOP = ((23U + 127U) << 23) + 1;

TEST_F(LlvmLibcHypotfExhaustiveTest, RoundNearestTieToEven) {
  test_full_range(START, STOP, mpfr::RoundingMode::Nearest);
}

TEST_F(LlvmLibcHypotfExhaustiveTest, RoundUp) {
  test_full_range(START, STOP, mpfr::RoundingMode::Upward);
}

TEST_F(LlvmLibcHypotfExhaustiveTest, RoundDown) {
  test_full_range(START, STOP, mpfr::RoundingMode::Downward);
}

TEST_F(LlvmLibcHypotfExhaustiveTest, RoundTowardZero) {
  test_full_range(START, STOP, mpfr::RoundingMode::TowardZero);
}

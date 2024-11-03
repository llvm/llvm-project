//===-- Exhaustive test for log1pf ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "exhaustive_test.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/log1pf.h"
#include "test/UnitTest/FPMatcher.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

using FPBits = __llvm_libc::fputil::FPBits<float>;

namespace mpfr = __llvm_libc::testing::mpfr;

struct LlvmLibcLog1pfExhaustiveTest : public LlvmLibcExhaustiveTest<uint32_t> {
  bool check(uint32_t start, uint32_t stop,
             mpfr::RoundingMode rounding) override {
    mpfr::ForceRoundingMode r(rounding);
    uint32_t bits = start;
    bool result = true;
    do {
      FPBits xbits(bits);
      float x = float(xbits);
      result &= EXPECT_MPFR_MATCH(mpfr::Operation::Log1p, x,
                                  __llvm_libc::log1pf(x), 0.5, rounding);
    } while (bits++ < stop);
    return result;
  }
};

// Range: All non-negative;
static constexpr uint32_t START = 0x0000'0000U;
static constexpr uint32_t STOP = 0x7f80'0000U;

TEST_F(LlvmLibcLog1pfExhaustiveTest, RoundNearestTieToEven) {
  test_full_range(START, STOP, mpfr::RoundingMode::Nearest);
}

TEST_F(LlvmLibcLog1pfExhaustiveTest, RoundUp) {
  test_full_range(START, STOP, mpfr::RoundingMode::Upward);
}

TEST_F(LlvmLibcLog1pfExhaustiveTest, RoundDown) {
  test_full_range(START, STOP, mpfr::RoundingMode::Downward);
}

TEST_F(LlvmLibcLog1pfExhaustiveTest, RoundTowardZero) {
  test_full_range(START, STOP, mpfr::RoundingMode::TowardZero);
}

// Range: [-1, 0];
static constexpr uint32_t NEG_START = 0x8000'0000U;
static constexpr uint32_t NEG_STOP = 0xbf7f'ffffU;

TEST_F(LlvmLibcLog1pfExhaustiveTest, NegativeRoundNearestTieToEven) {
  test_full_range(NEG_START, NEG_STOP, mpfr::RoundingMode::Nearest);
}

TEST_F(LlvmLibcLog1pfExhaustiveTest, NegativeRoundUp) {
  test_full_range(NEG_START, NEG_STOP, mpfr::RoundingMode::Upward);
}

TEST_F(LlvmLibcLog1pfExhaustiveTest, NegativeRoundDown) {
  test_full_range(NEG_START, NEG_STOP, mpfr::RoundingMode::Downward);
}

TEST_F(LlvmLibcLog1pfExhaustiveTest, NegativeRoundTowardZero) {
  test_full_range(NEG_START, NEG_STOP, mpfr::RoundingMode::TowardZero);
}

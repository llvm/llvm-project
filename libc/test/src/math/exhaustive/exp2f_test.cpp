//===-- Exhaustive test for exp2f -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "exhaustive_test.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/exp2f.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include "utils/UnitTest/FPMatcher.h"

using FPBits = __llvm_libc::fputil::FPBits<float>;

namespace mpfr = __llvm_libc::testing::mpfr;

struct LlvmLibcExp2fExhaustiveTest : public LlvmLibcExhaustiveTest<uint32_t> {
  bool check(uint32_t start, uint32_t stop,
             mpfr::RoundingMode rounding) override {
    mpfr::ForceRoundingMode r(rounding);
    uint32_t bits = start;
    bool result = true;
    do {
      FPBits xbits(bits);
      float x = float(xbits);
      result &= EXPECT_MPFR_MATCH(mpfr::Operation::Exp2, x,
                                  __llvm_libc::exp2f(x), 0.5, rounding);
    } while (bits++ < stop);
    return result;
  }
};

static constexpr int NUM_THREADS = 16;

// Range: [0, 128];
static constexpr uint32_t POS_START = 0x0000'0000U;
static constexpr uint32_t POS_STOP = 0x4300'0000U;

TEST_F(LlvmLibcExp2fExhaustiveTest, PostiveRangeRoundNearestTieToEven) {
  test_full_range(POS_START, POS_STOP, NUM_THREADS,
                  mpfr::RoundingMode::Nearest);
}

TEST_F(LlvmLibcExp2fExhaustiveTest, PostiveRangeRoundUp) {
  test_full_range(POS_START, POS_STOP, NUM_THREADS, mpfr::RoundingMode::Upward);
}

TEST_F(LlvmLibcExp2fExhaustiveTest, PostiveRangeRoundDown) {
  test_full_range(POS_START, POS_STOP, NUM_THREADS,
                  mpfr::RoundingMode::Downward);
}

TEST_F(LlvmLibcExp2fExhaustiveTest, PostiveRangeRoundTowardZero) {
  test_full_range(POS_START, POS_STOP, NUM_THREADS,
                  mpfr::RoundingMode::TowardZero);
}

// Range: [-150, 0];
static constexpr uint32_t NEG_START = 0x8000'0000U;
static constexpr uint32_t NEG_STOP = 0xc316'0000U;

TEST_F(LlvmLibcExp2fExhaustiveTest, NegativeRangeRoundNearestTieToEven) {
  test_full_range(NEG_START, NEG_STOP, NUM_THREADS,
                  mpfr::RoundingMode::Nearest);
}

TEST_F(LlvmLibcExp2fExhaustiveTest, NegativeRangeRoundUp) {
  test_full_range(NEG_START, NEG_STOP, NUM_THREADS, mpfr::RoundingMode::Upward);
}

TEST_F(LlvmLibcExp2fExhaustiveTest, NegativeRangeRoundDown) {
  test_full_range(NEG_START, NEG_STOP, NUM_THREADS,
                  mpfr::RoundingMode::Downward);
}

TEST_F(LlvmLibcExp2fExhaustiveTest, NegativeRangeRoundTowardZero) {
  test_full_range(NEG_START, NEG_STOP, NUM_THREADS,
                  mpfr::RoundingMode::TowardZero);
}

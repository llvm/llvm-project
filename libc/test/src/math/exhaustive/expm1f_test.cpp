//===-- Exhaustive test for expm1f ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "exhaustive_test.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/expm1f.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include "utils/UnitTest/FPMatcher.h"

#include <thread>

using FPBits = __llvm_libc::fputil::FPBits<float>;

namespace mpfr = __llvm_libc::testing::mpfr;

struct LlvmLibcExpm1fExhaustiveTest : public LlvmLibcExhaustiveTest<uint32_t> {
  bool check(uint32_t start, uint32_t stop,
             mpfr::RoundingMode rounding) override {
    mpfr::ForceRoundingMode r(rounding);
    uint32_t bits = stop;
    bool result = true;
    do {
      FPBits xbits(bits);
      float x = float(xbits);
      result &= EXPECT_MPFR_MATCH(mpfr::Operation::Expm1, x,
                                  __llvm_libc::expm1f(x), 0.5, rounding);
    } while (bits-- > start);
    return result;
  }
};

static const int NUM_THREADS = std::thread::hardware_concurrency();

// Range: [0, 89];
static constexpr uint32_t POS_START = 0x0000'0000U;
static constexpr uint32_t POS_STOP = 0x42b2'0000U;

TEST_F(LlvmLibcExpm1fExhaustiveTest, PostiveRangeRoundNearestTieToEven) {
  test_full_range(POS_START, POS_STOP, NUM_THREADS,
                  mpfr::RoundingMode::Nearest);
}

TEST_F(LlvmLibcExpm1fExhaustiveTest, PostiveRangeRoundUp) {
  test_full_range(POS_START, POS_STOP, NUM_THREADS, mpfr::RoundingMode::Upward);
}

TEST_F(LlvmLibcExpm1fExhaustiveTest, PostiveRangeRoundDown) {
  test_full_range(POS_START, POS_STOP, NUM_THREADS,
                  mpfr::RoundingMode::Downward);
}

TEST_F(LlvmLibcExpm1fExhaustiveTest, PostiveRangeRoundTowardZero) {
  test_full_range(POS_START, POS_STOP, NUM_THREADS,
                  mpfr::RoundingMode::TowardZero);
}

// Range: [-104, 0];
static constexpr uint32_t NEG_START = 0x8000'0000U;
static constexpr uint32_t NEG_STOP = 0xc2d0'0000U;

TEST_F(LlvmLibcExpm1fExhaustiveTest, NegativeRangeRoundNearestTieToEven) {
  test_full_range(NEG_START, NEG_STOP, NUM_THREADS,
                  mpfr::RoundingMode::Nearest);
}

TEST_F(LlvmLibcExpm1fExhaustiveTest, NegativeRangeRoundUp) {
  test_full_range(NEG_START, NEG_STOP, NUM_THREADS, mpfr::RoundingMode::Upward);
}

TEST_F(LlvmLibcExpm1fExhaustiveTest, NegativeRangeRoundDown) {
  test_full_range(NEG_START, NEG_STOP, NUM_THREADS,
                  mpfr::RoundingMode::Downward);
}

TEST_F(LlvmLibcExpm1fExhaustiveTest, NegativeRangeRoundTowardZero) {
  test_full_range(NEG_START, NEG_STOP, NUM_THREADS,
                  mpfr::RoundingMode::TowardZero);
}

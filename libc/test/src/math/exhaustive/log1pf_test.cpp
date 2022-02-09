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
#include "utils/MPFRWrapper/MPFRUtils.h"
#include "utils/UnitTest/FPMatcher.h"

using FPBits = __llvm_libc::fputil::FPBits<float>;

namespace mpfr = __llvm_libc::testing::mpfr;

struct LlvmLibclog1pfExhaustiveTest : public LlvmLibcExhaustiveTest<uint32_t> {
  void check(uint32_t start, uint32_t stop,
             mpfr::RoundingMode rounding) override {
    mpfr::ForceRoundingMode r(rounding);
    uint32_t bits = start;
    do {
      FPBits xbits(bits);
      float x = float(xbits);
      EXPECT_MPFR_MATCH(mpfr::Operation::Log1p, x, __llvm_libc::log1pf(x), 0.5,
                        rounding);
    } while (bits++ < stop);
  }
};

// Range: All non-negative;
static constexpr uint32_t START = 0x0000'0000U;
static constexpr uint32_t STOP = 0x7f80'0000U;
// Range: [-1, 0];
// static constexpr uint32_t START = 0x8000'0000U;
// static constexpr uint32_t STOP  = 0xbf80'0000U;
static constexpr int NUM_THREADS = 16;

TEST_F(LlvmLibclog1pfExhaustiveTest, RoundNearestTieToEven) {
  test_full_range(START, STOP, NUM_THREADS, mpfr::RoundingMode::Nearest);
}

TEST_F(LlvmLibclog1pfExhaustiveTest, RoundUp) {
  test_full_range(START, STOP, NUM_THREADS, mpfr::RoundingMode::Upward);
}

TEST_F(LlvmLibclog1pfExhaustiveTest, RoundDown) {
  test_full_range(START, STOP, NUM_THREADS, mpfr::RoundingMode::Downward);
}

TEST_F(LlvmLibclog1pfExhaustiveTest, RoundTowardZero) {
  test_full_range(START, STOP, NUM_THREADS, mpfr::RoundingMode::TowardZero);
}

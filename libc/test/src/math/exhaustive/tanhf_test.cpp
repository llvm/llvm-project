//===-- Exhaustive test for tanhf -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "exhaustive_test.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/tanhf.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

#include <thread>

using FPBits = __llvm_libc::fputil::FPBits<float>;

namespace mpfr = __llvm_libc::testing::mpfr;

struct LlvmLibcTanhfExhaustiveTest : public LlvmLibcExhaustiveTest<uint32_t> {
  bool check(uint32_t start, uint32_t stop,
             mpfr::RoundingMode rounding) override {
    mpfr::ForceRoundingMode r(rounding);
    uint32_t bits = start;
    bool result = true;
    do {
      FPBits xbits(bits);
      float x = float(xbits);
      result &= EXPECT_MPFR_MATCH(mpfr::Operation::Tanh, x,
                                  __llvm_libc::tanhf(x), 0.5, rounding);
    } while (bits++ < stop);
    return result;
  }
};

static const int NUM_THREADS = std::thread::hardware_concurrency();

// Range: [0, INF];
static const uint32_t POS_START = 0x0000'0000U;
static const uint32_t POS_STOP = FPBits::inf(false).uintval();

TEST_F(LlvmLibcTanhfExhaustiveTest, PostiveRangeRoundNearestTieToEven) {
  test_full_range(POS_START, POS_STOP, mpfr::RoundingMode::Nearest);
}

TEST_F(LlvmLibcTanhfExhaustiveTest, PostiveRangeRoundUp) {
  test_full_range(POS_START, POS_STOP, mpfr::RoundingMode::Upward);
}

TEST_F(LlvmLibcTanhfExhaustiveTest, PostiveRangeRoundDown) {
  test_full_range(POS_START, POS_STOP, mpfr::RoundingMode::Downward);
}

TEST_F(LlvmLibcTanhfExhaustiveTest, PostiveRangeRoundTowardZero) {
  test_full_range(POS_START, POS_STOP, mpfr::RoundingMode::TowardZero);
}

// Range: [-INF, 0];
static const uint32_t NEG_START = 0x8000'0000U;
static const uint32_t NEG_STOP = FPBits::inf(true).uintval();

TEST_F(LlvmLibcTanhfExhaustiveTest, NegativeRangeRoundNearestTieToEven) {
  test_full_range(NEG_START, NEG_STOP, mpfr::RoundingMode::Nearest);
}

TEST_F(LlvmLibcTanhfExhaustiveTest, NegativeRangeRoundUp) {
  test_full_range(NEG_START, NEG_STOP, mpfr::RoundingMode::Upward);
}

TEST_F(LlvmLibcTanhfExhaustiveTest, NegativeRangeRoundDown) {
  test_full_range(NEG_START, NEG_STOP, mpfr::RoundingMode::Downward);
}

TEST_F(LlvmLibcTanhfExhaustiveTest, NegativeRangeRoundTowardZero) {
  test_full_range(NEG_START, NEG_STOP, mpfr::RoundingMode::TowardZero);
}

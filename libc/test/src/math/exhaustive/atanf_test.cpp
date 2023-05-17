//===-- Exhaustive test for atanf -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "exhaustive_test.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/atanf.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

#include <thread>

using FPBits = __llvm_libc::fputil::FPBits<float>;

namespace mpfr = __llvm_libc::testing::mpfr;

struct LlvmLibcAtanfExhaustiveTest : public LlvmLibcExhaustiveTest<uint32_t> {
  bool check(uint32_t start, uint32_t stop,
             mpfr::RoundingMode rounding) override {
    mpfr::ForceRoundingMode r(rounding);
    uint32_t bits = start;
    bool result = true;
    do {
      FPBits xbits(bits);
      float x = float(xbits);
      result &= EXPECT_MPFR_MATCH(mpfr::Operation::Atan, x,
                                  __llvm_libc::atanf(x), 0.5, rounding);
    } while (bits++ < stop);
    return result;
  }
};

static const int NUM_THREADS = std::thread::hardware_concurrency();

// Range: [0, 1.0];
static const uint32_t POS_START = 0x0000'0000U;
static const uint32_t POS_STOP = FPBits::inf().uintval();

TEST_F(LlvmLibcAtanfExhaustiveTest, PostiveRangeRoundNearestTieToEven) {
  test_full_range(POS_START, POS_STOP, mpfr::RoundingMode::Nearest);
}

TEST_F(LlvmLibcAtanfExhaustiveTest, PostiveRangeRoundUp) {
  test_full_range(POS_START, POS_STOP, mpfr::RoundingMode::Upward);
}

TEST_F(LlvmLibcAtanfExhaustiveTest, PostiveRangeRoundDown) {
  test_full_range(POS_START, POS_STOP, mpfr::RoundingMode::Downward);
}

TEST_F(LlvmLibcAtanfExhaustiveTest, PostiveRangeRoundTowardZero) {
  test_full_range(POS_START, POS_STOP, mpfr::RoundingMode::TowardZero);
}

// Range: [-1.0, 0];
static const uint32_t NEG_START = 0x8000'0000U;
static const uint32_t NEG_STOP = FPBits::neg_inf().uintval();

TEST_F(LlvmLibcAtanfExhaustiveTest, NegativeRangeRoundNearestTieToEven) {
  test_full_range(NEG_START, NEG_STOP, mpfr::RoundingMode::Nearest);
}

TEST_F(LlvmLibcAtanfExhaustiveTest, NegativeRangeRoundUp) {
  test_full_range(NEG_START, NEG_STOP, mpfr::RoundingMode::Upward);
}

TEST_F(LlvmLibcAtanfExhaustiveTest, NegativeRangeRoundDown) {
  test_full_range(NEG_START, NEG_STOP, mpfr::RoundingMode::Downward);
}

TEST_F(LlvmLibcAtanfExhaustiveTest, NegativeRangeRoundTowardZero) {
  test_full_range(NEG_START, NEG_STOP, mpfr::RoundingMode::TowardZero);
}

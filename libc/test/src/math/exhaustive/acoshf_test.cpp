//===-- Exhaustive test for acoshf ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "exhaustive_test.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/acoshf.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

#include <thread>

using FPBits = __llvm_libc::fputil::FPBits<float>;

namespace mpfr = __llvm_libc::testing::mpfr;

struct LlvmLibcAcoshfExhaustiveTest : public LlvmLibcExhaustiveTest<uint32_t> {
  bool check(uint32_t start, uint32_t stop,
             mpfr::RoundingMode rounding) override {
    mpfr::ForceRoundingMode r(rounding);
    uint32_t bits = start;
    bool result = true;
    do {
      FPBits xbits(bits);
      float x = float(xbits);
      result &= EXPECT_MPFR_MATCH(mpfr::Operation::Acosh, x,
                                  __llvm_libc::acoshf(x), 0.5, rounding);
    } while (bits++ < stop);
    return result;
  }
};

static const int NUM_THREADS = std::thread::hardware_concurrency();

// Range: [1, Inf];
static const uint32_t POS_START = 0x3f80'0000U;
static const uint32_t POS_STOP = 0x7f80'0000U;

TEST_F(LlvmLibcAcoshfExhaustiveTest, PostiveRangeRoundNearestTieToEven) {
  test_full_range(POS_START, POS_STOP, mpfr::RoundingMode::Nearest);
}

TEST_F(LlvmLibcAcoshfExhaustiveTest, PostiveRangeRoundUp) {
  test_full_range(POS_START, POS_STOP, mpfr::RoundingMode::Upward);
}

TEST_F(LlvmLibcAcoshfExhaustiveTest, PostiveRangeRoundDown) {
  test_full_range(POS_START, POS_STOP, mpfr::RoundingMode::Downward);
}

TEST_F(LlvmLibcAcoshfExhaustiveTest, PostiveRangeRoundTowardZero) {
  test_full_range(POS_START, POS_STOP, mpfr::RoundingMode::TowardZero);
}

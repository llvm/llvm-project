//===-- Exhaustive test for log2f -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "exhaustive_test.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/log2f.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include "utils/UnitTest/FPMatcher.h"

using FPBits = __llvm_libc::fputil::FPBits<float>;

namespace mpfr = __llvm_libc::testing::mpfr;

struct LlvmLibcLog2fExhaustiveTest : public LlvmLibcExhaustiveTest<uint32_t> {
  bool check(uint32_t start, uint32_t stop,
             mpfr::RoundingMode rounding) override {
    mpfr::ForceRoundingMode r(rounding);
    uint32_t bits = start;
    bool result = true;
    do {
      FPBits xbits(bits);
      float x = float(xbits);
      result &= EXPECT_MPFR_MATCH(mpfr::Operation::Log2, x,
                                  __llvm_libc::log2f(x), 0.5, rounding);
    } while (bits++ < stop);
    return result;
  }
};

TEST_F(LlvmLibcLog2fExhaustiveTest, RoundNearestTieToEven) {
  test_full_range(/*start=*/0U, /*stop=*/0x7f80'0000U, /*nthreads=*/16,
                  mpfr::RoundingMode::Nearest);
}

TEST_F(LlvmLibcLog2fExhaustiveTest, RoundUp) {
  test_full_range(/*start=*/0U, /*stop=*/0x7f80'0000U, /*nthreads=*/16,
                  mpfr::RoundingMode::Upward);
}

TEST_F(LlvmLibcLog2fExhaustiveTest, RoundDown) {
  test_full_range(/*start=*/0U, /*stop=*/0x7f80'0000U, /*nthreads=*/16,
                  mpfr::RoundingMode::Downward);
}

TEST_F(LlvmLibcLog2fExhaustiveTest, RoundTowardZero) {
  test_full_range(/*start=*/0U, /*stop=*/0x7f80'0000U, /*nthreads=*/16,
                  mpfr::RoundingMode::TowardZero);
}

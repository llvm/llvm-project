//===-- Exhaustive test for atan2f16 --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "exhaustive_test.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/atan2f16.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

struct Atan2f16Checker : public virtual LIBC_NAMESPACE::testing::Test {
  using FloatType = float16;
  using FPBits = LIBC_NAMESPACE::fputil::FPBits<float16>;
  using StorageType = uint16_t;

  uint64_t check(uint16_t x_start, uint16_t x_stop, uint16_t y_start,
                 uint16_t y_stop, mpfr::RoundingMode rounding) {
    mpfr::ForceRoundingMode r(rounding);
    if (!r.success)
      return (x_stop > x_start) || (y_stop > y_start);
    uint16_t xbits = x_start;
    uint64_t failed = 0;
    do {
      float16 x = FPBits(xbits).get_val();
      uint16_t ybits = y_start;
      do {
        float16 y = FPBits(ybits).get_val();
        mpfr::BinaryInput<float16> input{y, x}; // atan2(y, x)
        bool correct = TEST_MPFR_MATCH_ROUNDING_SILENTLY(
            mpfr::Operation::Atan2, input, LIBC_NAMESPACE::atan2f16(y, x), 0.5,
            rounding);
        failed += (!correct);
        if (!correct) {
          EXPECT_MPFR_MATCH_ROUNDING(mpfr::Operation::Atan2, input,
                                     LIBC_NAMESPACE::atan2f16(y, x), 0.5,
                                     rounding);
        }
      } while (ybits++ < y_stop);
    } while (xbits++ < x_stop);
    return failed;
  }
};

using LlvmLibcAtan2f16ExhaustiveTest =
    LlvmLibcExhaustiveMathTest<Atan2f16Checker, 1 << 8>;

static constexpr uint16_t ALL_BITS_START = 0x0000U;
static constexpr uint16_t ALL_BITS_STOP = 0xFC00U; // finite + inf, no NaN

TEST_F(LlvmLibcAtan2f16ExhaustiveTest, AllInputs) {
  test_full_range_all_roundings(ALL_BITS_START, ALL_BITS_STOP, ALL_BITS_START,
                                ALL_BITS_STOP);
}

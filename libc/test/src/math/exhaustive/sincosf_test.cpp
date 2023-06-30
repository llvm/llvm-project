//===-- Exhaustive test for sincosf ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "exhaustive_test.h"
#include "src/math/sincosf.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

namespace mpfr = __llvm_libc::testing::mpfr;

struct SincosfChecker : public virtual __llvm_libc::testing::Test {
  using FloatType = float;
  using FPBits = __llvm_libc::fputil::FPBits<float>;
  using UIntType = uint32_t;

  uint64_t check(UIntType start, UIntType stop, mpfr::RoundingMode rounding) {
    mpfr::ForceRoundingMode r(rounding);
    if (!r.success)
      return (stop > start);
    UIntType bits = start;
    uint64_t failed = 0;
    do {
      FPBits xbits(bits);
      FloatType x = FloatType(xbits);
      FloatType sinx, cosx;
      __llvm_libc::sincosf(x, &sinx, &cosx);

      bool correct = TEST_MPFR_MATCH_ROUNDING_SILENTLY(mpfr::Operation::Sin, x,
                                                       sinx, 0.5, rounding);
      correct = correct && TEST_MPFR_MATCH_ROUNDING_SILENTLY(
                               mpfr::Operation::Cos, x, cosx, 0.5, rounding);
      failed += (!correct);
      // Uncomment to print out failed values.
      // if (!correct) {
      //   TEST_MPFR_MATCH(mpfr::Operation::Sin, x, sinx, 0.5, rounding);
      //   TEST_MPFR_MATCH(mpfr::Operation::Cos, x, cosx, 0.5, rounding);
      // }
    } while (bits++ < stop);
    return failed;
  }
};

using LlvmLibcSincosfExhaustiveTest =
    LlvmLibcExhaustiveMathTest<SincosfChecker>;

// Range: [0, Inf];
static constexpr uint32_t POS_START = 0x0000'0000U;
static constexpr uint32_t POS_STOP = 0x7f80'0000U;

TEST_F(LlvmLibcSincosfExhaustiveTest, PostiveRange) {
  test_full_range_all_roundings(POS_START, POS_STOP);
}

// Range: [-1, 0];
static constexpr uint32_t NEG_START = 0xb000'0000U;
static constexpr uint32_t NEG_STOP = 0xbf7f'ffffU;

TEST_F(LlvmLibcSincosfExhaustiveTest, NegativeRange) {
  test_full_range_all_roundings(NEG_START, NEG_STOP);
}

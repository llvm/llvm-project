//===-- Exhaustive tests for float -> bfloat16 conversion -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "exhaustive_test.h"
#include "src/__support/FPUtil/bfloat16.h"
#include "utils/MPFRWrapper/MPCommon.h"

using BFloat16 = LIBC_NAMESPACE::fputil::BFloat16;
namespace mpfr = LIBC_NAMESPACE::testing::mpfr;
using MPFRNumber = LIBC_NAMESPACE::testing::mpfr::MPFRNumber;

template <typename InType>
struct Bfloat16ConversionChecker
    : public virtual LIBC_NAMESPACE::testing::Test {
  using FloatType = InType;
  using FPBits = LIBC_NAMESPACE::fputil::FPBits<FloatType>;
  using StorageType = typename FPBits::StorageType;

  // Check in a range, return the number of failures.
  // Slightly modified version of UnaryOpChecker.
  uint64_t check(StorageType start, StorageType stop,
                 mpfr::RoundingMode rounding) {
    mpfr::ForceRoundingMode r(rounding);
    if (!r.success)
      return (stop > start);
    StorageType bits = start;
    uint64_t failed = 0;
    do {
      FPBits x_bits(bits);
      FloatType x = x_bits.get_val();

      const BFloat16 libc_bfloat{x};
      const BFloat16 mpfr_bfloat = MPFRNumber(x).as<BFloat16>();

      const bool correct =
          LIBC_NAMESPACE::testing::getMatcher<
              LIBC_NAMESPACE::testing::TestCond::EQ>(mpfr_bfloat)
              .match(libc_bfloat);

      failed += (!correct);
    } while (bits++ < stop);
    return failed;
  }
};

template <typename FloatType>
using LlvmLibcBfloat16ExhaustiveTest =
    LlvmLibcExhaustiveMathTest<Bfloat16ConversionChecker<FloatType>>;
using LlvmLibcBfloat16FromFloatTest = LlvmLibcBfloat16ExhaustiveTest<float>;

// Positive Range: [0, Inf];
constexpr uint32_t POS_START = 0x0000'0000U;
constexpr uint32_t POS_STOP = 0x7f80'0000U;

// Negative Range: [-Inf, 0];
constexpr uint32_t NEG_START = 0xb000'0000U;
constexpr uint32_t NEG_STOP = 0xff80'0000U;

TEST_F(LlvmLibcBfloat16FromFloatTest, PostiveRange) {
  test_full_range_all_roundings(POS_START, POS_STOP);
}

TEST_F(LlvmLibcBfloat16FromFloatTest, NegativeRange) {
  test_full_range_all_roundings(NEG_START, NEG_STOP);
}

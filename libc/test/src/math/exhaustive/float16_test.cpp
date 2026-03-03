//===-- Exhaustive tests for float -> float16 conversion ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "exhaustive_test.h"
#include "src/__support/FPUtil/cast.h"
#include "utils/MPFRWrapper/MPCommon.h"

using namespace LIBC_NAMESPACE::fputil;
namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

template <typename InType>
struct Float16ConversionChecker : public virtual LIBC_NAMESPACE::testing::Test {
  using FloatType = InType;
  using FPBits = LIBC_NAMESPACE::fputil::FPBits<FloatType>;
  using StorageType = typename FPBits::StorageType;

  // Check in a range, return the number of failures.
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

      const float16 libc_result = cast<float16>(x);
      const float16 mpfr_result = mpfr::MPFRNumber(x).as<float16>();

      const bool correct =
          LIBC_NAMESPACE::testing::getMatcher<
              LIBC_NAMESPACE::testing::TestCond::EQ>(mpfr_result)
              .match(libc_result);

      failed += (!correct);
    } while (bits++ < stop);
    return failed;
  }
};

template <typename FloatType>
using LlvmLibcFloat16ExhaustiveTest =
    LlvmLibcExhaustiveMathTest<Float16ConversionChecker<FloatType>>;
using LlvmLibcFloat16FromFloatTest = LlvmLibcFloat16ExhaustiveTest<float>;

// Positive Range: [0, Inf];
constexpr uint32_t POS_START = 0x0000'0000U;
constexpr uint32_t POS_STOP = 0x7f80'0000U;

// Negative Range: [-Inf, 0];
constexpr uint32_t NEG_START = 0xb000'0000U;
constexpr uint32_t NEG_STOP = 0xff80'0000U;

TEST_F(LlvmLibcFloat16FromFloatTest, PostiveRange) {
  test_full_range_all_roundings(POS_START, POS_STOP);
}

TEST_F(LlvmLibcFloat16FromFloatTest, NegativeRange) {
  test_full_range_all_roundings(NEG_START, NEG_STOP);
}

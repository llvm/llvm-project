//===-- Unittests for math_extras -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/UInt128.h" // UInt128
#include "src/__support/integer_literals.h"
#include "src/__support/math_extras.h"
#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE {

TEST(LlvmLibcBlockMathExtrasTest, mask_trailing_ones) {
  EXPECT_EQ(0_u8, (mask_leading_ones<uint8_t, 0>()));
  EXPECT_EQ(0_u8, (mask_trailing_ones<uint8_t, 0>()));
  EXPECT_EQ(0_u16, (mask_leading_ones<uint16_t, 0>()));
  EXPECT_EQ(0_u16, (mask_trailing_ones<uint16_t, 0>()));
  EXPECT_EQ(0_u32, (mask_leading_ones<uint32_t, 0>()));
  EXPECT_EQ(0_u32, (mask_trailing_ones<uint32_t, 0>()));
  EXPECT_EQ(0_u64, (mask_leading_ones<uint64_t, 0>()));
  EXPECT_EQ(0_u64, (mask_trailing_ones<uint64_t, 0>()));

  EXPECT_EQ(0x00000003_u32, (mask_trailing_ones<uint32_t, 2>()));
  EXPECT_EQ(0xC0000000_u32, (mask_leading_ones<uint32_t, 2>()));

  EXPECT_EQ(0x000007FF_u32, (mask_trailing_ones<uint32_t, 11>()));
  EXPECT_EQ(0xFFE00000_u32, (mask_leading_ones<uint32_t, 11>()));

  EXPECT_EQ(0xFFFFFFFF_u32, (mask_trailing_ones<uint32_t, 32>()));
  EXPECT_EQ(0xFFFFFFFF_u32, (mask_leading_ones<uint32_t, 32>()));
  EXPECT_EQ(0xFFFFFFFFFFFFFFFF_u64, (mask_trailing_ones<uint64_t, 64>()));
  EXPECT_EQ(0xFFFFFFFFFFFFFFFF_u64, (mask_leading_ones<uint64_t, 64>()));

  EXPECT_EQ(0x0000FFFFFFFFFFFF_u64, (mask_trailing_ones<uint64_t, 48>()));
  EXPECT_EQ(0xFFFFFFFFFFFF0000_u64, (mask_leading_ones<uint64_t, 48>()));

  EXPECT_EQ(0_u128, (mask_trailing_ones<UInt128, 0>()));
  EXPECT_EQ(0_u128, (mask_leading_ones<UInt128, 0>()));

  EXPECT_EQ(0x00000000000000007FFFFFFFFFFFFFFF_u128,
            (mask_trailing_ones<UInt128, 63>()));
  EXPECT_EQ(0xFFFFFFFFFFFFFFFE0000000000000000_u128,
            (mask_leading_ones<UInt128, 63>()));

  EXPECT_EQ(0x0000000000000000FFFFFFFFFFFFFFFF_u128,
            (mask_trailing_ones<UInt128, 64>()));
  EXPECT_EQ(0xFFFFFFFFFFFFFFFF0000000000000000_u128,
            (mask_leading_ones<UInt128, 64>()));

  EXPECT_EQ(0x0000000000000001FFFFFFFFFFFFFFFF_u128,
            (mask_trailing_ones<UInt128, 65>()));
  EXPECT_EQ(0xFFFFFFFFFFFFFFFF8000000000000000_u128,
            (mask_leading_ones<UInt128, 65>()));

  EXPECT_EQ(0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF_u128,
            (mask_trailing_ones<UInt128, 128>()));
  EXPECT_EQ(0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF_u128,
            (mask_leading_ones<UInt128, 128>()));
}

} // namespace LIBC_NAMESPACE

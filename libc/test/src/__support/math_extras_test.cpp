//===-- Unittests for math_extras -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/math_extras.h"
#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE {

TEST(LlvmLibcBlockMathExtrasTest, mask_trailing_ones) {
  EXPECT_EQ(uint8_t(0), (mask_leading_ones<uint8_t, 0>()));
  EXPECT_EQ(uint8_t(0), (mask_trailing_ones<uint8_t, 0>()));
  EXPECT_EQ(uint16_t(0), (mask_leading_ones<uint16_t, 0>()));
  EXPECT_EQ(uint16_t(0), (mask_trailing_ones<uint16_t, 0>()));
  EXPECT_EQ(uint32_t(0), (mask_leading_ones<uint32_t, 0>()));
  EXPECT_EQ(uint32_t(0), (mask_trailing_ones<uint32_t, 0>()));
  EXPECT_EQ(uint64_t(0), (mask_leading_ones<uint64_t, 0>()));
  EXPECT_EQ(uint64_t(0), (mask_trailing_ones<uint64_t, 0>()));

  EXPECT_EQ(uint32_t(0x00000003), (mask_trailing_ones<uint32_t, 2>()));
  EXPECT_EQ(uint32_t(0xC0000000), (mask_leading_ones<uint32_t, 2>()));

  EXPECT_EQ(uint32_t(0x000007FF), (mask_trailing_ones<uint32_t, 11>()));
  EXPECT_EQ(uint32_t(0xFFE00000), (mask_leading_ones<uint32_t, 11>()));

  EXPECT_EQ(uint32_t(0xFFFFFFFF), (mask_trailing_ones<uint32_t, 32>()));
  EXPECT_EQ(uint32_t(0xFFFFFFFF), (mask_leading_ones<uint32_t, 32>()));
  EXPECT_EQ(uint64_t(0xFFFFFFFFFFFFFFFF), (mask_trailing_ones<uint64_t, 64>()));
  EXPECT_EQ(uint64_t(0xFFFFFFFFFFFFFFFF), (mask_leading_ones<uint64_t, 64>()));

  EXPECT_EQ(uint64_t(0x0000FFFFFFFFFFFF), (mask_trailing_ones<uint64_t, 48>()));
  EXPECT_EQ(uint64_t(0xFFFFFFFFFFFF0000), (mask_leading_ones<uint64_t, 48>()));
}

} // namespace LIBC_NAMESPACE

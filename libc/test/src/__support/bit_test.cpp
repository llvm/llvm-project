//===-- Unittests for BlockStore ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/bit.h"
#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE {

TEST(LlvmLibcBlockBitTest, TODO) {
  // TODO Implement me.
}

TEST(LlvmLibcBlockBitTest, OffsetTo) {
  ASSERT_EQ(offset_to(0, 512), 0);
  ASSERT_EQ(offset_to(1, 512), 511);
  ASSERT_EQ(offset_to(2, 512), 510);
  ASSERT_EQ(offset_to(13, 1), 0);
  ASSERT_EQ(offset_to(13, 4), 3);
  for (unsigned int i = 0; i < 31; ++i) {
    ASSERT_EQ((offset_to(i, 1u << i) + i) % (1u << i), 0u);
  }
}

TEST(LlvmLibcBlockBitTest, RotateLeft) {
  {
    unsigned current = 1;
    for (unsigned i = 0; i < 8 * sizeof(unsigned); ++i) {
      ASSERT_EQ(1u << i, current);
      ASSERT_EQ(current, rotate_left(1u, i));
      current = rotate_left(current, 1u);
    }
    ASSERT_EQ(current, 1u);
  }
  {
    int current = 1;
    for (int i = 0; i < 8 * static_cast<int>(sizeof(int)); ++i) {
      ASSERT_EQ(1 << i, current);
      ASSERT_EQ(current, rotate_left(1, i));
      current = rotate_left(current, 1);
    }
    ASSERT_EQ(current, 1);
  }
}

TEST(LlvmLibcBlockBitTest, NextPowerOfTwo) {
  ASSERT_EQ(1u, next_power_of_two(0u));
  for (unsigned int i = 0; i < 31; ++i) {
    ASSERT_EQ(1u << (i + 1), next_power_of_two((1u << i) + 1));
    ASSERT_EQ(1u << i, next_power_of_two(1u << i));
  }
}

TEST(LlvmLibcBlockBitTest, IsPowerOfTwo) {
  ASSERT_FALSE(is_power_of_two(0u));
  ASSERT_TRUE(is_power_of_two(1u));
  for (unsigned int i = 1; i < 31; ++i) {
    ASSERT_TRUE(is_power_of_two(1u << i));
    ASSERT_FALSE(is_power_of_two((1u << i) + 1));
  }
}

} // namespace LIBC_NAMESPACE

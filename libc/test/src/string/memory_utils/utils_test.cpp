//===-- Unittests for memory_utils ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/array.h"
#include "src/string/memory_utils/utils.h"
#include "utils/UnitTest/Test.h"

namespace __llvm_libc {

TEST(LlvmLibcUtilsTest, IsPowerOfTwoOrZero) {
  static const cpp::array<bool, 65> kExpectedValues{
      1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, // 0-15
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 16-31
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 32-47
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 48-63
      1                                               // 64
  };
  for (size_t i = 0; i < kExpectedValues.size(); ++i)
    EXPECT_EQ(is_power2_or_zero(i), kExpectedValues[i]);
}

TEST(LlvmLibcUtilsTest, IsPowerOfTwo) {
  static const cpp::array<bool, 65> kExpectedValues{
      0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, // 0-15
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 16-31
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 32-47
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 48-63
      1                                               // 64
  };
  for (size_t i = 0; i < kExpectedValues.size(); ++i)
    EXPECT_EQ(is_power2(i), kExpectedValues[i]);
}

TEST(LlvmLibcUtilsTest, Log2) {
  static const cpp::array<size_t, 65> kExpectedValues{
      0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, // 0-15
      4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, // 16-31
      5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, // 32-47
      5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, // 48-63
      6                                               // 64
  };
  for (size_t i = 0; i < kExpectedValues.size(); ++i)
    EXPECT_EQ(log2(i), kExpectedValues[i]);
}

TEST(LlvmLibcUtilsTest, LEPowerOf2) {
  static const cpp::array<size_t, 65> kExpectedValues{
      0,  1,  2,  2,  4,  4,  4,  4,  8,  8,  8,  8,  8,  8,  8,  8,  // 0-15
      16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, // 16-31
      32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, // 32-47
      32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, // 48-63
      64                                                              // 64
  };
  for (size_t i = 0; i < kExpectedValues.size(); ++i)
    EXPECT_EQ(le_power2(i), kExpectedValues[i]);
}

TEST(LlvmLibcUtilsTest, GEPowerOf2) {
  static const cpp::array<size_t, 66> kExpectedValues{
      0,  1,  2,  4,  4,  8,  8,  8,  8,  16, 16, 16, 16, 16, 16, 16, // 0-15
      16, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, // 16-31
      32, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, // 32-47
      64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, // 48-63
      64, 128                                                         // 64-65
  };
  for (size_t i = 0; i < kExpectedValues.size(); ++i)
    EXPECT_EQ(ge_power2(i), kExpectedValues[i]);
}

using I = intptr_t;

// Converts an offset into a pointer.
const void *forge(size_t offset) {
  return reinterpret_cast<const void *>(offset);
}

TEST(LlvmLibcUtilsTest, OffsetToNextAligned) {
  EXPECT_EQ(offset_to_next_aligned<16>(forge(0)), I(0));
  EXPECT_EQ(offset_to_next_aligned<16>(forge(1)), I(15));
  EXPECT_EQ(offset_to_next_aligned<16>(forge(16)), I(0));
  EXPECT_EQ(offset_to_next_aligned<16>(forge(15)), I(1));
  EXPECT_EQ(offset_to_next_aligned<32>(forge(16)), I(16));
}

TEST(LlvmLibcUtilsTest, OffsetFromLastAligned) {
  EXPECT_EQ(offset_from_last_aligned<16>(forge(0)), I(0));
  EXPECT_EQ(offset_from_last_aligned<16>(forge(1)), I(1));
  EXPECT_EQ(offset_from_last_aligned<16>(forge(16)), I(0));
  EXPECT_EQ(offset_from_last_aligned<16>(forge(15)), I(15));
  EXPECT_EQ(offset_from_last_aligned<32>(forge(16)), I(16));
}

TEST(LlvmLibcUtilsTest, OffsetToNextCacheLine) {
  EXPECT_GT(LLVM_LIBC_CACHELINE_SIZE, 0);
  EXPECT_EQ(offset_to_next_cache_line(forge(0)), I(0));
  EXPECT_EQ(offset_to_next_cache_line(forge(1)),
            I(LLVM_LIBC_CACHELINE_SIZE - 1));
  EXPECT_EQ(offset_to_next_cache_line(forge(LLVM_LIBC_CACHELINE_SIZE)), I(0));
  EXPECT_EQ(offset_to_next_cache_line(forge(LLVM_LIBC_CACHELINE_SIZE - 1)),
            I(1));
}
} // namespace __llvm_libc

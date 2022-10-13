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

TEST(LlvmLibcUtilsTest, Adjust1) {
  char a;
  const size_t base_size = 10;
  for (size_t I = -2; I < 2; ++I) {
    auto *ptr = &a;
    size_t size = base_size;
    adjust(I, ptr, size);
    EXPECT_EQ(intptr_t(ptr), intptr_t(&a + I));
    EXPECT_EQ(size, base_size - I);
  }
}

TEST(LlvmLibcUtilsTest, Adjust2) {
  char a, b;
  const size_t base_size = 10;
  for (size_t I = -2; I < 2; ++I) {
    auto *p1 = &a;
    auto *p2 = &b;
    size_t size = base_size;
    adjust(I, p1, p2, size);
    EXPECT_EQ(intptr_t(p1), intptr_t(&a + I));
    EXPECT_EQ(intptr_t(p2), intptr_t(&b + I));
    EXPECT_EQ(size, base_size - I);
  }
}

TEST(LlvmLibcUtilsTest, Align1) {
  char a;
  const size_t base_size = 10;
  {
    auto *ptr = &a;
    size_t size = base_size;
    align<128>(ptr, size);
    EXPECT_TRUE(uintptr_t(ptr) % 128 == 0);
    EXPECT_GE(ptr, &a);
    EXPECT_EQ(size_t(ptr - &a), base_size - size);
  }
}

TEST(LlvmLibcUtilsTest, Align2) {
  char a, b;
  const size_t base_size = 10;
  {
    auto *p1 = &a;
    auto *p2 = &b;
    size_t size = base_size;
    align<128, Arg::_1>(p1, p2, size);
    EXPECT_TRUE(uintptr_t(p1) % 128 == 0);
    EXPECT_GE(p1, &a);
    EXPECT_GE(p2, &b);
    EXPECT_EQ(size_t(p1 - &a), base_size - size);
    EXPECT_EQ(size_t(p2 - &b), base_size - size);
  }
  {
    auto *p1 = &a;
    auto *p2 = &b;
    size_t size = base_size;
    align<128, Arg::_2>(p1, p2, size);
    EXPECT_TRUE(uintptr_t(p2) % 128 == 0);
    EXPECT_GE(p1, &a);
    EXPECT_GE(p2, &b);
    EXPECT_EQ(size_t(p1 - &a), base_size - size);
    EXPECT_EQ(size_t(p2 - &b), base_size - size);
  }
}

} // namespace __llvm_libc

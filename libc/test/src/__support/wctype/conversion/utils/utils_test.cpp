//===-- Unittests for wctype conversion utils -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/integer_literals.h"
#include "src/__support/wctype/conversion/utils/utils.h"
#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE_DECL {

namespace wctype_internal {

namespace conversion_utils {

LIBC_INLINE static constexpr size_t operator""_usize(unsigned long long value) {
  return static_cast<size_t>(value);
}

TEST(LlvmLibcMulHighTest, BasicCases) {
  EXPECT_EQ(mul_high(0_u64, 123_u64), 0_u64);
  EXPECT_EQ(mul_high(1_u64, 1_u64), 0_u64);
}

TEST(LlvmLibcMulHighTest, LargeValues) {
  uint64_t a = 0xFFFFFFFFFFFFFFFF;
  uint64_t b = 0xFFFFFFFFFFFFFFFF;

  uint64_t result = mul_high(a, b);

  // (2^64 - 1)^2 = 2^128 - 2^65 + 1
  EXPECT_EQ(result, 0xFFFFFFFFFFFFFFFE);
}

TEST(LlvmLibcDivCeilTest, PositiveNumbers) {
  EXPECT_EQ(div_ceil(10, 3), 4);
  EXPECT_EQ(div_ceil(9, 3), 3);
}

TEST(LlvmLibcDivCeilTest, NegativeNumbers) {
  EXPECT_EQ(div_ceil(-10, 3), -3);
  EXPECT_EQ(div_ceil(10, -3), -3);
  EXPECT_EQ(div_ceil(-10, -3), 4);
}

TEST(LlvmLibcDivCeilTest, ZeroDividend) { EXPECT_EQ(div_ceil(0, 5), 0); }

TEST(LlvmLibcIsPowerOfTwoTest, Unsigned) {
  EXPECT_TRUE(is_power_of_two<uint32_t>(1));
  EXPECT_TRUE(is_power_of_two<uint32_t>(2));
  EXPECT_TRUE(is_power_of_two<uint32_t>(1024));

  EXPECT_FALSE(is_power_of_two<uint32_t>(0));
  EXPECT_FALSE(is_power_of_two<uint32_t>(3));
  EXPECT_FALSE(is_power_of_two<uint32_t>(6));
}

TEST(LlvmLibcIsPowerOfTwoSignedTest, Signed) {
  EXPECT_TRUE(is_power_of_two_signed<int>(1));
  EXPECT_TRUE(is_power_of_two_signed<int>(2));
  EXPECT_TRUE(is_power_of_two_signed<int>(8));

  EXPECT_FALSE(is_power_of_two_signed<int>(0));
  EXPECT_FALSE(is_power_of_two_signed<int>(-2));
  EXPECT_FALSE(is_power_of_two_signed<int>(3));
}

TEST(LlvmLibcMapTest, MapsValues) {
  cpp::array<int, 4> input{1, 2, 3, 4};

  auto result = map(input, [](int x) { return x * 2; });

  EXPECT_EQ(result[0], 2);
  EXPECT_EQ(result[1], 4);
  EXPECT_EQ(result[2], 6);
  EXPECT_EQ(result[3], 8);
}

TEST(LlvmLibcTryForEachTest, CompletesNormally) {
  cpp::array<int, 3> data{1, 2, 3};

  bool result = try_for_each(data, [](int x) { return x < 10; });

  EXPECT_TRUE(result);
}

TEST(LlvmLibcTryForEachTest, EarlyExit) {
  cpp::array<int, 4> data{1, 2, 3, 4};

  int visited = 0;
  bool result = try_for_each(data, [&](int x) {
    visited++;
    return x < 3;
  });

  EXPECT_FALSE(result);
  EXPECT_EQ(visited, 3);
}

TEST(LlvmLibcSumTest, BasicSum) {
  size_t data[] = {1, 2, 3, 4};
  Slice<size_t> s(data, 4);

  EXPECT_EQ(sum(s), 10_u64);
}

TEST(LlvmLibcSumTest, EmptySlice) {
  Slice<size_t> s;
  EXPECT_EQ(sum(s), 0_u64);
}

TEST(LlvmLibcWrappingAddTest, NoOverflow) {
  EXPECT_EQ(wrapping_add(10, 20), 30);
}

TEST(LlvmLibcWrappingAddTest, Overflow) {
  EXPECT_EQ(wrapping_add<uint8_t>(255_u8, 1_u8), 0_u8);
}

TEST(LlvmLibcWrappingMulTest, Basic) { EXPECT_EQ(wrapping_mul(5, 3), 15); }

TEST(LlvmLibcWrappingMulTest, Overflow) {
  EXPECT_EQ(wrapping_mul<uint8_t>(128_u8, 2_u8), 0_u8);
}

TEST(LlvmLibcCountZerosTest, MixedValues) {
  cpp::array<int, 6> arr{0, 1, 0, 2, 0, 3};

  EXPECT_EQ((count_zeros(arr)), 3_usize);
}

TEST(LlvmLibcCountZerosTest, NoZeros) {
  cpp::array<int, 3> arr{1, 2, 3};

  EXPECT_EQ((count_zeros(arr)), 0_usize);
}

TEST(LlvmLibcRotateRightTest, BasicRotation) {
  uint8_t value = 0b00000001_u8;
  uint8_t result = rotate_right(value, 1);

  EXPECT_EQ(result, 0b10000000_u8);
}

TEST(LlvmLibcRotateRightTest, F_u64Rotation) {
  uint16_t value = 0x1234;
  EXPECT_EQ(rotate_right(value, 16), value);
}

TEST(LlvmLibcToLeBytesTest, ByteOrder) {
  uint32_t value = 0x11223344;
  auto bytes = to_le_bytes(value);

  EXPECT_EQ(bytes[0], 0x44_u8);
  EXPECT_EQ(bytes[1], 0x33_u8);
  EXPECT_EQ(bytes[2], 0x22_u8);
  EXPECT_EQ(bytes[3], 0x11_u8);
}

TEST(LlvmLibcPtrBitCastTest, BitCopy) {
  uint32_t value = 0xAABBCCDD;
  auto bytes = ptr_bit_cast<cpp::array<uint8_t, 4>>(&value);

  EXPECT_EQ(bytes[0], 0xDD_u8);
  EXPECT_EQ(bytes[1], 0xCC_u8);
  EXPECT_EQ(bytes[2], 0xBB_u8);
  EXPECT_EQ(bytes[3], 0xAA_u8);
}

TEST(LlvmLibcArraySortTest, AlreadySorted) {
  cpp::array<int, 4> arr{1, 2, 3, 4};
  auto result = array_sort(arr);

  EXPECT_EQ(result[0], arr[0]);
  EXPECT_EQ(result[1], arr[1]);
  EXPECT_EQ(result[2], arr[2]);
  EXPECT_EQ(result[3], arr[3]);
}

TEST(LlvmLibcArraySortTest, ReverseOrder) {
  cpp::array<int, 5> arr{5, 4, 3, 2, 1};
  cpp::array<int, 5> result = array_sort(arr);

  EXPECT_EQ(result[0], 1);
  EXPECT_EQ(result[4], 5);
}

TEST(LlvmLibcRangeTest, ForwardIteration) {
  Range r(0, 5);
  int expected = 0;

  for (int v : r) {
    EXPECT_EQ(v, expected++);
  }

  EXPECT_EQ(expected, 5);
}

TEST(LlvmLibcRangeTest, ReverseIteration) {
  Range r(0, 5);
  auto rev = r.rev();

  int expected = 4;
  for (int v : rev) {
    EXPECT_EQ(v, expected--);
  }
}

TEST(LlvmLibcChunksMutTest, ChunkSizes) {
  cpp::array<int, 5> arr{1, 2, 3, 4, 5};
  auto chunks = chunks_mut(arr, 2);

  EXPECT_EQ(chunks.size(), 3_usize);

  EXPECT_EQ(chunks[0].size(), 2_usize);
  EXPECT_EQ(chunks[1].size(), 2_usize);
  EXPECT_EQ(chunks[2].size(), 1_usize);
}

TEST(LlvmLibcChunksMutTest, MutatesUnderlyingArray) {
  cpp::array<int, 4> arr{1, 2, 3, 4};
  auto chunks = chunks_mut(arr, 2);

  chunks[1][0] = 99;

  EXPECT_EQ(arr[2], 99);
}

} // namespace conversion_utils

} // namespace wctype_internal

} // namespace LIBC_NAMESPACE_DECL

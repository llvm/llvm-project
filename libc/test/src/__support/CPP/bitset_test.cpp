//===-- Unittests for bitset ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/bitset.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcBitsetTest, SetBitForSizeEqualToOne) {
  LIBC_NAMESPACE::cpp::bitset<1> bitset;
  EXPECT_FALSE(bitset.test(0));
  bitset.set(0);
  EXPECT_TRUE(bitset.test(0));
}

TEST(LlvmLibcBitsetTest, SetsBitsForSizeEqualToTwo) {
  LIBC_NAMESPACE::cpp::bitset<2> bitset;
  bitset.set(0);
  EXPECT_TRUE(bitset.test(0));
  bitset.set(1);
  EXPECT_TRUE(bitset.test(1));
}

TEST(LlvmLibcBitsetTest, SetsAllBitsForSizeLessThanEight) {
  LIBC_NAMESPACE::cpp::bitset<7> bitset;
  for (size_t i = 0; i < 7; ++i)
    bitset.set(i);
  // Verify all bits are now set.
  for (size_t j = 0; j < 7; ++j)
    EXPECT_TRUE(bitset.test(j));
}

TEST(LlvmLibcBitsetTest, SetsAllBitsForSizeLessThanSixteen) {
  LIBC_NAMESPACE::cpp::bitset<15> bitset;
  for (size_t i = 0; i < 15; ++i)
    bitset.set(i);
  // Verify all bits are now set.
  for (size_t j = 0; j < 15; ++j)
    EXPECT_TRUE(bitset.test(j));
}

TEST(LlvmLibcBitsetTest, SetsAllBitsForSizeLessThanThirtyTwo) {
  LIBC_NAMESPACE::cpp::bitset<31> bitset;
  for (size_t i = 0; i < 31; ++i)
    bitset.set(i);
  // Verify all bits are now set.
  for (size_t j = 0; j < 31; ++j)
    EXPECT_TRUE(bitset.test(j));
}

TEST(LlvmLibcBitsetTest, DefaultHasNoSetBits) {
  LIBC_NAMESPACE::cpp::bitset<64> bitset;
  for (size_t i = 0; i < 64; ++i) {
    EXPECT_FALSE(bitset.test(i));
  }
  // Same for odd number.
  LIBC_NAMESPACE::cpp::bitset<65> odd_bitset;
  for (size_t i = 0; i < 65; ++i) {
    EXPECT_FALSE(odd_bitset.test(i));
  }
}

TEST(LlvmLibcBitsetTest, SettingBitXDoesNotSetBitY) {
  for (size_t i = 0; i < 256; ++i) {
    // Initialize within the loop to start with a fresh bitset.
    LIBC_NAMESPACE::cpp::bitset<256> bitset;
    bitset.set(i);

    for (size_t neighbor = 0; neighbor < 256; ++neighbor) {
      if (neighbor == i)
        EXPECT_TRUE(bitset.test(neighbor));
      else
        EXPECT_FALSE(bitset.test(neighbor));
    }
  }
  // Same for odd number.
  for (size_t i = 0; i < 255; ++i) {

    LIBC_NAMESPACE::cpp::bitset<255> bitset;
    bitset.set(i);

    for (size_t neighbor = 0; neighbor < 255; ++neighbor) {
      if (neighbor == i)
        EXPECT_TRUE(bitset.test(neighbor));
      else
        EXPECT_FALSE(bitset.test(neighbor));
    }
  }
}

TEST(LlvmLibcBitsetTest, SettingBitXDoesNotResetBitY) {
  LIBC_NAMESPACE::cpp::bitset<128> bitset;
  for (size_t i = 0; i < 128; ++i)
    bitset.set(i);

  // Verify all bits are now set.
  for (size_t j = 0; j < 128; ++j)
    EXPECT_TRUE(bitset.test(j));
}

TEST(LlvmLibcBitsetTest, FlipTest) {
  LIBC_NAMESPACE::cpp::bitset<128> bitset;

  bitset.flip();

  // Verify all bits are now set.
  for (size_t j = 0; j < 128; ++j)
    EXPECT_TRUE(bitset.test(j));

  bitset.flip();

  // Verify all bits are now unset.
  for (size_t j = 0; j < 128; ++j)
    EXPECT_FALSE(bitset.test(j));

  // Set the even bits
  for (size_t j = 0; j < 64; ++j)
    bitset.set(j * 2);

  // Verify
  for (size_t j = 0; j < 128; ++j)
    EXPECT_EQ(bitset.test(j), (j % 2) == 0);

  bitset.flip();

  // Check that the odd set of bits is now true.
  for (size_t j = 0; j < 128; ++j)
    EXPECT_EQ(bitset.test(j), j % 2 != 0);

  // Set the first half of the bits.
  for (size_t j = 0; j < 64; ++j)
    bitset.set(j);

  // The pattern should now be 111...1110101...010

  // Flip to get 000...0001010...101
  bitset.flip();

  // Verify that the first half of bits are false and the even bits in the
  // second half are true.
  for (size_t j = 0; j < 128; ++j)
    EXPECT_EQ(bitset.test(j), (j > 63) && (j % 2 == 0));
}

TEST(LlvmLibcBitsetTest, EqualTest) {
  LIBC_NAMESPACE::cpp::bitset<128> bitset_a;
  LIBC_NAMESPACE::cpp::bitset<128> bitset_b;

  // New sets should be empty, and so they should be equal.
  ASSERT_TRUE(bitset_a == bitset_b);

  bitset_a.set(0);

  // Setting one bit should be enough.
  ASSERT_FALSE(bitset_a == bitset_b);

  bitset_b.set(64);

  // Setting the same bit on a different unit shouldn't be equal.
  ASSERT_FALSE(bitset_a == bitset_b);

  bitset_b.set(0);

  // The first unit matching shouldn't be equal.
  ASSERT_FALSE(bitset_a == bitset_b);

  bitset_a.set(64);

  // Now they should be equal.
  ASSERT_TRUE(bitset_a == bitset_b);
}

TEST(LlvmLibcBitsetTest, SetRangeTest) {
  LIBC_NAMESPACE::cpp::bitset<256> bitset;

  // Range from 1 to 1 should only set bit 1
  bitset.set_range(1, 1);

  for (size_t j = 0; j < 256; ++j)
    EXPECT_EQ(bitset.test(j), j == 1);

  // reset all bits back to 0.
  bitset.reset();

  // Range from 2 to 5 should set bits 2-5
  bitset.set_range(2, 5);
  for (size_t j = 0; j < 256; ++j)
    EXPECT_EQ(bitset.test(j), (j >= 2 && j <= 5));
  bitset.reset();

  // Check setting exactly one unit
  bitset.set_range(0, 63);
  for (size_t j = 0; j < 256; ++j)
    EXPECT_EQ(bitset.test(j), (j >= 0 && j <= 63));
  bitset.reset();

  // Check ranges across unit boundaries work.
  bitset.set_range(1, 64);
  for (size_t j = 0; j < 256; ++j)
    EXPECT_EQ(bitset.test(j), (j >= 1 && j <= 64));
  bitset.reset();

  // Same, but closer together.
  bitset.set_range(63, 64);
  for (size_t j = 0; j < 256; ++j)
    EXPECT_EQ(bitset.test(j), (j >= 63 && j <= 64));
  bitset.reset();

  // Check that ranges with a unit in the middle work.
  bitset.set_range(63, 129);
  for (size_t j = 0; j < 256; ++j)
    EXPECT_EQ(bitset.test(j), (j >= 63 && j <= 129));
  bitset.reset();

  // Check that the whole range being set works.
  bitset.set_range(0, 255);
  for (size_t j = 0; j < 256; ++j)
    EXPECT_TRUE(bitset.test(j));
  bitset.reset();
}

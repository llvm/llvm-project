//===-- Unittests for Bitset ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/Bitset.h"
#include "utils/UnitTest/Test.h"

TEST(LlvmLibcBitsetTest, SetBitForSizeEqualToOne) {
  __llvm_libc::cpp::Bitset<1> bitset;
  EXPECT_FALSE(bitset.test(0));
  bitset.set(0);
  EXPECT_TRUE(bitset.test(0));
}

TEST(LlvmLibcBitsetTest, SetsBitsForSizeEqualToTwo) {
  __llvm_libc::cpp::Bitset<2> bitset;
  bitset.set(0);
  EXPECT_TRUE(bitset.test(0));
  bitset.set(1);
  EXPECT_TRUE(bitset.test(1));
}

TEST(LlvmLibcBitsetTest, SetsAllBitsForSizeLessThanEight) {
  __llvm_libc::cpp::Bitset<7> bitset;
  for (size_t i = 0; i < 7; ++i)
    bitset.set(i);
  // Verify all bits are now set.
  for (size_t j = 0; j < 7; ++j)
    EXPECT_TRUE(bitset.test(j));
}

TEST(LlvmLibcBitsetTest, SetsAllBitsForSizeLessThanSixteen) {
  __llvm_libc::cpp::Bitset<15> bitset;
  for (size_t i = 0; i < 15; ++i)
    bitset.set(i);
  // Verify all bits are now set.
  for (size_t j = 0; j < 15; ++j)
    EXPECT_TRUE(bitset.test(j));
}

TEST(LlvmLibcBitsetTest, SetsAllBitsForSizeLessThanThirtyTwo) {
  __llvm_libc::cpp::Bitset<31> bitset;
  for (size_t i = 0; i < 31; ++i)
    bitset.set(i);
  // Verify all bits are now set.
  for (size_t j = 0; j < 31; ++j)
    EXPECT_TRUE(bitset.test(j));
}

TEST(LlvmLibcBitsetTest, DefaultHasNoSetBits) {
  __llvm_libc::cpp::Bitset<64> bitset;
  for (size_t i = 0; i < 64; ++i) {
    EXPECT_FALSE(bitset.test(i));
  }
  // Same for odd number.
  __llvm_libc::cpp::Bitset<65> odd_bitset;
  for (size_t i = 0; i < 65; ++i) {
    EXPECT_FALSE(odd_bitset.test(i));
  }
}

TEST(LlvmLibcBitsetTest, SettingBitXDoesNotSetBitY) {
  for (size_t i = 0; i < 256; ++i) {
    // Initialize within the loop to start with a fresh Bitset.
    __llvm_libc::cpp::Bitset<256> bitset;
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

    __llvm_libc::cpp::Bitset<255> bitset;
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
  __llvm_libc::cpp::Bitset<128> bitset;
  for (size_t i = 0; i < 128; ++i)
    bitset.set(i);

  // Verify all bits are now set.
  for (size_t j = 0; j < 128; ++j)
    EXPECT_TRUE(bitset.test(j));
}

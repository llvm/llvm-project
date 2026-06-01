//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for TLSFFreeStore.
///
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/limits.h"
#include "src/__support/block.h"
#include "src/__support/tlsf_freestore.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::Block;
using LIBC_NAMESPACE::TLSFFreeStore;
using LIBC_NAMESPACE::cpp::byte;

constexpr size_t BITS_PER_ENTRY =
    LIBC_NAMESPACE::cpp::numeric_limits<uintptr_t>::digits;
constexpr size_t NUM_TABLE_ENTRIES = 192 / BITS_PER_ENTRY;

// Test helper that exposes protected internals for focused unit tests.
template <size_t UNIT_SIZE, size_t STEP_SIZE_BITS, size_t NUM_STEP_BITS,
          size_t NUM_TABLE_ENTRIES_VAL>
struct TestStore : public TLSFFreeStore<UNIT_SIZE, STEP_SIZE_BITS,
                                        NUM_STEP_BITS, NUM_TABLE_ENTRIES_VAL> {
  using Base = TLSFFreeStore<UNIT_SIZE, STEP_SIZE_BITS, NUM_STEP_BITS,
                             NUM_TABLE_ENTRIES_VAL>;
  using Base::clear_bit;
  using Base::find_first_bit_set_after;
  using Base::get_bit;
  using Base::remove_first_fit_in_list;
  using Base::set_bit;
  using Base::size_to_bit_index;
  using Base::TOTAL_BITS;
};

using Store = TestStore<32, 3, 2, NUM_TABLE_ENTRIES>;

TEST(LlvmLibcTLSFFreeStoreTest, SizeToBitIndex) {
  // 1. Small sizes (linear region):
  EXPECT_EQ(Store::size_to_bit_index(0), size_t(0));
  EXPECT_EQ(Store::size_to_bit_index(31), size_t(0));
  EXPECT_EQ(Store::size_to_bit_index(32), size_t(1));
  EXPECT_EQ(Store::size_to_bit_index(63), size_t(1));
  EXPECT_EQ(Store::size_to_bit_index(64), size_t(2));
  EXPECT_EQ(Store::size_to_bit_index(992), size_t(31));
  EXPECT_EQ(Store::size_to_bit_index(1024), size_t(32));

  // 2. Large sizes (2-D exponential region):
  // Row 0 (Base 1024):
  // Col 0: [1024, 1279] -> bit index 32
  EXPECT_EQ(Store::size_to_bit_index(1025), size_t(32));
  EXPECT_EQ(Store::size_to_bit_index(1279), size_t(32));
  // Col 1: [1280, 1535] -> bit index 33
  EXPECT_EQ(Store::size_to_bit_index(1280), size_t(33));
  EXPECT_EQ(Store::size_to_bit_index(1535), size_t(33));
  // Col 2: [1536, 1791] -> bit index 34
  EXPECT_EQ(Store::size_to_bit_index(1536), size_t(34));
  EXPECT_EQ(Store::size_to_bit_index(1791), size_t(34));
  // Col 3: [1792, 2047] -> bit index 35
  EXPECT_EQ(Store::size_to_bit_index(1792), size_t(35));
  EXPECT_EQ(Store::size_to_bit_index(2047), size_t(35));

  // Row 1 (Base 2048):
  // Col 0: [2048, 2559] -> bit index 36
  EXPECT_EQ(Store::size_to_bit_index(2048), size_t(36));
  EXPECT_EQ(Store::size_to_bit_index(2559), size_t(36));
  // Col 1: [2560, 3071] -> bit index 37
  EXPECT_EQ(Store::size_to_bit_index(2560), size_t(37));
  EXPECT_EQ(Store::size_to_bit_index(3000), size_t(37));
  EXPECT_EQ(Store::size_to_bit_index(3071), size_t(37));

  // Row 2 (Base 4096):
  EXPECT_EQ(Store::size_to_bit_index(4096), size_t(40));
  EXPECT_EQ(Store::size_to_bit_index(5120), size_t(41));

  // 3. Clamping of extremely large sizes to Store::TOTAL_BITS - 1 (191):
  if constexpr (sizeof(size_t) == 8) {
    EXPECT_EQ(Store::size_to_bit_index(1ULL << 52), Store::TOTAL_BITS - 1);
  } else {
    EXPECT_EQ(Store::size_to_bit_index(4294967295ULL), size_t(120));
  }
}

TEST(LlvmLibcTLSFFreeStoreTest, BitManipulation) {
  Store store;

  // Initially all bits should be 0.
  for (size_t i = 0; i < Store::TOTAL_BITS; ++i)
    EXPECT_FALSE(store.get_bit(i));

  // Test set_bit / get_bit.
  store.set_bit(10);
  EXPECT_TRUE(store.get_bit(10));
  store.set_bit(150);
  EXPECT_TRUE(store.get_bit(150));

  // Test find_first_bit_set_after (strictly after the given index).
  EXPECT_EQ(store.find_first_bit_set_after(0), size_t(10));
  EXPECT_EQ(store.find_first_bit_set_after(5), size_t(10));
  EXPECT_EQ(store.find_first_bit_set_after(9), size_t(10));
  EXPECT_EQ(store.find_first_bit_set_after(10), size_t(150));
  EXPECT_EQ(store.find_first_bit_set_after(149), size_t(150));
  EXPECT_EQ(store.find_first_bit_set_after(150), Store::TOTAL_BITS);
  EXPECT_EQ(store.find_first_bit_set_after(151), Store::TOTAL_BITS);

  // Test clear_bit.
  store.clear_bit(10);
  EXPECT_FALSE(store.get_bit(10));
  EXPECT_EQ(store.find_first_bit_set_after(0), size_t(150));

  store.clear_bit(150);
  EXPECT_FALSE(store.get_bit(150));
  EXPECT_EQ(store.find_first_bit_set_after(0), Store::TOTAL_BITS);
}

TEST(LlvmLibcTLSFFreeStoreTest, InsertAndRemove) {
  Store store;

  alignas(Block::MIN_ALIGN) byte buf[1024];
  auto result = Block::init(buf);
  ASSERT_TRUE(result.has_value());
  Block *block = *result;
  block->mark_free();

  // Insert block.
  store.insert(block);

  // Verify that bit is set.
  size_t bit_index = Store::size_to_bit_index(block->inner_size());
  EXPECT_TRUE(store.get_bit(bit_index));

  // Verify find_first_bit_set_after.
  EXPECT_EQ(store.find_first_bit_set_after(0), bit_index);

  // Remove block.
  store.remove(block);

  // Verify that bit is cleared.
  EXPECT_FALSE(store.get_bit(bit_index));
  EXPECT_EQ(store.find_first_bit_set_after(0), Store::TOTAL_BITS);
}

TEST(LlvmLibcTLSFFreeStoreTest, RemoveFirstFitInList) {
  Store store;

  alignas(Block::MIN_ALIGN) byte buf[4096];
  auto result = Block::init(buf);
  ASSERT_TRUE(result.has_value());
  Block *block = *result;
  block->mark_free();

  // Split into block1 (size 1024) and the rest.
  auto split_res = block->split(1024);
  ASSERT_TRUE(split_res.has_value());
  Block *block1 = block;
  Block *block2 = *split_res;

  // Split block2 to get size 1120 and the rest.
  auto split_res2 = block2->split(1120);
  ASSERT_TRUE(split_res2.has_value());

  block1->mark_free();
  block2->mark_free();

  // Verify that both block1 and block2 map to the same bit index 32.
  size_t bit_index1 = Store::size_to_bit_index(block1->inner_size());
  size_t bit_index2 = Store::size_to_bit_index(block2->inner_size());
  EXPECT_EQ(bit_index1, size_t(32));
  EXPECT_EQ(bit_index2, size_t(32));

  // Insert both blocks into the store.
  store.insert(block1);
  store.insert(block2);

  // Verify that bit 32 is set.
  EXPECT_TRUE(store.get_bit(32));

  // Request a size (1050) that only block2 can fit.
  Block *removed = store.remove_first_fit_in_list(32, 1050);
  EXPECT_EQ(removed, block2);

  // Verify that bit 32 is still set (since block1 is still in the list).
  EXPECT_TRUE(store.get_bit(32));

  // Request a size (1000) that block1 can fit.
  removed = store.remove_first_fit_in_list(32, 1000);
  EXPECT_EQ(removed, block1);

  // Verify that bit 32 is now cleared.
  EXPECT_FALSE(store.get_bit(32));
}

TEST(LlvmLibcTLSFFreeStoreTest, FindAndRemoveFit) {
  Store store;

  alignas(Block::MIN_ALIGN) byte buf[4096];
  auto result = Block::init(buf);
  ASSERT_TRUE(result.has_value());
  Block *block = *result;
  block->mark_free();

  // Split to get block1 (1120 B) and block2 (2500 B).
  auto split_res = block->split(1120);
  ASSERT_TRUE(split_res.has_value());
  Block *block1 = block;
  Block *block2 = *split_res;

  auto split_res2 = block2->split(2500);
  ASSERT_TRUE(split_res2.has_value());

  block1->mark_free();
  block2->mark_free();

  // Verify their bit indexes.
  EXPECT_EQ(Store::size_to_bit_index(block1->inner_size()), size_t(32));
  EXPECT_EQ(Store::size_to_bit_index(block2->inner_size()), size_t(36));

  store.insert(block1);
  store.insert(block2);

  // 1. Test oversized bin search path (guaranteed O(1) fit takes priority).
  Block *removed = store.find_and_remove_fit(1050);
  EXPECT_EQ(removed, block2);

  // Verify that bit 36 is now cleared and bit 32 is still set.
  EXPECT_FALSE(store.get_bit(36));
  EXPECT_TRUE(store.get_bit(32));

  // 2. Test exact size-class search path (fallback linear scan).
  removed = store.find_and_remove_fit(1050);
  EXPECT_EQ(removed, block1);
  EXPECT_FALSE(store.get_bit(32));
}

TEST(LlvmLibcTLSFFreeStoreTest, FindAndRemoveFitSkipsNonFitExactClass) {
  Store store;

  alignas(Block::MIN_ALIGN) byte buf[4096];
  auto result = Block::init(buf);
  ASSERT_TRUE(result.has_value());
  Block *block = *result;
  block->mark_free();

  // Split into block1 (size 1024) and the rest.
  auto split_res = block->split(1024);
  ASSERT_TRUE(split_res.has_value());
  Block *block1 = block;
  Block *block2 = *split_res;

  auto split_res2 = block2->split(2500);
  ASSERT_TRUE(split_res2.has_value());

  block1->mark_free();
  block2->mark_free();

  EXPECT_EQ(Store::size_to_bit_index(block1->inner_size()), size_t(32));
  EXPECT_EQ(Store::size_to_bit_index(block2->inner_size()), size_t(36));

  store.insert(block1);
  store.insert(block2);

  Block *removed = store.find_and_remove_fit(1050);
  EXPECT_EQ(removed, block2);
  EXPECT_TRUE(store.get_bit(32));
  EXPECT_FALSE(store.get_bit(36));
}

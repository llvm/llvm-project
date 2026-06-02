//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for a FreeTable.
///
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/limits.h"
#include "src/__support/block.h"
#include "src/__support/freetable.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::Block;
using LIBC_NAMESPACE::FreeTable;

// A test helper struct that inherits from FreeTable to expose its protected
// internal helper methods to the unit tests.
template <size_t UNIT_SIZE, size_t STEP_SIZE_BITS, size_t NUM_STEP_BITS,
          size_t NUM_TABLE_ENTRIES>
struct TestTable : public FreeTable<UNIT_SIZE, STEP_SIZE_BITS, NUM_STEP_BITS,
                                    NUM_TABLE_ENTRIES> {
  using Base =
      FreeTable<UNIT_SIZE, STEP_SIZE_BITS, NUM_STEP_BITS, NUM_TABLE_ENTRIES>;
  using Base::clear_bit;
  using Base::find_first_bit_set_after;
  using Base::get_bit;
  using Base::remove_first_fit_in_list;
  using Base::set_bit;
  using Base::size_to_bit_index;
  using Base::TOTAL_BITS;
};

TEST(LlvmLibcFreeTableTest, SizeToBitIndex) {
  constexpr size_t BITS_PER_ENTRY =
      LIBC_NAMESPACE::cpp::numeric_limits<uintptr_t>::digits;
  constexpr size_t NUM_TABLE_ENTRIES = 192 / BITS_PER_ENTRY;

  // Parameters matching our visual/mathematical layout example:
  // UNIT_SIZE = 32, STEP_SIZE = 8, NUM_STEPS = 4
  using Table = TestTable<32, 3, 2, NUM_TABLE_ENTRIES>;

  // 1. Small sizes (linear region):
  EXPECT_EQ(Table::size_to_bit_index(0), size_t(0));
  EXPECT_EQ(Table::size_to_bit_index(31), size_t(0));
  EXPECT_EQ(Table::size_to_bit_index(32), size_t(1));
  EXPECT_EQ(Table::size_to_bit_index(63), size_t(1));
  EXPECT_EQ(Table::size_to_bit_index(64), size_t(2));
  EXPECT_EQ(Table::size_to_bit_index(992), size_t(31));
  EXPECT_EQ(Table::size_to_bit_index(1024), size_t(32));

  // 2. Large sizes (2-D exponential region):
  // Row 0 (Base 1024):
  // Col 0: [1024, 1279] -> bit index 32
  EXPECT_EQ(Table::size_to_bit_index(1025), size_t(32));
  EXPECT_EQ(Table::size_to_bit_index(1279), size_t(32));
  // Col 1: [1280, 1535] -> bit index 33
  EXPECT_EQ(Table::size_to_bit_index(1280), size_t(33));
  EXPECT_EQ(Table::size_to_bit_index(1535), size_t(33));
  // Col 2: [1536, 1791] -> bit index 34
  EXPECT_EQ(Table::size_to_bit_index(1536), size_t(34));
  EXPECT_EQ(Table::size_to_bit_index(1791), size_t(34));
  // Col 3: [1792, 2047] -> bit index 35
  EXPECT_EQ(Table::size_to_bit_index(1792), size_t(35));
  EXPECT_EQ(Table::size_to_bit_index(2047), size_t(35));

  // Row 1 (Base 2048):
  // Col 0: [2048, 2559] -> bit index 36
  EXPECT_EQ(Table::size_to_bit_index(2048), size_t(36));
  EXPECT_EQ(Table::size_to_bit_index(2559), size_t(36));
  // Col 1: [2560, 3071] -> bit index 37
  EXPECT_EQ(Table::size_to_bit_index(2560), size_t(37));
  EXPECT_EQ(Table::size_to_bit_index(3000), size_t(37));
  EXPECT_EQ(Table::size_to_bit_index(3071), size_t(37));

  // Row 2 (Base 4096):
  EXPECT_EQ(Table::size_to_bit_index(4096), size_t(40));
  EXPECT_EQ(Table::size_to_bit_index(5120), size_t(41));

  // 3. Clamping of extremely large sizes to Table::TOTAL_BITS - 1 (191):
  if constexpr (sizeof(size_t) == 8) {
    EXPECT_EQ(Table::size_to_bit_index(1ULL << 52), Table::TOTAL_BITS - 1);
  } else {
    EXPECT_EQ(Table::size_to_bit_index(4294967295ULL), size_t(120));
  }
}

TEST(LlvmLibcFreeTableTest, BitManipulation) {
  constexpr size_t BITS_PER_ENTRY =
      LIBC_NAMESPACE::cpp::numeric_limits<uintptr_t>::digits;
  constexpr size_t NUM_TABLE_ENTRIES = 192 / BITS_PER_ENTRY;

  // 192-bit table
  using Table = TestTable<32, 3, 2, NUM_TABLE_ENTRIES>;
  Table table;

  // Initially all bits should be 0
  for (size_t i = 0; i < Table::TOTAL_BITS; ++i)
    EXPECT_FALSE(table.get_bit(i));

  // Test set_bit / get_bit
  table.set_bit(10);
  EXPECT_TRUE(table.get_bit(10));
  table.set_bit(150);
  EXPECT_TRUE(table.get_bit(150));

  // Test find_first_bit_set_after (strictly after the given index)
  EXPECT_EQ(table.find_first_bit_set_after(0), size_t(10));
  EXPECT_EQ(table.find_first_bit_set_after(5), size_t(10));
  EXPECT_EQ(table.find_first_bit_set_after(9), size_t(10));
  EXPECT_EQ(table.find_first_bit_set_after(10), size_t(150));
  EXPECT_EQ(table.find_first_bit_set_after(149), size_t(150));
  EXPECT_EQ(table.find_first_bit_set_after(150),
            Table::TOTAL_BITS); // TOTAL_BITS
  EXPECT_EQ(table.find_first_bit_set_after(151),
            Table::TOTAL_BITS); // TOTAL_BITS

  // Test clear_bit
  table.clear_bit(10);
  EXPECT_FALSE(table.get_bit(10));
  EXPECT_EQ(table.find_first_bit_set_after(0), size_t(150));

  table.clear_bit(150);
  EXPECT_FALSE(table.get_bit(150));
  EXPECT_EQ(table.find_first_bit_set_after(0), Table::TOTAL_BITS);
}

TEST(LlvmLibcFreeTableTest, InsertAndRemove) {
  constexpr size_t BITS_PER_ENTRY =
      LIBC_NAMESPACE::cpp::numeric_limits<uintptr_t>::digits;
  constexpr size_t NUM_TABLE_ENTRIES = 192 / BITS_PER_ENTRY;

  // 192-bit table
  using Table = TestTable<32, 3, 2, NUM_TABLE_ENTRIES>;
  Table table;

  alignas(Block::MIN_ALIGN) LIBC_NAMESPACE::cpp::byte buf[1024];
  auto result = Block::init(buf);
  ASSERT_TRUE(result.has_value());
  Block *block = *result;

  // Make sure block is free
  block->mark_free();

  // Insert block
  table.insert(block);

  // Verify that bit is set
  size_t bit_index = Table::size_to_bit_index(block->inner_size());
  EXPECT_TRUE(table.get_bit(bit_index));

  // Verify find_first_bit_set_after
  EXPECT_EQ(table.find_first_bit_set_after(0), bit_index);

  // Remove block
  table.remove(block);

  // Verify that bit is cleared
  EXPECT_FALSE(table.get_bit(bit_index));
  EXPECT_EQ(table.find_first_bit_set_after(0), Table::TOTAL_BITS);
}

TEST(LlvmLibcFreeTableTest, RemoveFirstFitInList) {
  constexpr size_t BITS_PER_ENTRY =
      LIBC_NAMESPACE::cpp::numeric_limits<uintptr_t>::digits;
  constexpr size_t NUM_TABLE_ENTRIES = 192 / BITS_PER_ENTRY;

  // 192-bit table
  using Table = TestTable<32, 3, 2, NUM_TABLE_ENTRIES>;
  Table table;

  alignas(Block::MIN_ALIGN) LIBC_NAMESPACE::cpp::byte buf[4096];
  auto result = Block::init(buf);
  ASSERT_TRUE(result.has_value());
  Block *block = *result;
  block->mark_free();

  // Split into block1 (size 1024) and the rest
  auto split_res = block->split(1024);
  ASSERT_TRUE(split_res.has_value());
  Block *block1 = block;
  Block *block2 = *split_res;

  // Split block2 to get size 1120 and the rest
  auto split_res2 = block2->split(1120);
  ASSERT_TRUE(split_res2.has_value());

  block1->mark_free();
  block2->mark_free();

  // Verify that both block1 and block2 map to the same bit index 32
  size_t bit_index1 = Table::size_to_bit_index(block1->inner_size());
  size_t bit_index2 = Table::size_to_bit_index(block2->inner_size());
  EXPECT_EQ(bit_index1, size_t(32));
  EXPECT_EQ(bit_index2, size_t(32));

  // Insert both blocks into the table
  table.insert(block1);
  table.insert(block2);

  // Verify that the bit 32 is set
  EXPECT_TRUE(table.get_bit(32));

  // Request a size (1050) that only block2 can fit
  Block *removed = table.remove_first_fit_in_list(32, 1050);
  EXPECT_EQ(removed, block2);

  // Verify that bit 32 is still set (since block1 is still in the list)
  EXPECT_TRUE(table.get_bit(32));

  // Request a size (1000) that block1 can fit
  removed = table.remove_first_fit_in_list(32, 1000);
  EXPECT_EQ(removed, block1);

  // Verify that bit 32 is now cleared
  EXPECT_FALSE(table.get_bit(32));
}

TEST(LlvmLibcFreeTableTest, FindAndRemoveFit) {
  constexpr size_t BITS_PER_ENTRY =
      LIBC_NAMESPACE::cpp::numeric_limits<uintptr_t>::digits;
  constexpr size_t NUM_TABLE_ENTRIES = 192 / BITS_PER_ENTRY;

  // 192-bit table
  using Table = TestTable<32, 3, 2, NUM_TABLE_ENTRIES>;
  Table table;

  alignas(Block::MIN_ALIGN) LIBC_NAMESPACE::cpp::byte buf[4096];
  auto result = Block::init(buf);
  ASSERT_TRUE(result.has_value());
  Block *block = *result;
  block->mark_free();

  // Split to get block1 (1120 B) and block2 (2500 B)
  auto split_res = block->split(1120);
  ASSERT_TRUE(split_res.has_value());
  Block *block1 = block;
  Block *block2 = *split_res;

  auto split_res2 = block2->split(2500);
  ASSERT_TRUE(split_res2.has_value());

  block1->mark_free();
  block2->mark_free();

  // Verify their bit indexes
  size_t bit_index1 = Table::size_to_bit_index(block1->inner_size());
  size_t bit_index2 = Table::size_to_bit_index(block2->inner_size());
  EXPECT_EQ(bit_index1, size_t(32));
  EXPECT_EQ(bit_index2, size_t(36));

  // 1. Test Exact Fit search path
  table.insert(block1);
  EXPECT_TRUE(table.get_bit(32));

  Block *removed = table.find_and_remove_fit(1050);
  EXPECT_EQ(removed, block1);
  EXPECT_FALSE(table.get_bit(32));

  // 2. Test Oversized bin (Best-Fit) search path
  table.insert(block2);
  EXPECT_TRUE(table.get_bit(36));

  // Request 1050 B (maps to 32). Bin 32 is empty, so it will find and pop the
  // next set bit (36)
  removed = table.find_and_remove_fit(1050);
  EXPECT_EQ(removed, block2);
  EXPECT_FALSE(table.get_bit(36));
}

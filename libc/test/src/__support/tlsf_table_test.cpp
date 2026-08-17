//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for a TLSFTable.
///
//===----------------------------------------------------------------------===//

#include "src/__support/tlsf_table.h"
#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE_DECL {

using Table = TLSFTable<DefaultFreeStoreConfig>;

TEST(LlvmLibcTLSFTableTest, SizeToBin) {
  EXPECT_EQ(Table::size_to_bin(0), static_cast<size_t>(0));
  EXPECT_EQ(Table::size_to_bin(Table::MIN_INNER_SIZE), static_cast<size_t>(0));
  EXPECT_EQ(Table::size_to_bin(Table::MIN_INNER_SIZE + 1),
            static_cast<size_t>(1));
}

TEST(LlvmLibcTLSFTableTest, BinToMinSize) {
  EXPECT_EQ(Table::bin_to_min_size(0), static_cast<size_t>(0));
  EXPECT_EQ(Table::bin_to_min_size(1),
            static_cast<size_t>(Table::MIN_INNER_SIZE + 1));

  size_t prev_size = 0;
  for (size_t i = 1; i < 64; ++i) {
    size_t min_size = Table::bin_to_min_size(i);
    EXPECT_GT(min_size, prev_size);
    prev_size = min_size;
  }
}

TEST(LlvmLibcTLSFTableTest, OccupancyQueriesAndMutations) {
  Table table;
  for (size_t i = 0; i < Table::TOTAL_BINS; ++i) {
    EXPECT_FALSE(table.is_occupied(i));
  }

  table.mark_occupied(5);
  EXPECT_TRUE(table.is_occupied(5));
  EXPECT_FALSE(table.is_occupied(4));
  EXPECT_FALSE(table.is_occupied(6));

  table.mark_occupied(100);
  EXPECT_TRUE(table.is_occupied(100));

  table.mark_vacant(5);
  EXPECT_FALSE(table.is_occupied(5));
  EXPECT_TRUE(table.is_occupied(100));
}

TEST(LlvmLibcTLSFTableTest, FindFirstOccupiedAfter) {
  Table table;
  EXPECT_EQ(table.find_first_occupied_after(0), Table::TOTAL_BINS);
  EXPECT_EQ(table.find_first_occupied_after(10), Table::TOTAL_BINS);

  table.mark_occupied(10);
  table.mark_occupied(65);

  EXPECT_EQ(table.find_first_occupied_after(0), static_cast<size_t>(10));
  EXPECT_EQ(table.find_first_occupied_after(9), static_cast<size_t>(10));
  EXPECT_EQ(table.find_first_occupied_after(10), static_cast<size_t>(65));
  EXPECT_EQ(table.find_first_occupied_after(64), static_cast<size_t>(65));
  EXPECT_EQ(table.find_first_occupied_after(65), Table::TOTAL_BINS);
}

} // namespace LIBC_NAMESPACE_DECL

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for TrieFreeStore.
///
//===----------------------------------------------------------------------===//

#include <stddef.h>

#include "src/__support/freetrie.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::Block;
using LIBC_NAMESPACE::FreeList;
using LIBC_NAMESPACE::FreeTrie;
using LIBC_NAMESPACE::TrieFreeStore;
using LIBC_NAMESPACE::cpp::byte;
using LIBC_NAMESPACE::cpp::optional;

// Inserting or removing blocks too small to be tracked does nothing.
TEST(LlvmLibcTrieFreeStore, TooSmall) {
  byte mem[1024];
  optional<Block *> maybeBlock = Block::init(mem);
  ASSERT_TRUE(maybeBlock.has_value());
  Block *too_small = *maybeBlock;
  maybeBlock = too_small->split(Block::PREV_FIELD_SIZE);
  ASSERT_TRUE(maybeBlock.has_value());
  // On platforms with high alignment the smallest legal block may be large
  // enough for a node.
  if (too_small->outer_size() >= sizeof(Block) + sizeof(FreeList::Node))
    return;
  Block *remainder = *maybeBlock;

  TrieFreeStore store;
  store.set_range({0, 4096});
  store.insert(too_small);
  store.insert(remainder);

  EXPECT_EQ(store.find_and_remove_fit(too_small->inner_size()), remainder);
  store.remove(too_small);
}

TEST(LlvmLibcTrieFreeStore, FindAndRemoveFit) {
  byte mem[1024];
  optional<Block *> maybeBlock = Block::init(mem);
  ASSERT_TRUE(maybeBlock.has_value());

  Block *smallest = *maybeBlock;
  maybeBlock = smallest->split(sizeof(FreeList::Node) + Block::PREV_FIELD_SIZE);
  ASSERT_TRUE(maybeBlock.has_value());

  Block *largest_small = *maybeBlock;
  maybeBlock = largest_small->split(sizeof(FreeTrie::Node) +
                                    Block::PREV_FIELD_SIZE - Block::MIN_ALIGN);
  ASSERT_TRUE(maybeBlock.has_value());
  if (largest_small->inner_size() == smallest->inner_size())
    largest_small = smallest;
  ASSERT_GE(largest_small->inner_size(), smallest->inner_size());

  Block *remainder = *maybeBlock;

  TrieFreeStore store;
  store.set_range({0, 4096});
  store.insert(smallest);
  if (largest_small != smallest)
    store.insert(largest_small);
  store.insert(remainder);

  // Find exact match for smallest.
  ASSERT_EQ(store.find_and_remove_fit(smallest->inner_size()), smallest);
  store.insert(smallest);

  // Find exact match for largest.
  ASSERT_EQ(store.find_and_remove_fit(largest_small->inner_size()),
            largest_small);
  store.insert(largest_small);

  // Search small list for best fit.
  Block *next_smallest = largest_small == smallest ? remainder : largest_small;
  ASSERT_EQ(store.find_and_remove_fit(smallest->inner_size() + 1),
            next_smallest);
  store.insert(next_smallest);

  // Continue search for best fit to large blocks.
  EXPECT_EQ(store.find_and_remove_fit(largest_small->inner_size() + 1),
            remainder);
}

TEST(LlvmLibcTrieFreeStore, Remove) {
  byte mem[1024];
  optional<Block *> maybeBlock = Block::init(mem);
  ASSERT_TRUE(maybeBlock.has_value());

  Block *small = *maybeBlock;
  maybeBlock = small->split(sizeof(FreeList::Node) + Block::PREV_FIELD_SIZE);
  ASSERT_TRUE(maybeBlock.has_value());

  Block *remainder = *maybeBlock;

  TrieFreeStore store;
  store.set_range({0, 4096});
  store.insert(small);
  store.insert(remainder);

  store.remove(remainder);
  ASSERT_EQ(store.find_and_remove_fit(remainder->inner_size()),
            static_cast<Block *>(nullptr));
  store.remove(small);
  ASSERT_EQ(store.find_and_remove_fit(small->inner_size()),
            static_cast<Block *>(nullptr));
}

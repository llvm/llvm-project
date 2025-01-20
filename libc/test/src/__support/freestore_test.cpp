//===-- Unittests for a freestore -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stddef.h>

#include "src/__support/freestore.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::Block;
using LIBC_NAMESPACE::FreeList;
using LIBC_NAMESPACE::FreeStore;
using LIBC_NAMESPACE::FreeTrie;
using LIBC_NAMESPACE::cpp::byte;
using LIBC_NAMESPACE::cpp::optional;

// Inserting or removing blocks too small to be tracked does nothing.
TEST(LlvmLibcFreeStore, TooSmall) {
  byte mem[1024];
  optional<Block *> maybeBlock = Block::init(mem);
  ASSERT_TRUE(maybeBlock.has_value());
  Block *too_small = *maybeBlock;
  maybeBlock = too_small->split(sizeof(size_t));
  ASSERT_TRUE(maybeBlock.has_value());
  Block *remainder = *maybeBlock;

  FreeStore store;
  store.set_range({0, 4096});
  store.insert(too_small);
  store.insert(remainder);

  EXPECT_EQ(store.remove_best_fit(too_small->inner_size()), remainder);
  store.remove(too_small);
}

TEST(LlvmLibcFreeStore, RemoveBestFit) {
  byte mem[1024];
  optional<Block *> maybeBlock = Block::init(mem);
  ASSERT_TRUE(maybeBlock.has_value());

  Block *smallest = *maybeBlock;
  maybeBlock = smallest->split(sizeof(FreeList::Node) + sizeof(size_t));
  ASSERT_TRUE(maybeBlock.has_value());

  Block *largest_small = *maybeBlock;
  maybeBlock = largest_small->split(sizeof(FreeTrie::Node) + sizeof(size_t) -
                                    alignof(max_align_t));
  ASSERT_TRUE(maybeBlock.has_value());
  if (largest_small->inner_size() == smallest->inner_size())
    largest_small = smallest;
  ASSERT_GE(largest_small->inner_size(), smallest->inner_size());

  Block *remainder = *maybeBlock;

  FreeStore store;
  store.set_range({0, 4096});
  store.insert(smallest);
  if (largest_small != smallest)
    store.insert(largest_small);
  store.insert(remainder);

  // Find exact match for smallest.
  ASSERT_EQ(store.remove_best_fit(smallest->inner_size()), smallest);
  store.insert(smallest);

  // Find exact match for largest.
  ASSERT_EQ(store.remove_best_fit(largest_small->inner_size()), largest_small);
  store.insert(largest_small);

  // Search small list for best fit.
  Block *next_smallest = largest_small == smallest ? remainder : largest_small;
  ASSERT_EQ(store.remove_best_fit(smallest->inner_size() + 1), next_smallest);
  store.insert(next_smallest);

  // Continue search for best fit to large blocks.
  EXPECT_EQ(store.remove_best_fit(largest_small->inner_size() + 1), remainder);
}

TEST(LlvmLibcFreeStore, Remove) {
  byte mem[1024];
  optional<Block *> maybeBlock = Block::init(mem);
  ASSERT_TRUE(maybeBlock.has_value());

  Block *small = *maybeBlock;
  maybeBlock = small->split(sizeof(FreeList::Node) + sizeof(size_t));
  ASSERT_TRUE(maybeBlock.has_value());

  Block *remainder = *maybeBlock;

  FreeStore store;
  store.set_range({0, 4096});
  store.insert(small);
  store.insert(remainder);

  store.remove(remainder);
  ASSERT_EQ(store.remove_best_fit(remainder->inner_size()),
            static_cast<Block *>(nullptr));
  store.remove(small);
  ASSERT_EQ(store.remove_best_fit(small->inner_size()),
            static_cast<Block *>(nullptr));
}

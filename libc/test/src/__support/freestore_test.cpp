//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for a freestore.
///
//===----------------------------------------------------------------------===//

#include <stddef.h>

#include "src/__support/freestore.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::BlockRef;
using LIBC_NAMESPACE::FreeList;
using LIBC_NAMESPACE::FreeStore;
using LIBC_NAMESPACE::FreeTrie;
using LIBC_NAMESPACE::cpp::byte;
using LIBC_NAMESPACE::cpp::optional;

// Inserting or removing blocks too small to be tracked does nothing.
TEST(LlvmLibcFreeStore, TooSmall) {
  byte mem[1024];
  optional<BlockRef> maybeBlock = BlockRef::init(mem);
  ASSERT_TRUE(maybeBlock.has_value());
  BlockRef too_small = *maybeBlock;
  maybeBlock = too_small.split(BlockRef::PREV_FIELD_SIZE);
  ASSERT_TRUE(maybeBlock.has_value());
  // On platforms with high alignment the smallest legal block may be large
  // enough for a node.
  if (too_small.outer_size() >= BlockRef::HEADER_SIZE + sizeof(FreeList::Node))
    return;
  BlockRef remainder = *maybeBlock;

  FreeStore store;
  store.set_range({0, 4096});
  store.insert(too_small);
  store.insert(remainder);

  EXPECT_EQ(store.remove_best_fit(too_small.inner_size()).addr(),
            remainder.addr());
  store.remove(too_small);
}

TEST(LlvmLibcFreeStore, RemoveFit) {
  byte mem[1024];
  optional<BlockRef> maybeBlock = BlockRef::init(mem);
  ASSERT_TRUE(maybeBlock.has_value());

  BlockRef smallest = *maybeBlock;
  maybeBlock =
      smallest.split(sizeof(FreeList::Node) + BlockRef::PREV_FIELD_SIZE);
  ASSERT_TRUE(maybeBlock.has_value());

  BlockRef largest_small = *maybeBlock;
  maybeBlock = largest_small.split(
      sizeof(FreeTrie::Node) + BlockRef::PREV_FIELD_SIZE - BlockRef::MIN_ALIGN);
  ASSERT_TRUE(maybeBlock.has_value());
  if (largest_small.inner_size() == smallest.inner_size())
    largest_small = smallest;
  ASSERT_GE(largest_small.inner_size(), smallest.inner_size());

  BlockRef remainder = *maybeBlock;

  FreeStore store;
  store.set_range({0, 4096});
  store.insert(smallest);
  if (largest_small != smallest)
    store.insert(largest_small);
  store.insert(remainder);

  // Requesting smallest size returns a valid block fitting the size.
  BlockRef block1 = store.remove_best_fit(smallest.inner_size());
  ASSERT_NE(block1.addr(), BlockRef().addr());
  ASSERT_GE(block1.inner_size(), smallest.inner_size());
  store.insert(block1);

  // Requesting largest_small size returns a valid block fitting the size.
  BlockRef block2 = store.remove_best_fit(largest_small.inner_size());
  ASSERT_NE(block2.addr(), BlockRef().addr());
  ASSERT_GE(block2.inner_size(), largest_small.inner_size());
  store.insert(block2);

  // Requesting smallest inner_size + 1 returns a valid block fitting the size.
  BlockRef block3 = store.remove_best_fit(smallest.inner_size() + 1);
  ASSERT_NE(block3.addr(), BlockRef().addr());
  ASSERT_GE(block3.inner_size(), smallest.inner_size() + 1);
  store.insert(block3);

  // Requesting largest_small inner_size + 1 returns a valid block fitting the
  // size.
  BlockRef block4 = store.remove_best_fit(largest_small.inner_size() + 1);
  ASSERT_NE(block4.addr(), BlockRef().addr());
  ASSERT_GE(block4.inner_size(), largest_small.inner_size() + 1);
}

TEST(LlvmLibcFreeStore, Remove) {
  byte mem[1024];
  optional<BlockRef> maybeBlock = BlockRef::init(mem);
  ASSERT_TRUE(maybeBlock.has_value());

  BlockRef small = *maybeBlock;
  maybeBlock = small.split(sizeof(FreeList::Node) + BlockRef::PREV_FIELD_SIZE);
  ASSERT_TRUE(maybeBlock.has_value());

  BlockRef remainder = *maybeBlock;

  FreeStore store;
  store.set_range({0, 4096});
  store.insert(small);
  store.insert(remainder);

  store.remove(remainder);
  ASSERT_EQ(store.remove_best_fit(remainder.inner_size()).addr(),
            BlockRef().addr());
  store.remove(small);
  ASSERT_EQ(store.remove_best_fit(small.inner_size()).addr(),
            BlockRef().addr());
}

TEST(LlvmLibcFreeStore, IndexToMinSize) {
  constexpr size_t min_size_0 = FreeStore::index_to_min_size(0);
  EXPECT_EQ(min_size_0, static_cast<size_t>(0));

  constexpr size_t min_size_1 = FreeStore::index_to_min_size(1);
  EXPECT_EQ(min_size_1, static_cast<size_t>(FreeStore::MIN_INNER_SIZE + 1));

  size_t prev_size = 0;
  for (size_t i = 1; i < 64; ++i) {
    size_t min_size = FreeStore::index_to_min_size(i);
    EXPECT_GT(min_size, prev_size);
    prev_size = min_size;
  }
}

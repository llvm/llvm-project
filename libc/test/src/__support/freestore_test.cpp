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
  store.insert(too_small);
  store.insert(remainder);

  EXPECT_EQ(store.remove_best_fit(too_small.inner_size()).addr(),
            remainder.addr());
  store.remove(too_small);
}

TEST(LlvmLibcFreeStore, RemoveBestFit) {
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
  store.insert(smallest);
  if (largest_small != smallest)
    store.insert(largest_small);
  store.insert(remainder);

  // For TLSF (oversized first), asking for a size will return the block from
  // the first non-empty oversized bin if one exists, bypassing the exact bin.
  if (largest_small != smallest) {
    BlockRef block = store.remove_best_fit(smallest.inner_size());
    ASSERT_EQ(block.addr(), largest_small.addr());
    store.insert(block);

    BlockRef block2 = store.remove_best_fit(largest_small.inner_size());
    ASSERT_EQ(block2.addr(), remainder.addr());
    store.insert(block2);
  } else {
    BlockRef block = store.remove_best_fit(smallest.inner_size());
    ASSERT_EQ(block.addr(), remainder.addr());
    store.insert(block);
  }

  // Search small list for best fit.
  BlockRef next_smallest =
      largest_small == smallest ? remainder : largest_small;
  ASSERT_EQ(store.remove_best_fit(smallest.inner_size() + 1).addr(),
            next_smallest.addr());
  store.insert(next_smallest);

  // Continue search for best fit to large blocks.
  EXPECT_EQ(store.remove_best_fit(largest_small.inner_size() + 1).addr(),
            remainder.addr());
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
  store.insert(small);
  store.insert(remainder);

  store.remove(remainder);
  ASSERT_EQ(store.remove_best_fit(remainder.inner_size()).addr(),
            BlockRef().addr());
  store.remove(small);
  ASSERT_EQ(store.remove_best_fit(small.inner_size()).addr(),
            BlockRef().addr());
}

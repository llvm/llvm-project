//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for a block of memory.
///
//===----------------------------------------------------------------------===//
#include <stddef.h>

#include "src/__support/CPP/array.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/CPP/span.h"
#include "src/__support/block.h"
#include "src/string/memcpy.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::BlockRef;
using LIBC_NAMESPACE::cpp::array;
using LIBC_NAMESPACE::cpp::bit_ceil;
using LIBC_NAMESPACE::cpp::byte;
using LIBC_NAMESPACE::cpp::span;

TEST(LlvmLibcBlockTest, CanCreateSingleAlignedBlock) {
  constexpr size_t kN = 1024;
  alignas(BlockRef::MIN_ALIGN) array<byte, kN> bytes;

  auto result = BlockRef::init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockRef block = *result;

  EXPECT_EQ(block.addr() % alignof(size_t), size_t{0});
  EXPECT_TRUE(block.is_usable_space_aligned(BlockRef::MIN_ALIGN));

  BlockRef last = block.next();
  ASSERT_NE(last.addr(), BlockRef().addr());
  EXPECT_EQ(last.addr() % alignof(size_t), size_t{0});

  EXPECT_EQ(last.outer_size(), BlockRef::HEADER_SIZE);
  EXPECT_EQ(last.prev_free().addr(), block.addr());
  EXPECT_TRUE(last.used());

  size_t block_outer_size = last.addr() - block.addr();
  EXPECT_EQ(block.outer_size(), block_outer_size);
  EXPECT_EQ(block.inner_size(), block_outer_size - BlockRef::HEADER_SIZE +
                                    BlockRef::PREV_FIELD_SIZE);
  EXPECT_EQ(block.prev_free().addr(), BlockRef().addr());
  EXPECT_FALSE(block.used());
}

TEST(LlvmLibcBlockTest, CanCreateUnalignedSingleBlock) {
  constexpr size_t kN = 1024;

  // Force alignment, so we can un-force it below
  alignas(BlockRef::MIN_ALIGN) array<byte, kN> bytes;
  span<byte> aligned(bytes);

  auto result = BlockRef::init(aligned.subspan(1));
  EXPECT_TRUE(result.has_value());

  BlockRef block = *result;
  EXPECT_EQ(block.addr() % alignof(size_t), size_t{0});
  EXPECT_TRUE(block.is_usable_space_aligned(BlockRef::MIN_ALIGN));

  BlockRef last = block.next();
  ASSERT_NE(last.addr(), BlockRef().addr());
  EXPECT_EQ(last.addr() % alignof(size_t), size_t{0});
}

TEST(LlvmLibcBlockTest, CannotCreateTooSmallBlock) {
  array<byte, 2> bytes;
  auto result = BlockRef::init(bytes);
  EXPECT_FALSE(result.has_value());
}

TEST(LlvmLibcBlockTest, CanSplitBlock) {
  constexpr size_t kN = 1024;

  // Choose a split position such that the next block's usable space is 512
  // bytes from this one's. This should be sufficient for any machine's
  // alignment.
  const size_t kSplitN = BlockRef::inner_size(512);

  array<byte, kN> bytes;
  auto result = BlockRef::init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockRef block1 = *result;
  size_t orig_size = block1.outer_size();

  result = block1.split(kSplitN);
  ASSERT_TRUE(result.has_value());
  BlockRef block2 = *result;

  EXPECT_EQ(block1.inner_size(), kSplitN);
  EXPECT_EQ(block1.outer_size(),
            kSplitN - BlockRef::PREV_FIELD_SIZE + BlockRef::HEADER_SIZE);

  EXPECT_EQ(block2.outer_size(), orig_size - block1.outer_size());
  EXPECT_FALSE(block2.used());
  EXPECT_EQ(block2.addr() % alignof(size_t), size_t{0});
  EXPECT_TRUE(block2.is_usable_space_aligned(BlockRef::MIN_ALIGN));

  EXPECT_EQ(block1.next().addr(), block2.addr());
  EXPECT_EQ(block2.prev_free().addr(), block1.addr());
}

TEST(LlvmLibcBlockTest, CanSplitBlockUnaligned) {
  constexpr size_t kN = 1024;

  array<byte, kN> bytes;
  auto result = BlockRef::init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockRef block1 = *result;
  size_t orig_size = block1.outer_size();

  constexpr size_t kSplitN = 513;

  result = block1.split(kSplitN);
  ASSERT_TRUE(result.has_value());
  BlockRef block2 = *result;

  EXPECT_GE(block1.inner_size(), kSplitN);

  EXPECT_EQ(block2.outer_size(), orig_size - block1.outer_size());
  EXPECT_FALSE(block2.used());
  EXPECT_EQ(block2.addr() % alignof(size_t), size_t{0});
  EXPECT_TRUE(block2.is_usable_space_aligned(BlockRef::MIN_ALIGN));

  EXPECT_EQ(block1.next().addr(), block2.addr());
  EXPECT_EQ(block2.prev_free().addr(), block1.addr());
}

TEST(LlvmLibcBlockTest, CanSplitMidBlock) {
  // split once, then split the original block again to ensure that the
  // pointers get rewired properly.
  // I.e.
  // [[             BLOCK 1            ]]
  // block1.split()
  // [[       BLOCK1       ]][[ BLOCK2 ]]
  // block1.split()
  // [[ BLOCK1 ]][[ BLOCK3 ]][[ BLOCK2 ]]

  constexpr size_t kN = 1024;
  constexpr size_t kSplit1 = 512;
  constexpr size_t kSplit2 = 256;

  array<byte, kN> bytes;
  auto result = BlockRef::init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockRef block1 = *result;

  result = block1.split(kSplit1);
  ASSERT_TRUE(result.has_value());
  BlockRef block2 = *result;

  result = block1.split(kSplit2);
  ASSERT_TRUE(result.has_value());
  BlockRef block3 = *result;

  EXPECT_EQ(block1.next().addr(), block3.addr());
  EXPECT_EQ(block3.prev_free().addr(), block1.addr());
  EXPECT_EQ(block3.next().addr(), block2.addr());
  EXPECT_EQ(block2.prev_free().addr(), block3.addr());
}

TEST(LlvmLibcBlockTest, CannotSplitTooSmallBlock) {
  constexpr size_t kN = 64;

  array<byte, kN> bytes;
  auto result = BlockRef::init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockRef block = *result;

  result = block.split(block.inner_size() + 1);
  ASSERT_FALSE(result.has_value());
}

TEST(LlvmLibcBlockTest, CannotSplitBlockWithoutHeaderSpace) {
  constexpr size_t kN = 1024;

  array<byte, kN> bytes;
  auto result = BlockRef::init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockRef block = *result;

  result = block.split(block.inner_size() - BlockRef::HEADER_SIZE + 1);
  ASSERT_FALSE(result.has_value());
}

TEST(LlvmLibcBlockTest, CannotMakeBlockLargerInSplit) {
  // Ensure that we can't ask for more space than the block actually has...
  constexpr size_t kN = 1024;

  array<byte, kN> bytes;
  auto result = BlockRef::init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockRef block = *result;

  result = block.split(block.inner_size() + 1);
  ASSERT_FALSE(result.has_value());
}

TEST(LlvmLibcBlockTest, CanMakeMinimalSizeFirstBlock) {
  // This block does support splitting with minimal payload size.
  constexpr size_t kN = 1024;

  array<byte, kN> bytes;
  auto result = BlockRef::init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockRef block = *result;

  result = block.split(0);
  ASSERT_TRUE(result.has_value());
  EXPECT_LE(block.outer_size(), BlockRef::HEADER_SIZE + BlockRef::MIN_ALIGN);
}

TEST(LlvmLibcBlockTest, CanMakeMinimalSizeSecondBlock) {
  // Likewise, the split block can be minimal-width.
  constexpr size_t kN = 1024;

  array<byte, kN> bytes;
  auto result = BlockRef::init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockRef block1 = *result;

  result =
      block1.split(BlockRef::prev_possible_block_start(block1.next().addr()) -
                   reinterpret_cast<uintptr_t>(block1.usable_space()) +
                   BlockRef::PREV_FIELD_SIZE);
  ASSERT_TRUE(result.has_value());
  EXPECT_LE((*result).outer_size(),
            BlockRef::HEADER_SIZE + BlockRef::MIN_ALIGN);
}

TEST(LlvmLibcBlockTest, CanMarkBlockUsed) {
  constexpr size_t kN = 1024;

  array<byte, kN> bytes;
  auto result = BlockRef::init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockRef block = *result;
  size_t orig_size = block.outer_size();

  block.mark_used();
  EXPECT_TRUE(block.used());
  EXPECT_EQ(block.outer_size(), orig_size);

  block.mark_free();
  EXPECT_FALSE(block.used());
}

TEST(LlvmLibcBlockTest, CanSplitUsedBlock) {
  constexpr size_t kN = 1024;
  constexpr size_t kSplitN = 512;

  array<byte, kN> bytes;
  auto result = BlockRef::init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockRef block = *result;

  block.mark_used();
  result = block.split(kSplitN);
  ASSERT_TRUE(result.has_value());
  BlockRef new_block = *result;

  EXPECT_TRUE(block.used());
  EXPECT_FALSE(new_block.used());
  EXPECT_EQ(block.next().addr(), new_block.addr());
  EXPECT_EQ(new_block.prev_free().addr(), BlockRef().addr());
}

TEST(LlvmLibcBlockTest, CanMergeWithNextBlock) {
  // Do the three way merge from "CanSplitMidBlock", and let's
  // merge block 3 and 2
  constexpr size_t kN = 1024;
  constexpr size_t kSplit1 = 512;
  constexpr size_t kSplit2 = 256;
  array<byte, kN> bytes;
  auto result = BlockRef::init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockRef block1 = *result;
  size_t total_size = block1.outer_size();

  result = block1.split(kSplit1);
  ASSERT_TRUE(result.has_value());

  result = block1.split(kSplit2);
  size_t block1_size = block1.outer_size();
  ASSERT_TRUE(result.has_value());
  BlockRef block3 = *result;

  EXPECT_TRUE(block3.merge_next());

  EXPECT_EQ(block1.next().addr(), block3.addr());
  EXPECT_EQ(block3.prev_free().addr(), block1.addr());
  EXPECT_EQ(block1.outer_size(), block1_size);
  EXPECT_EQ(block3.outer_size(), total_size - block1.outer_size());
}

TEST(LlvmLibcBlockTest, CannotMergeWithFirstOrLastBlock) {
  constexpr size_t kN = 1024;
  constexpr size_t kSplitN = 512;

  array<byte, kN> bytes;
  auto result = BlockRef::init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockRef block1 = *result;

  // Do a split, just to check that the checks on next/prev are different...
  result = block1.split(kSplitN);
  ASSERT_TRUE(result.has_value());
  BlockRef block2 = *result;

  EXPECT_FALSE(block2.merge_next());
}

TEST(LlvmLibcBlockTest, CannotMergeUsedBlock) {
  constexpr size_t kN = 1024;
  constexpr size_t kSplitN = 512;

  array<byte, kN> bytes;
  auto result = BlockRef::init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockRef block = *result;

  // Do a split, just to check that the checks on next/prev are different...
  result = block.split(kSplitN);
  ASSERT_TRUE(result.has_value());

  block.mark_used();
  EXPECT_FALSE(block.merge_next());
}

TEST(LlvmLibcBlockTest, CanGetBlockFromUsableSpace) {
  array<byte, 1024> bytes;
  auto result = BlockRef::init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockRef block1 = *result;

  void *ptr = block1.usable_space();
  BlockRef block2 = BlockRef::from_usable_space(ptr);
  EXPECT_EQ(block1.addr(), block2.addr());
}

TEST(LlvmLibcBlockTest, CanGetConstBlockFromUsableSpace) {
  constexpr size_t kN = 1024;

  array<byte, kN> bytes{};
  auto result = BlockRef::init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockRef block1 = *result;

  const void *ptr = block1.usable_space();
  BlockRef block2 = BlockRef::from_usable_space(ptr);
  EXPECT_EQ(block1.addr(), block2.addr());
}

TEST(LlvmLibcBlockTest, Allocate) {
  constexpr size_t kN = 1024;

  // Ensure we can allocate everything up to the block size within this block.
  for (size_t i = 0; i < kN; ++i) {
    array<byte, kN> bytes;
    auto result = BlockRef::init(bytes);
    ASSERT_TRUE(result.has_value());
    BlockRef block = *result;

    if (i > block.inner_size())
      continue;

    auto info = BlockRef::allocate(block, BlockRef::MIN_ALIGN, i);
    EXPECT_NE(info.block.addr(), BlockRef().addr());
  }

  // Ensure we can allocate a byte at every guaranteeable alignment.
  for (size_t i = 1; i < kN / BlockRef::MIN_ALIGN; ++i) {
    array<byte, kN> bytes;
    auto result = BlockRef::init(bytes);
    ASSERT_TRUE(result.has_value());
    BlockRef block = *result;

    size_t alignment = i * BlockRef::MIN_ALIGN;
    if (BlockRef::min_size_for_allocation(alignment, 1) > block.inner_size())
      continue;

    auto info = BlockRef::allocate(block, alignment, 1);
    EXPECT_NE(info.block.addr(), BlockRef().addr());
  }
}

TEST(LlvmLibcBlockTest, AllocateAlreadyAligned) {
  constexpr size_t kN = 1024;

  array<byte, kN> bytes;
  auto result = BlockRef::init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockRef block = *result;
  uintptr_t orig_end = block.addr() + block.outer_size();

  constexpr size_t SIZE = BlockRef::PREV_FIELD_SIZE + 1;

  auto [aligned_block, prev, next] =
      BlockRef::allocate(block, BlockRef::MIN_ALIGN, SIZE);

  // Since this is already aligned, there should be no previous block.
  EXPECT_EQ(prev.addr(), BlockRef().addr());

  // Ensure we the block is aligned and large enough.
  EXPECT_NE(aligned_block.addr(), BlockRef().addr());
  EXPECT_TRUE(aligned_block.is_usable_space_aligned(BlockRef::MIN_ALIGN));
  EXPECT_GE(aligned_block.inner_size(), SIZE);

  // Check the next block.
  EXPECT_NE(next.addr(), BlockRef().addr());
  EXPECT_EQ(aligned_block.next().addr(), next.addr());
  EXPECT_EQ(next.addr() + next.outer_size(), orig_end);
}

TEST(LlvmLibcBlockTest, AllocateNeedsAlignment) {
  constexpr size_t kN = 1024;

  array<byte, kN> bytes;
  auto result = BlockRef::init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockRef block = *result;

  uintptr_t orig_end = block.addr() + block.outer_size();

  // Now pick an alignment such that the usable space is not already aligned to
  // it. We want to explicitly test that the block will split into one before
  // it.
  size_t alignment = BlockRef::MIN_ALIGN;
  while (block.is_usable_space_aligned(alignment))
    alignment += BlockRef::MIN_ALIGN;

  auto [aligned_block, prev, next] = BlockRef::allocate(block, alignment, 10);

  // Check the previous block was created appropriately. Since this block is the
  // first block, a new one should be made before this.
  EXPECT_NE(prev.addr(), BlockRef().addr());
  EXPECT_EQ(aligned_block.prev_free().addr(), prev.addr());
  EXPECT_EQ(prev.next().addr(), aligned_block.addr());
  EXPECT_EQ(prev.outer_size(), aligned_block.addr() - prev.addr());

  // Ensure we the block is aligned and the size we expect.
  EXPECT_NE(next.addr(), BlockRef().addr());
  EXPECT_TRUE(aligned_block.is_usable_space_aligned(alignment));

  // Check the next block.
  EXPECT_NE(next.addr(), BlockRef().addr());
  EXPECT_EQ(aligned_block.next().addr(), next.addr());
  EXPECT_EQ(next.addr() + next.outer_size(), orig_end);
}

TEST(LlvmLibcBlockTest, PreviousBlockMergedIfNotFirst) {
  constexpr size_t kN = 1024;

  array<byte, kN> bytes;
  auto result = BlockRef::init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockRef block = *result;

  // Split the block roughly halfway and work on the second half.
  auto result2 = block.split(kN / 2);
  ASSERT_TRUE(result2.has_value());
  BlockRef newblock = *result2;
  ASSERT_EQ(newblock.prev_free().addr(), block.addr());
  size_t old_prev_size = block.outer_size();

  // Now pick an alignment such that the usable space is not already aligned to
  // it. We want to explicitly test that the block will split into one before
  // it.
  size_t alignment = BlockRef::MIN_ALIGN;
  while (newblock.is_usable_space_aligned(alignment))
    alignment += BlockRef::MIN_ALIGN;

  // Ensure we can allocate in the new block.
  auto [aligned_block, prev, next] = BlockRef::allocate(newblock, alignment, 1);

  // Now there should be no new previous block. Instead, the padding we did
  // create should be merged into the original previous block.
  EXPECT_EQ(prev.addr(), BlockRef().addr());
  EXPECT_EQ(aligned_block.prev_free().addr(), block.addr());
  EXPECT_EQ(block.next().addr(), aligned_block.addr());
  EXPECT_GT(block.outer_size(), old_prev_size);
}

TEST(LlvmLibcBlockTest, CanRemergeBlockAllocations) {
  // Finally to ensure we made the split blocks correctly via allocate. We
  // should be able to reconstruct the original block from the blocklets.
  //
  // This is the same setup as with the `AllocateNeedsAlignment` test case.
  constexpr size_t kN = 1024;

  array<byte, kN> bytes;
  auto result = BlockRef::init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockRef block = *result;

  BlockRef orig_block = block;
  size_t orig_size = orig_block.outer_size();

  BlockRef last = block.next();

  ASSERT_EQ(block.prev_free().addr(), BlockRef().addr());

  // Now pick an alignment such that the usable space is not already aligned to
  // it. We want to explicitly test that the block will split into one before
  // it.
  size_t alignment = BlockRef::MIN_ALIGN;
  while (block.is_usable_space_aligned(alignment))
    alignment += BlockRef::MIN_ALIGN;

  auto [aligned_block, prev, next] = BlockRef::allocate(block, alignment, 1);

  // Check we have the appropriate blocks.
  ASSERT_NE(prev.addr(), BlockRef().addr());
  ASSERT_EQ(aligned_block.prev_free().addr(), prev.addr());
  EXPECT_NE(next.addr(), BlockRef().addr());
  EXPECT_EQ(aligned_block.next().addr(), next.addr());
  EXPECT_EQ(next.next().addr(), last.addr());

  // Now check for successful merges.
  EXPECT_TRUE(prev.merge_next());
  EXPECT_EQ(prev.next().addr(), next.addr());
  EXPECT_TRUE(prev.merge_next());
  EXPECT_EQ(prev.next().addr(), last.addr());

  // We should have the original buffer.
  EXPECT_EQ(prev.addr(), orig_block.addr());
  EXPECT_EQ(prev.outer_size(), orig_size);
}

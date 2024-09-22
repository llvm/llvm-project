//===-- Unittests for a block of memory -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <stddef.h>

#include "src/__support/CPP/array.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/CPP/span.h"
#include "src/__support/block.h"
#include "src/string/memcpy.h"
#include "test/UnitTest/Test.h"

// Block types.
using LargeOffsetBlock = LIBC_NAMESPACE::Block<uint64_t>;
using SmallOffsetBlock = LIBC_NAMESPACE::Block<uint16_t>;

// For each of the block types above, we'd like to run the same tests since
// they should work independently of the parameter sizes. Rather than re-writing
// the same test for each case, let's instead create a custom test framework for
// each test case that invokes the actual testing function for each block type.
//
// It's organized this way because the ASSERT/EXPECT macros only work within a
// `Test` class due to those macros expanding to `test` methods.
#define TEST_FOR_EACH_BLOCK_TYPE(TestCase)                                     \
  class LlvmLibcBlockTest##TestCase : public LIBC_NAMESPACE::testing::Test {   \
  public:                                                                      \
    template <typename BlockType> void RunTest();                              \
  };                                                                           \
  TEST_F(LlvmLibcBlockTest##TestCase, TestCase) {                              \
    RunTest<LargeOffsetBlock>();                                               \
    RunTest<SmallOffsetBlock>();                                               \
  }                                                                            \
  template <typename BlockType> void LlvmLibcBlockTest##TestCase::RunTest()

using LIBC_NAMESPACE::cpp::array;
using LIBC_NAMESPACE::cpp::bit_ceil;
using LIBC_NAMESPACE::cpp::byte;
using LIBC_NAMESPACE::cpp::span;

TEST_FOR_EACH_BLOCK_TYPE(CanCreateSingleAlignedBlock) {
  constexpr size_t kN = 1024;
  alignas(BlockType::ALIGNMENT) array<byte, kN> bytes;

  auto result = BlockType::init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockType *block = *result;

  BlockType *last = block->next();
  ASSERT_NE(last, static_cast<BlockType *>(nullptr));
  constexpr size_t last_outer_size = BlockType::BLOCK_OVERHEAD;
  EXPECT_EQ(last->outer_size(), last_outer_size);
  EXPECT_EQ(last->prev_free(), block);
  EXPECT_TRUE(last->used());

  EXPECT_EQ(block->outer_size(), kN - last_outer_size);
  constexpr size_t last_prev_field_size =
      sizeof(typename BlockType::offset_type);
  EXPECT_EQ(block->inner_size(), kN - last_outer_size -
                                     BlockType::BLOCK_OVERHEAD +
                                     last_prev_field_size);
  EXPECT_EQ(block->prev_free(), static_cast<BlockType *>(nullptr));
  EXPECT_FALSE(block->used());
}

TEST_FOR_EACH_BLOCK_TYPE(CanCreateUnalignedSingleBlock) {
  constexpr size_t kN = 1024;

  // Force alignment, so we can un-force it below
  alignas(BlockType::ALIGNMENT) array<byte, kN> bytes;
  span<byte> aligned(bytes);

  auto result = BlockType::init(aligned.subspan(1));
  EXPECT_TRUE(result.has_value());
}

TEST_FOR_EACH_BLOCK_TYPE(CannotCreateTooSmallBlock) {
  array<byte, 2> bytes;
  auto result = BlockType::init(bytes);
  EXPECT_FALSE(result.has_value());
}

// This test specifically checks that we cannot allocate a block with a size
// larger than what can be held by the offset type, we don't need to test with
// multiple block types for this particular check, so we use the normal TEST
// macro and not the custom framework.
TEST(LlvmLibcBlockTest, CannotCreateTooLargeBlock) {
  using BlockType = LIBC_NAMESPACE::Block<uint8_t>;
  constexpr size_t kN = 1024;

  alignas(BlockType::ALIGNMENT) array<byte, kN> bytes;
  auto result = BlockType::init(bytes);
  EXPECT_FALSE(result.has_value());
}

TEST_FOR_EACH_BLOCK_TYPE(CanSplitBlock) {
  constexpr size_t kN = 1024;
  constexpr size_t prev_field_size = sizeof(typename BlockType::offset_type);
  // Give the split position a large alignment.
  constexpr size_t kSplitN = 512 + prev_field_size;

  alignas(BlockType::ALIGNMENT) array<byte, kN> bytes;
  auto result = BlockType::init(bytes);
  ASSERT_TRUE(result.has_value());
  auto *block1 = *result;
  size_t orig_size = block1->outer_size();

  result = block1->split(kSplitN);
  ASSERT_TRUE(result.has_value());
  auto *block2 = *result;

  EXPECT_EQ(block1->inner_size(), kSplitN);
  EXPECT_EQ(block1->outer_size(),
            kSplitN - prev_field_size + BlockType::BLOCK_OVERHEAD);

  EXPECT_EQ(block2->outer_size(), orig_size - block1->outer_size());
  EXPECT_FALSE(block2->used());

  EXPECT_EQ(block1->next(), block2);
  EXPECT_EQ(block2->prev_free(), block1);
}

TEST_FOR_EACH_BLOCK_TYPE(CanSplitBlockUnaligned) {
  constexpr size_t kN = 1024;

  alignas(BlockType::ALIGNMENT) array<byte, kN> bytes;
  auto result = BlockType::init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockType *block1 = *result;
  size_t orig_size = block1->outer_size();

  constexpr size_t kSplitN = 513;
  constexpr size_t prev_field_size = sizeof(typename BlockType::offset_type);
  uintptr_t split_addr =
      reinterpret_cast<uintptr_t>(block1) + (kSplitN - prev_field_size);
  // Round split_addr up to a multiple of the alignment.
  split_addr += alignof(BlockType) - (split_addr % alignof(BlockType));
  uintptr_t split_len = split_addr - (uintptr_t)&bytes + prev_field_size;

  result = block1->split(kSplitN);
  ASSERT_TRUE(result.has_value());
  BlockType *block2 = *result;

  EXPECT_EQ(block1->inner_size(), split_len);

  EXPECT_EQ(block2->outer_size(), orig_size - block1->outer_size());
  EXPECT_FALSE(block2->used());

  EXPECT_EQ(block1->next(), block2);
  EXPECT_EQ(block2->prev_free(), block1);
}

TEST_FOR_EACH_BLOCK_TYPE(CanSplitMidBlock) {
  // split once, then split the original block again to ensure that the
  // pointers get rewired properly.
  // I.e.
  // [[             BLOCK 1            ]]
  // block1->split()
  // [[       BLOCK1       ]][[ BLOCK2 ]]
  // block1->split()
  // [[ BLOCK1 ]][[ BLOCK3 ]][[ BLOCK2 ]]

  constexpr size_t kN = 1024;
  constexpr size_t kSplit1 = 512;
  constexpr size_t kSplit2 = 256;

  alignas(BlockType::ALIGNMENT) array<byte, kN> bytes;
  auto result = BlockType::init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockType *block1 = *result;

  result = block1->split(kSplit1);
  ASSERT_TRUE(result.has_value());
  BlockType *block2 = *result;

  result = block1->split(kSplit2);
  ASSERT_TRUE(result.has_value());
  BlockType *block3 = *result;

  EXPECT_EQ(block1->next(), block3);
  EXPECT_EQ(block3->prev_free(), block1);
  EXPECT_EQ(block3->next(), block2);
  EXPECT_EQ(block2->prev_free(), block3);
}

TEST_FOR_EACH_BLOCK_TYPE(CannotSplitTooSmallBlock) {
  constexpr size_t kN = 64;
  constexpr size_t kSplitN = kN + 1;

  alignas(BlockType::ALIGNMENT) array<byte, kN> bytes;
  auto result = BlockType::init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockType *block = *result;

  result = block->split(kSplitN);
  ASSERT_FALSE(result.has_value());
}

TEST_FOR_EACH_BLOCK_TYPE(CannotSplitBlockWithoutHeaderSpace) {
  constexpr size_t kN = 1024;
  constexpr size_t kSplitN = kN - 2 * BlockType::BLOCK_OVERHEAD - 1;

  alignas(BlockType::ALIGNMENT) array<byte, kN> bytes;
  auto result = BlockType::init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockType *block = *result;

  result = block->split(kSplitN);
  ASSERT_FALSE(result.has_value());
}

TEST_FOR_EACH_BLOCK_TYPE(CannotMakeBlockLargerInSplit) {
  // Ensure that we can't ask for more space than the block actually has...
  constexpr size_t kN = 1024;

  alignas(BlockType::ALIGNMENT) array<byte, kN> bytes;
  auto result = BlockType::init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockType *block = *result;

  result = block->split(block->inner_size() + 1);
  ASSERT_FALSE(result.has_value());
}

TEST_FOR_EACH_BLOCK_TYPE(CannotMakeSecondBlockLargerInSplit) {
  // Ensure that the second block in split is at least of the size of header.
  constexpr size_t kN = 1024;

  alignas(BlockType::ALIGNMENT) array<byte, kN> bytes;
  auto result = BlockType::init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockType *block = *result;

  result = block->split(block->inner_size() - BlockType::BLOCK_OVERHEAD + 1);
  ASSERT_FALSE(result.has_value());
}

TEST_FOR_EACH_BLOCK_TYPE(CannotMakeZeroSizeFirstBlock) {
  // This block doesn't support splitting with zero payload size, since the
  // prev_ field of the next block is always available.
  constexpr size_t kN = 1024;

  alignas(BlockType::ALIGNMENT) array<byte, kN> bytes;
  auto result = BlockType::init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockType *block = *result;

  result = block->split(0);
  EXPECT_FALSE(result.has_value());
}

TEST_FOR_EACH_BLOCK_TYPE(CanMakeMinimalSizeFirstBlock) {
  // This block does support splitting with minimal payload size.
  constexpr size_t kN = 1024;
  constexpr size_t minimal_size = sizeof(typename BlockType::offset_type);

  alignas(BlockType::ALIGNMENT) array<byte, kN> bytes;
  auto result = BlockType::init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockType *block = *result;

  result = block->split(minimal_size);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(block->inner_size(), minimal_size);
}

TEST_FOR_EACH_BLOCK_TYPE(CanMakeMinimalSizeSecondBlock) {
  // Likewise, the split block can be minimal-width.
  constexpr size_t kN = 1024;
  constexpr size_t minimal_size = sizeof(typename BlockType::offset_type);

  alignas(BlockType::ALIGNMENT) array<byte, kN> bytes;
  auto result = BlockType::init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockType *block1 = *result;

  result = block1->split(block1->inner_size() - BlockType::BLOCK_OVERHEAD);
  ASSERT_TRUE(result.has_value());
  BlockType *block2 = *result;

  EXPECT_EQ(block2->inner_size(), minimal_size);
}

TEST_FOR_EACH_BLOCK_TYPE(CanMarkBlockUsed) {
  constexpr size_t kN = 1024;

  alignas(BlockType::ALIGNMENT) array<byte, kN> bytes;
  auto result = BlockType::init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockType *block = *result;
  size_t orig_size = block->outer_size();

  block->mark_used();
  EXPECT_TRUE(block->used());
  EXPECT_EQ(block->outer_size(), orig_size);

  block->mark_free();
  EXPECT_FALSE(block->used());
}

TEST_FOR_EACH_BLOCK_TYPE(CannotSplitUsedBlock) {
  constexpr size_t kN = 1024;
  constexpr size_t kSplitN = 512;

  alignas(BlockType::ALIGNMENT) array<byte, kN> bytes;
  auto result = BlockType::init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockType *block = *result;

  block->mark_used();
  result = block->split(kSplitN);
  ASSERT_FALSE(result.has_value());
}

TEST_FOR_EACH_BLOCK_TYPE(CanMergeWithNextBlock) {
  // Do the three way merge from "CanSplitMidBlock", and let's
  // merge block 3 and 2
  constexpr size_t kN = 1024;
  // Give the split positions large alignments.
  constexpr size_t prev_field_size = sizeof(typename BlockType::offset_type);
  constexpr size_t kSplit1 = 512 + prev_field_size;
  constexpr size_t kSplit2 = 256 + prev_field_size;

  alignas(BlockType::ALIGNMENT) array<byte, kN> bytes;
  auto result = BlockType::init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockType *block1 = *result;
  size_t orig_size = block1->outer_size();

  result = block1->split(kSplit1);
  ASSERT_TRUE(result.has_value());

  result = block1->split(kSplit2);
  ASSERT_TRUE(result.has_value());
  BlockType *block3 = *result;

  EXPECT_TRUE(block3->merge_next());

  EXPECT_EQ(block1->next(), block3);
  EXPECT_EQ(block3->prev_free(), block1);
  EXPECT_EQ(block1->inner_size(), kSplit2);
  EXPECT_EQ(block3->outer_size(), orig_size - block1->outer_size());
}

TEST_FOR_EACH_BLOCK_TYPE(CannotMergeWithFirstOrLastBlock) {
  constexpr size_t kN = 1024;
  constexpr size_t kSplitN = 512;

  alignas(BlockType::ALIGNMENT) array<byte, kN> bytes;
  auto result = BlockType::init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockType *block1 = *result;

  // Do a split, just to check that the checks on next/prev are different...
  result = block1->split(kSplitN);
  ASSERT_TRUE(result.has_value());
  BlockType *block2 = *result;

  EXPECT_FALSE(block2->merge_next());
}

TEST_FOR_EACH_BLOCK_TYPE(CannotMergeUsedBlock) {
  constexpr size_t kN = 1024;
  constexpr size_t kSplitN = 512;

  alignas(BlockType::ALIGNMENT) array<byte, kN> bytes;
  auto result = BlockType::init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockType *block = *result;

  // Do a split, just to check that the checks on next/prev are different...
  result = block->split(kSplitN);
  ASSERT_TRUE(result.has_value());

  block->mark_used();
  EXPECT_FALSE(block->merge_next());
}

TEST_FOR_EACH_BLOCK_TYPE(CanGetBlockFromUsableSpace) {
  constexpr size_t kN = 1024;

  array<byte, kN> bytes{};
  auto result = BlockType::init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockType *block1 = *result;

  void *ptr = block1->usable_space();
  BlockType *block2 = BlockType::from_usable_space(ptr);
  EXPECT_EQ(block1, block2);
}

TEST_FOR_EACH_BLOCK_TYPE(CanGetConstBlockFromUsableSpace) {
  constexpr size_t kN = 1024;

  array<byte, kN> bytes{};
  auto result = BlockType::init(bytes);
  ASSERT_TRUE(result.has_value());
  const BlockType *block1 = *result;

  const void *ptr = block1->usable_space();
  const BlockType *block2 = BlockType::from_usable_space(ptr);
  EXPECT_EQ(block1, block2);
}

TEST_FOR_EACH_BLOCK_TYPE(CanAllocate) {
  constexpr size_t kN = 1024 + BlockType::BLOCK_OVERHEAD;

  // Ensure we can allocate everything up to the block size within this block.
  for (size_t i = 0; i < kN - 2 * BlockType::BLOCK_OVERHEAD; ++i) {
    alignas(BlockType::ALIGNMENT) array<byte, kN> bytes{};
    auto result = BlockType::init(bytes);
    ASSERT_TRUE(result.has_value());
    BlockType *block = *result;

    constexpr size_t ALIGN = 1; // Effectively ignores alignment.
    EXPECT_TRUE(block->can_allocate(ALIGN, i));

    // For each can_allocate, we should be able to do a successful call to
    // allocate.
    auto info = BlockType::allocate(block, ALIGN, i);
    EXPECT_NE(info.block, static_cast<BlockType *>(nullptr));
  }

  alignas(BlockType::ALIGNMENT) array<byte, kN> bytes{};
  auto result = BlockType::init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockType *block = *result;

  // Given a block of size N (assuming it's also a power of two), we should be
  // able to allocate a block within it that's aligned to N/2. This is
  // because regardless of where the buffer is located, we can always find a
  // starting location within it that meets this alignment.
  EXPECT_TRUE(block->can_allocate(block->outer_size() / 2, 1));
  auto info = BlockType::allocate(block, block->outer_size() / 2, 1);
  EXPECT_NE(info.block, static_cast<BlockType *>(nullptr));
}

TEST_FOR_EACH_BLOCK_TYPE(AllocateAlreadyAligned) {
  constexpr size_t kN = 1024;

  alignas(BlockType::ALIGNMENT) array<byte, kN> bytes{};
  auto result = BlockType::init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockType *block = *result;

  // This should result in no new blocks.
  constexpr size_t kAlignment = BlockType::ALIGNMENT;
  constexpr size_t prev_field_size = sizeof(typename BlockType::offset_type);
  constexpr size_t kExpectedSize = BlockType::ALIGNMENT + prev_field_size;
  EXPECT_TRUE(block->can_allocate(kAlignment, kExpectedSize));

  auto [aligned_block, prev, next] =
      BlockType::allocate(block, BlockType::ALIGNMENT, kExpectedSize);

  // Since this is already aligned, there should be no previous block.
  EXPECT_EQ(prev, static_cast<BlockType *>(nullptr));

  // Ensure we the block is aligned and the size we expect.
  EXPECT_NE(aligned_block, static_cast<BlockType *>(nullptr));
  EXPECT_TRUE(aligned_block->is_usable_space_aligned(BlockType::ALIGNMENT));
  EXPECT_EQ(aligned_block->inner_size(), kExpectedSize);

  // Check the next block.
  EXPECT_NE(next, static_cast<BlockType *>(nullptr));
  EXPECT_EQ(aligned_block->next(), next);
  EXPECT_EQ(reinterpret_cast<byte *>(next) + next->outer_size(),
            bytes.data() + bytes.size() - BlockType::BLOCK_OVERHEAD);
}

TEST_FOR_EACH_BLOCK_TYPE(AllocateNeedsAlignment) {
  constexpr size_t kN = 1024;

  alignas(kN) array<byte, kN> bytes{};
  auto result = BlockType::init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockType *block = *result;

  // Ensure first the usable_data is only aligned to the block alignment.
  ASSERT_EQ(block->usable_space(), bytes.data() + BlockType::BLOCK_OVERHEAD);
  ASSERT_EQ(block->prev_free(), static_cast<BlockType *>(nullptr));

  // Now pick an alignment such that the usable space is not already aligned to
  // it. We want to explicitly test that the block will split into one before
  // it.
  constexpr size_t kAlignment = bit_ceil(BlockType::BLOCK_OVERHEAD) * 8;
  ASSERT_FALSE(block->is_usable_space_aligned(kAlignment));

  constexpr size_t kSize = 10;
  EXPECT_TRUE(block->can_allocate(kAlignment, kSize));

  auto [aligned_block, prev, next] =
      BlockType::allocate(block, kAlignment, kSize);

  // Check the previous block was created appropriately. Since this block is the
  // first block, a new one should be made before this.
  EXPECT_NE(prev, static_cast<BlockType *>(nullptr));
  EXPECT_EQ(aligned_block->prev_free(), prev);
  EXPECT_EQ(prev->next(), aligned_block);
  EXPECT_EQ(prev->outer_size(), reinterpret_cast<uintptr_t>(aligned_block) -
                                    reinterpret_cast<uintptr_t>(prev));

  // Ensure we the block is aligned and the size we expect.
  EXPECT_NE(next, static_cast<BlockType *>(nullptr));
  EXPECT_TRUE(aligned_block->is_usable_space_aligned(kAlignment));

  // Check the next block.
  EXPECT_NE(next, static_cast<BlockType *>(nullptr));
  EXPECT_EQ(aligned_block->next(), next);
  EXPECT_EQ(reinterpret_cast<byte *>(next) + next->outer_size(),
            bytes.data() + bytes.size() - BlockType::BLOCK_OVERHEAD);
}

TEST_FOR_EACH_BLOCK_TYPE(PreviousBlockMergedIfNotFirst) {
  constexpr size_t kN = 1024;

  alignas(kN) array<byte, kN> bytes{};
  auto result = BlockType::init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockType *block = *result;

  // Split the block roughly halfway and work on the second half.
  auto result2 = block->split(kN / 2);
  ASSERT_TRUE(result2.has_value());
  BlockType *newblock = *result2;
  ASSERT_EQ(newblock->prev_free(), block);
  size_t old_prev_size = block->outer_size();

  // Now pick an alignment such that the usable space is not already aligned to
  // it. We want to explicitly test that the block will split into one before
  // it.
  constexpr size_t kAlignment = bit_ceil(BlockType::BLOCK_OVERHEAD) * 8;
  ASSERT_FALSE(newblock->is_usable_space_aligned(kAlignment));

  // Ensure we can allocate in the new block.
  constexpr size_t kSize = BlockType::ALIGNMENT;
  EXPECT_TRUE(newblock->can_allocate(kAlignment, kSize));

  auto [aligned_block, prev, next] =
      BlockType::allocate(newblock, kAlignment, kSize);

  // Now there should be no new previous block. Instead, the padding we did
  // create should be merged into the original previous block.
  EXPECT_EQ(prev, static_cast<BlockType *>(nullptr));
  EXPECT_EQ(aligned_block->prev_free(), block);
  EXPECT_EQ(block->next(), aligned_block);
  EXPECT_GT(block->outer_size(), old_prev_size);
}

TEST_FOR_EACH_BLOCK_TYPE(CanRemergeBlockAllocations) {
  // Finally to ensure we made the split blocks correctly via allocate. We
  // should be able to reconstruct the original block from the blocklets.
  //
  // This is the same setup as with the `AllocateNeedsAlignment` test case.
  constexpr size_t kN = 1024;

  alignas(kN) array<byte, kN> bytes{};
  auto result = BlockType::init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockType *block = *result;
  BlockType *last = block->next();

  // Ensure first the usable_data is only aligned to the block alignment.
  ASSERT_EQ(block->usable_space(), bytes.data() + BlockType::BLOCK_OVERHEAD);
  ASSERT_EQ(block->prev_free(), static_cast<BlockType *>(nullptr));

  // Now pick an alignment such that the usable space is not already aligned to
  // it. We want to explicitly test that the block will split into one before
  // it.
  constexpr size_t kAlignment = bit_ceil(BlockType::BLOCK_OVERHEAD) * 8;
  ASSERT_FALSE(block->is_usable_space_aligned(kAlignment));

  constexpr size_t kSize = BlockType::ALIGNMENT;
  EXPECT_TRUE(block->can_allocate(kAlignment, kSize));

  auto [aligned_block, prev, next] =
      BlockType::allocate(block, kAlignment, kSize);

  // Check we have the appropriate blocks.
  ASSERT_NE(prev, static_cast<BlockType *>(nullptr));
  ASSERT_EQ(aligned_block->prev_free(), prev);
  EXPECT_NE(next, static_cast<BlockType *>(nullptr));
  EXPECT_EQ(aligned_block->next(), next);
  EXPECT_EQ(next->next(), last);

  // Now check for successful merges.
  EXPECT_TRUE(prev->merge_next());
  EXPECT_EQ(prev->next(), next);
  EXPECT_TRUE(prev->merge_next());
  EXPECT_EQ(prev->next(), last);

  // We should have the original buffer.
  EXPECT_EQ(reinterpret_cast<byte *>(prev), &*bytes.begin());
  EXPECT_EQ(prev->outer_size(), bytes.size() - BlockType::BLOCK_OVERHEAD);
  EXPECT_EQ(reinterpret_cast<byte *>(prev) + prev->outer_size(),
            &*bytes.end() - BlockType::BLOCK_OVERHEAD);
}

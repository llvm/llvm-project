//===-- Unittests for freelist_malloc -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/freelist_heap.h"
#include "src/stdlib/aligned_alloc.h"
#include "src/stdlib/calloc.h"
#include "src/stdlib/free.h"
#include "src/stdlib/malloc.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::freelist_heap;
using LIBC_NAMESPACE::FreeListHeap;
using LIBC_NAMESPACE::FreeListHeapBuffer;

TEST(LlvmLibcFreeListMalloc, Malloc) {
  constexpr size_t kAllocSize = 256;
  constexpr size_t kCallocNum = 4;
  constexpr size_t kCallocSize = 64;

  typedef FreeListHeap<>::BlockType Block;

  void *ptr1 = LIBC_NAMESPACE::malloc(kAllocSize);
  auto *block = Block::from_usable_space(ptr1);
  EXPECT_GE(block->inner_size(), kAllocSize);

  LIBC_NAMESPACE::free(ptr1);
  ASSERT_NE(block->next(), static_cast<Block *>(nullptr));
  ASSERT_EQ(block->next()->next(), static_cast<Block *>(nullptr));
  size_t heap_size = block->inner_size();

  void *ptr2 = LIBC_NAMESPACE::calloc(kCallocNum, kCallocSize);
  ASSERT_EQ(ptr2, ptr1);
  EXPECT_GE(block->inner_size(), kCallocNum * kCallocSize);

  for (size_t i = 0; i < kCallocNum * kCallocSize; ++i)
    EXPECT_EQ(reinterpret_cast<uint8_t *>(ptr2)[i], uint8_t(0));

  LIBC_NAMESPACE::free(ptr2);
  EXPECT_EQ(block->inner_size(), heap_size);

  constexpr size_t ALIGN = kAllocSize;
  void *ptr3 = LIBC_NAMESPACE::aligned_alloc(ALIGN, kAllocSize);
  EXPECT_NE(ptr3, static_cast<void *>(nullptr));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr3) % ALIGN, size_t(0));
  auto *aligned_block = reinterpret_cast<Block *>(ptr3);
  EXPECT_GE(aligned_block->inner_size(), kAllocSize);

  LIBC_NAMESPACE::free(ptr3);
  EXPECT_EQ(block->inner_size(), heap_size);
}

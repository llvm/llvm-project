//===-- Unittests for freelist_malloc -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/freelist_heap.h"
#include "src/stdlib/calloc.h"
#include "src/stdlib/free.h"
#include "src/stdlib/malloc.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::freelist_heap;

TEST(LlvmLibcFreeListMalloc, MallocStats) {
  constexpr size_t kAllocSize = 256;
  constexpr size_t kCallocNum = 4;
  constexpr size_t kCallocSize = 64;

  freelist_heap->reset_heap_stats(); // Do this because other tests might've
                                     // called the same global allocator.

  void *ptr1 = LIBC_NAMESPACE::malloc(kAllocSize);

  const auto &freelist_heap_stats = freelist_heap->heap_stats();

  ASSERT_NE(ptr1, static_cast<void *>(nullptr));
  EXPECT_EQ(freelist_heap_stats.bytes_allocated, kAllocSize);
  EXPECT_EQ(freelist_heap_stats.cumulative_allocated, kAllocSize);
  EXPECT_EQ(freelist_heap_stats.cumulative_freed, size_t(0));

  LIBC_NAMESPACE::free(ptr1);
  EXPECT_EQ(freelist_heap_stats.bytes_allocated, size_t(0));
  EXPECT_EQ(freelist_heap_stats.cumulative_allocated, kAllocSize);
  EXPECT_EQ(freelist_heap_stats.cumulative_freed, kAllocSize);

  void *ptr2 = LIBC_NAMESPACE::calloc(kCallocNum, kCallocSize);
  ASSERT_NE(ptr2, static_cast<void *>(nullptr));
  EXPECT_EQ(freelist_heap_stats.bytes_allocated, kCallocNum * kCallocSize);
  EXPECT_EQ(freelist_heap_stats.cumulative_allocated,
            kAllocSize + kCallocNum * kCallocSize);
  EXPECT_EQ(freelist_heap_stats.cumulative_freed, kAllocSize);

  for (size_t i = 0; i < kCallocNum * kCallocSize; ++i) {
    EXPECT_EQ(reinterpret_cast<uint8_t *>(ptr2)[i], uint8_t(0));
  }

  LIBC_NAMESPACE::free(ptr2);
  EXPECT_EQ(freelist_heap_stats.bytes_allocated, size_t(0));
  EXPECT_EQ(freelist_heap_stats.cumulative_allocated,
            kAllocSize + kCallocNum * kCallocSize);
  EXPECT_EQ(freelist_heap_stats.cumulative_freed,
            kAllocSize + kCallocNum * kCallocSize);
}

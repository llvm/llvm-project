//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for freelist_heap.
///
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/span.h"
#include "src/__support/freelist_heap.h"
#include "src/__support/freetable.h"
#include "src/__support/macros/config.h"
#include "src/string/memcmp.h"
#include "src/string/memcpy.h"
#include "src/string/memory_utils/inline_memset.h"
#include "test/UnitTest/Test.h"

asm(R"(
.globl _end, __llvm_libc_heap_limit
.bss
_end:
  .fill 1024
__llvm_libc_heap_limit:
)");

using LIBC_NAMESPACE::Block;
using LIBC_NAMESPACE::freelist_heap;
using LIBC_NAMESPACE::FreeListHeap;
using LIBC_NAMESPACE::FreeListHeapBuffer;
using LIBC_NAMESPACE::FreeTable;
using LIBC_NAMESPACE::cpp::byte;
using LIBC_NAMESPACE::cpp::span;

// A test FreeTable configuration (192 bits table)
using TestTable = FreeTable<32, 3, 2, 3>;

// Macro that generates a test fixture that runs the given test logic
// against both the FreeStore (Trie) and FreeTable (TLSF) backed heaps!
#define TEST_FOR_EACH_ALLOCATOR(TestCase, BufferSize)                          \
  class LlvmLibcFreeListHeapTest##TestCase                                     \
      : public LIBC_NAMESPACE::testing::Test {                                 \
  public:                                                                      \
    FreeListHeapBuffer<BufferSize, LIBC_NAMESPACE::FreeStore>                  \
        fake_global_buffer_store;                                              \
    FreeListHeapBuffer<BufferSize, TestTable> fake_global_buffer_table;        \
    void RunTestStore(FreeListHeap<LIBC_NAMESPACE::FreeStore> &allocator,      \
                      size_t N);                                               \
    void RunTestTable(FreeListHeap<TestTable> &allocator, size_t N);           \
    template <typename allocator_type>                                         \
    void RunTestImpl(allocator_type &allocator, [[maybe_unused]] size_t N);    \
  };                                                                           \
  TEST_F(LlvmLibcFreeListHeapTest##TestCase, TestCase) {                       \
    {                                                                          \
      byte buf[BufferSize] = {byte(0)};                                        \
      FreeListHeap<LIBC_NAMESPACE::FreeStore> allocator(buf);                  \
      RunTestStore(allocator, BufferSize);                                     \
    }                                                                          \
    {                                                                          \
      byte buf[BufferSize] = {byte(0)};                                        \
      FreeListHeap<TestTable> allocator(buf);                                  \
      RunTestTable(allocator, BufferSize);                                     \
    }                                                                          \
    {                                                                          \
      freelist_heap = new (&fake_global_buffer_store)                          \
          FreeListHeapBuffer<BufferSize, LIBC_NAMESPACE::FreeStore>;           \
      RunTestStore(*freelist_heap, freelist_heap->region().size());            \
    }                                                                          \
  }                                                                            \
  void LlvmLibcFreeListHeapTest##TestCase::RunTestStore(                       \
      FreeListHeap<LIBC_NAMESPACE::FreeStore> &allocator, size_t N) {          \
    RunTestImpl(allocator, N);                                                 \
  }                                                                            \
  void LlvmLibcFreeListHeapTest##TestCase::RunTestTable(                       \
      FreeListHeap<TestTable> &allocator, size_t N) {                          \
    RunTestImpl(allocator, N);                                                 \
  }                                                                            \
  template <typename allocator_type>                                           \
  void LlvmLibcFreeListHeapTest##TestCase::RunTestImpl(                        \
      allocator_type &allocator, [[maybe_unused]] size_t N)

TEST_FOR_EACH_ALLOCATOR(CanAllocate, 2048) {
  constexpr size_t ALLOC_SIZE = 512;

  void *ptr = allocator.allocate(ALLOC_SIZE);

  ASSERT_NE(ptr, static_cast<void *>(nullptr));
}

TEST_FOR_EACH_ALLOCATOR(AllocationsDontOverlap, 2048) {
  constexpr size_t ALLOC_SIZE = 512;

  void *ptr1 = allocator.allocate(ALLOC_SIZE);
  void *ptr2 = allocator.allocate(ALLOC_SIZE);

  ASSERT_NE(ptr1, static_cast<void *>(nullptr));
  ASSERT_NE(ptr2, static_cast<void *>(nullptr));

  uintptr_t ptr1_start = reinterpret_cast<uintptr_t>(ptr1);
  uintptr_t ptr1_end = ptr1_start + ALLOC_SIZE;
  uintptr_t ptr2_start = reinterpret_cast<uintptr_t>(ptr2);

  EXPECT_GT(ptr2_start, ptr1_end);
}

TEST_FOR_EACH_ALLOCATOR(CanFreeAndRealloc, 2048) {
  constexpr size_t ALLOC_SIZE = 512;

  void *ptr1 = allocator.allocate(ALLOC_SIZE);
  allocator.free(ptr1);
  void *ptr2 = allocator.allocate(ALLOC_SIZE);

  EXPECT_EQ(ptr1, ptr2);
}

TEST_FOR_EACH_ALLOCATOR(ReturnsNullWhenAllocationTooLarge, 2048) {
  EXPECT_EQ(allocator.allocate(N), static_cast<void *>(nullptr));
}

class LlvmLibcFreeListHeapHelper : public LIBC_NAMESPACE::testing::Test {
public:
  template <typename allocator_type> void test_returns_null_when_full() {
    constexpr size_t N = 2048;
    byte buf[N];

    allocator_type allocator(buf);

    bool went_null = false;
    for (size_t i = 0; i < N; i++) {
      if (!allocator.allocate(1)) {
        went_null = true;
        break;
      }
    }
    EXPECT_TRUE(went_null);
    EXPECT_EQ(allocator.allocate(1), static_cast<void *>(nullptr));
  }

  template <typename allocator_type>
  void test_aligned_alloc_unaligned_buffer() {
    byte buf[4096] = {byte(0)};

    // Ensure the underlying buffer is poorly aligned.
    allocator_type allocator(span<byte>(buf).subspan(1));

    constexpr size_t ALIGNMENTS[] = {1, 2, 4, 8, 16, 32, 64, 128, 256};
    constexpr size_t SIZE_SCALES[] = {1, 2, 3, 4, 5};

    for (size_t alignment : ALIGNMENTS) {
      for (size_t scale : SIZE_SCALES) {
        size_t size = alignment * scale;
        void *ptr = allocator.aligned_allocate(alignment, size);
        EXPECT_NE(ptr, static_cast<void *>(nullptr));
        EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % alignment, size_t(0));
        allocator.free(ptr);
      }
    }
  }
};

TEST_F(LlvmLibcFreeListHeapHelper, ReturnsNullWhenFull) {
  test_returns_null_when_full<FreeListHeap<LIBC_NAMESPACE::FreeStore>>();
  test_returns_null_when_full<FreeListHeap<TestTable>>();
}

TEST_FOR_EACH_ALLOCATOR(ReturnedPointersAreAligned, 2048) {
  void *ptr1 = allocator.allocate(1);

  uintptr_t ptr1_start = reinterpret_cast<uintptr_t>(ptr1);
  EXPECT_EQ(ptr1_start % Block::MIN_ALIGN, static_cast<size_t>(0));

  void *ptr2 = allocator.allocate(1);
  uintptr_t ptr2_start = reinterpret_cast<uintptr_t>(ptr2);

  EXPECT_EQ(ptr2_start % Block::MIN_ALIGN, static_cast<size_t>(0));
}

TEST_FOR_EACH_ALLOCATOR(CanRealloc, 2048) {
  constexpr size_t ALLOC_SIZE = 512;
  constexpr size_t kNewAllocSize = 768;

  void *ptr1 = allocator.allocate(ALLOC_SIZE);
  void *ptr2 = allocator.realloc(ptr1, kNewAllocSize);

  ASSERT_NE(ptr1, static_cast<void *>(nullptr));
  ASSERT_NE(ptr2, static_cast<void *>(nullptr));
}

TEST_FOR_EACH_ALLOCATOR(ReallocHasSameContent, 2048) {
  constexpr size_t ALLOC_SIZE = sizeof(int);
  constexpr size_t kNewAllocSize = sizeof(int) * 2;
  // Data inside the allocated block.
  byte data1[ALLOC_SIZE];
  // Data inside the reallocated block.
  byte data2[ALLOC_SIZE];

  int *ptr1 = reinterpret_cast<int *>(allocator.allocate(ALLOC_SIZE));
  *ptr1 = 42;
  LIBC_NAMESPACE::memcpy(data1, ptr1, ALLOC_SIZE);
  int *ptr2 = reinterpret_cast<int *>(allocator.realloc(ptr1, kNewAllocSize));
  LIBC_NAMESPACE::memcpy(data2, ptr2, ALLOC_SIZE);

  ASSERT_NE(ptr1, static_cast<int *>(nullptr));
  ASSERT_NE(ptr2, static_cast<int *>(nullptr));
  // Verify that data inside the allocated and reallocated chunks are the same.
  EXPECT_EQ(LIBC_NAMESPACE::memcmp(data1, data2, ALLOC_SIZE), 0);
}

TEST_FOR_EACH_ALLOCATOR(ReturnsNullReallocFreedPointer, 2048) {
  constexpr size_t ALLOC_SIZE = 512;
  constexpr size_t kNewAllocSize = 256;

  void *ptr1 = allocator.allocate(ALLOC_SIZE);
  allocator.free(ptr1);
  void *ptr2 = allocator.realloc(ptr1, kNewAllocSize);

  EXPECT_EQ(static_cast<void *>(nullptr), ptr2);
}

TEST_FOR_EACH_ALLOCATOR(ReallocSmallerSize, 2048) {
  constexpr size_t ALLOC_SIZE = 512;
  constexpr size_t kNewAllocSize = 256;

  void *ptr1 = allocator.allocate(ALLOC_SIZE);
  void *ptr2 = allocator.realloc(ptr1, kNewAllocSize);

  // For smaller sizes, realloc will shrink the block in-place, returning the
  // same pointer.
  EXPECT_EQ(ptr1, ptr2);
}

constexpr size_t SMALL_HEAP_SIZE = 512;

TEST_FOR_EACH_ALLOCATOR(ReallocShrinkAndAllocateTrailing, SMALL_HEAP_SIZE) {
  constexpr size_t ALLOC_SIZE = (SMALL_HEAP_SIZE * 3) / 4;
  constexpr size_t NEW_ALLOC_SIZE = SMALL_HEAP_SIZE / 4;
  constexpr size_t TRAILING_ALLOC_SIZE = SMALL_HEAP_SIZE / 2;

  void *ptr1 = allocator.allocate(ALLOC_SIZE);
  ASSERT_NE(ptr1, static_cast<void *>(nullptr));

  // Fill the block with a pattern to verify data preservation.
  LIBC_NAMESPACE::inline_memset(ptr1, 0x5A, ALLOC_SIZE);

  // Shrink the block in-place.
  void *ptr2 = allocator.realloc(ptr1, NEW_ALLOC_SIZE);
  EXPECT_EQ(ptr1, ptr2);

  // Verify that the data in the shrunk block is preserved.
  unsigned char *uptr2 = static_cast<unsigned char *>(ptr2);
  for (size_t i = 0; i < NEW_ALLOC_SIZE; ++i)
    EXPECT_EQ(uptr2[i], static_cast<unsigned char>(0x5A));

  // Allocate another trailing block from the newly split/freed space.
  void *ptr3 = allocator.allocate(TRAILING_ALLOC_SIZE);
  ASSERT_NE(ptr3, static_cast<void *>(nullptr));

  // Verify that the trailing block does not overlap with the shrunk block.
  uintptr_t addr2 = reinterpret_cast<uintptr_t>(ptr2);
  uintptr_t addr3 = reinterpret_cast<uintptr_t>(ptr3);
  EXPECT_GE(addr3, addr2 + NEW_ALLOC_SIZE);

  // Verify that writing to the trailing block works without corrupting the
  // shrunk block.
  LIBC_NAMESPACE::inline_memset(ptr3, 0xAA, TRAILING_ALLOC_SIZE);

  for (size_t i = 0; i < NEW_ALLOC_SIZE; ++i)
    EXPECT_EQ(uptr2[i], static_cast<unsigned char>(0x5A));

  // Free the allocations.
  allocator.free(ptr2);
  allocator.free(ptr3);
}

TEST_FOR_EACH_ALLOCATOR(ReallocTooLarge, 2048) {
  constexpr size_t ALLOC_SIZE = 512;
  size_t kNewAllocSize = N * 2; // Large enough to fail.

  void *ptr1 = allocator.allocate(ALLOC_SIZE);
  void *ptr2 = allocator.realloc(ptr1, kNewAllocSize);

  // realloc() will not invalidate the original pointer if realloc() fails
  EXPECT_NE(static_cast<void *>(nullptr), ptr1);
  EXPECT_EQ(static_cast<void *>(nullptr), ptr2);
}

TEST_FOR_EACH_ALLOCATOR(CanCalloc, 2048) {
  constexpr size_t ALLOC_SIZE = 128;
  constexpr size_t NUM = 4;
  constexpr int size = NUM * ALLOC_SIZE;
  constexpr byte zero{0};

  byte *ptr1 = reinterpret_cast<byte *>(allocator.calloc(NUM, ALLOC_SIZE));

  // calloc'd content is zero.
  for (int i = 0; i < size; i++) {
    EXPECT_EQ(ptr1[i], zero);
  }
}

TEST_FOR_EACH_ALLOCATOR(CanCallocWeirdSize, 2048) {
  constexpr size_t ALLOC_SIZE = 143;
  constexpr size_t NUM = 3;
  constexpr int size = NUM * ALLOC_SIZE;
  constexpr byte zero{0};

  byte *ptr1 = reinterpret_cast<byte *>(allocator.calloc(NUM, ALLOC_SIZE));

  // calloc'd content is zero.
  for (int i = 0; i < size; i++) {
    EXPECT_EQ(ptr1[i], zero);
  }
}

TEST_FOR_EACH_ALLOCATOR(CallocTooLarge, 2048) {
  size_t ALLOC_SIZE = N + 1;
  EXPECT_EQ(allocator.calloc(1, ALLOC_SIZE), static_cast<void *>(nullptr));
}

TEST_FOR_EACH_ALLOCATOR(AllocateZero, 2048) {
  void *ptr = allocator.allocate(0);
  ASSERT_EQ(ptr, static_cast<void *>(nullptr));
}

TEST_FOR_EACH_ALLOCATOR(AlignedAlloc, 2048) {
  constexpr size_t ALIGNMENTS[] = {1, 2, 4, 8, 16, 32, 64, 128, 256};
  constexpr size_t SIZE_SCALES[] = {1, 2, 3, 4, 5};

  for (size_t alignment : ALIGNMENTS) {
    for (size_t scale : SIZE_SCALES) {
      size_t size = alignment * scale;
      void *ptr = allocator.aligned_allocate(alignment, size);
      EXPECT_NE(ptr, static_cast<void *>(nullptr));
      EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % alignment, size_t(0));
      allocator.free(ptr);
    }
  }
}

TEST_F(LlvmLibcFreeListHeapHelper, AlignedAllocUnalignedBuffer) {
  test_aligned_alloc_unaligned_buffer<
      FreeListHeap<LIBC_NAMESPACE::FreeStore>>();
  test_aligned_alloc_unaligned_buffer<FreeListHeap<TestTable>>();
}

TEST_FOR_EACH_ALLOCATOR(InvalidAlignedAllocAlignment, 2048) {
  // Must be a power of 2.
  constexpr size_t ALIGNMENTS[] = {4, 8, 16, 32, 64, 128, 256};
  for (size_t alignment : ALIGNMENTS) {
    void *ptr = allocator.aligned_allocate(alignment - 1, alignment - 1);
    EXPECT_EQ(ptr, static_cast<void *>(nullptr));
  }

  // Size must be a multiple of alignment
  for (size_t alignment : ALIGNMENTS) {
    void *ptr = allocator.aligned_allocate(alignment, alignment + 1);
    EXPECT_EQ(ptr, static_cast<void *>(nullptr));
  }

  // Don't accept zero size.
  void *ptr = allocator.aligned_allocate(1, 0);
  EXPECT_EQ(ptr, static_cast<void *>(nullptr));

  // Don't accept zero alignment.
  ptr = allocator.aligned_allocate(0, 8);
  EXPECT_EQ(ptr, static_cast<void *>(nullptr));
}

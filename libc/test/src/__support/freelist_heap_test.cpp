//===-- Unittests for freelist_heap ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/span.h"
#include "src/__support/freelist_heap.h"
#include "src/string/memcmp.h"
#include "src/string/memcpy.h"
#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE {

using LIBC_NAMESPACE::freelist_heap;

// Similar to `LlvmLibcBlockTest` in block_test.cpp, we'd like to run the same
// tests independently for different parameters. In this case, we'd like to test
// functionality for a `FreeListHeap` and the global `freelist_heap` which was
// constinit'd. Functionally, it should operate the same if the FreeListHeap
// were initialized locally at runtime or at compile-time.
//
// Note that calls to `allocate` for each test case here don't always explicitly
// `free` them afterwards, so when testing the global allocator, allocations
// made in tests leak and aren't free'd. This is fine for the purposes of this
// test file.
#define TEST_FOR_EACH_ALLOCATOR(TestCase, BufferSize)                          \
  class LlvmLibcFreeListHeapTest##TestCase : public testing::Test {            \
  public:                                                                      \
    void RunTest(FreeListHeap<> &allocator, [[maybe_unused]] size_t N);        \
  };                                                                           \
  TEST_F(LlvmLibcFreeListHeapTest##TestCase, TestCase) {                       \
    alignas(FreeListHeap<>::BlockType)                                         \
        cpp::byte buf[BufferSize] = {cpp::byte(0)};                            \
    FreeListHeap<> allocator(buf);                                             \
    RunTest(allocator, BufferSize);                                            \
    RunTest(*freelist_heap, freelist_heap->region_size());                     \
  }                                                                            \
  void LlvmLibcFreeListHeapTest##TestCase::RunTest(FreeListHeap<> &allocator,  \
                                                   size_t N)

TEST_FOR_EACH_ALLOCATOR(CanAllocate, 2048) {
  constexpr size_t ALLOC_SIZE = 512;

  void *ptr = allocator.allocate(ALLOC_SIZE);

  ASSERT_NE(ptr, static_cast<void *>(nullptr));
  // In this case, the allocator should be returning us the start of the chunk.
  EXPECT_EQ(ptr, static_cast<void *>(
                     reinterpret_cast<cpp::byte *>(allocator.region_start()) +
                     FreeListHeap<>::BlockType::BLOCK_OVERHEAD));
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
  // There's not really a nice way to test that free works, apart from to try
  // and get that value back again.
  constexpr size_t ALLOC_SIZE = 512;

  void *ptr1 = allocator.allocate(ALLOC_SIZE);
  allocator.free(ptr1);
  void *ptr2 = allocator.allocate(ALLOC_SIZE);

  EXPECT_EQ(ptr1, ptr2);
}

TEST_FOR_EACH_ALLOCATOR(ReturnsNullWhenAllocationTooLarge, 2048) {
  EXPECT_EQ(allocator.allocate(N), static_cast<void *>(nullptr));
}

// NOTE: This doesn't use TEST_FOR_EACH_ALLOCATOR because the first `allocate`
// here will likely actually return a nullptr since the same global allocator
// is used for other test cases and we don't explicitly free them.
TEST(LlvmLibcFreeListHeap, ReturnsNullWhenFull) {
  constexpr size_t N = 2048;
  alignas(FreeListHeap<>::BlockType) cpp::byte buf[N] = {cpp::byte(0)};

  FreeListHeap<> allocator(buf);

  EXPECT_NE(allocator.allocate(N - FreeListHeap<>::BlockType::BLOCK_OVERHEAD),
            static_cast<void *>(nullptr));
  EXPECT_EQ(allocator.allocate(1), static_cast<void *>(nullptr));
}

TEST_FOR_EACH_ALLOCATOR(ReturnedPointersAreAligned, 2048) {
  void *ptr1 = allocator.allocate(1);

  // Should be aligned to native pointer alignment
  uintptr_t ptr1_start = reinterpret_cast<uintptr_t>(ptr1);
  size_t alignment = alignof(void *);

  EXPECT_EQ(ptr1_start % alignment, static_cast<size_t>(0));

  void *ptr2 = allocator.allocate(1);
  uintptr_t ptr2_start = reinterpret_cast<uintptr_t>(ptr2);

  EXPECT_EQ(ptr2_start % alignment, static_cast<size_t>(0));
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
  cpp::byte data1[ALLOC_SIZE];
  // Data inside the reallocated block.
  cpp::byte data2[ALLOC_SIZE];

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

  // For smaller sizes, realloc will not shrink the block.
  EXPECT_EQ(ptr1, ptr2);
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
  constexpr cpp::byte zero{0};

  cpp::byte *ptr1 =
      reinterpret_cast<cpp::byte *>(allocator.calloc(NUM, ALLOC_SIZE));

  // calloc'd content is zero.
  for (int i = 0; i < size; i++) {
    EXPECT_EQ(ptr1[i], zero);
  }
}

TEST_FOR_EACH_ALLOCATOR(CanCallocWeirdSize, 2048) {
  constexpr size_t ALLOC_SIZE = 143;
  constexpr size_t NUM = 3;
  constexpr int size = NUM * ALLOC_SIZE;
  constexpr cpp::byte zero{0};

  cpp::byte *ptr1 =
      reinterpret_cast<cpp::byte *>(allocator.calloc(NUM, ALLOC_SIZE));

  // calloc'd content is zero.
  for (int i = 0; i < size; i++) {
    EXPECT_EQ(ptr1[i], zero);
  }
}

TEST_FOR_EACH_ALLOCATOR(CallocTooLarge, 2048) {
  size_t ALLOC_SIZE = N + 1;
  EXPECT_EQ(allocator.calloc(1, ALLOC_SIZE), static_cast<void *>(nullptr));
}

} // namespace LIBC_NAMESPACE

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for flat_tlsf global heap.
///
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/new.h"
#include "src/__support/CPP/span.h"
#include "src/__support/flat_tlsf/common.h"
#include "src/__support/flat_tlsf/global.h"
#include "src/__support/macros/config.h"
#include "src/string/memory_utils/inline_memcpy.h"
#include "src/string/memory_utils/inline_memset.h"
#include "test/UnitTest/Test.h"
#include <stddef.h> // For size_t, uintptr_t

asm(R"(
.globl _end, __llvm_libc_heap_limit
.bss
_end:
  .fill 8192
__llvm_libc_heap_limit:
)");

using LIBC_NAMESPACE::inline_memcpy;
using LIBC_NAMESPACE::inline_memset;
using LIBC_NAMESPACE::cpp::byte;
using LIBC_NAMESPACE::cpp::span;
using LIBC_NAMESPACE::flat_tlsf::CHUNK_UNIT;
using LIBC_NAMESPACE::flat_tlsf::flat_tlsf_heap;
using LIBC_NAMESPACE::flat_tlsf::FlatTlsfHeap;
using LIBC_NAMESPACE::flat_tlsf::FlatTlsfHeapBuffer;

// We want to test both a locally-instantiated FlatTlsfHeap and the global
// flat_tlsf_heap pointer.
#define TEST_FOR_EACH_ALLOCATOR(TestCase, BufferSize)                          \
  class LlvmLibcFlatTlsfGlobalTest##TestCase                                   \
      : public LIBC_NAMESPACE::testing::Test {                                 \
  public:                                                                      \
    FlatTlsfHeapBuffer<BufferSize> fake_global_buffer;                         \
    void SetUp() override {                                                    \
      flat_tlsf_heap =                                                         \
          new (&fake_global_buffer) FlatTlsfHeapBuffer<BufferSize>;            \
    }                                                                          \
    void RunTest(FlatTlsfHeap &allocator, [[maybe_unused]] size_t N);          \
  };                                                                           \
  TEST_F(LlvmLibcFlatTlsfGlobalTest##TestCase, TestCase) {                     \
    alignas(8) byte buf[BufferSize] = {byte(0)};                               \
    FlatTlsfHeap allocator(buf);                                               \
    RunTest(allocator, BufferSize);                                            \
    RunTest(*flat_tlsf_heap, flat_tlsf_heap->region().size());                 \
  }                                                                            \
  void LlvmLibcFlatTlsfGlobalTest##TestCase::RunTest(                          \
      FlatTlsfHeap &allocator, [[maybe_unused]] size_t N)

TEST_FOR_EACH_ALLOCATOR(CanAllocate, 8192) {
  constexpr size_t ALLOC_SIZE = 256;

  void *ptr = allocator.allocate(ALLOC_SIZE);

  ASSERT_NE(ptr, static_cast<void *>(nullptr));
}

TEST_FOR_EACH_ALLOCATOR(AllocationsDontOverlap, 8192) {
  constexpr size_t ALLOC_SIZE = 256;

  void *ptr1 = allocator.allocate(ALLOC_SIZE);
  void *ptr2 = allocator.allocate(ALLOC_SIZE);

  ASSERT_NE(ptr1, static_cast<void *>(nullptr));
  ASSERT_NE(ptr2, static_cast<void *>(nullptr));

  uintptr_t ptr1_start = reinterpret_cast<uintptr_t>(ptr1);
  uintptr_t ptr1_end = ptr1_start + ALLOC_SIZE;
  uintptr_t ptr2_start = reinterpret_cast<uintptr_t>(ptr2);

  EXPECT_GT(ptr2_start, ptr1_end);
}

TEST_FOR_EACH_ALLOCATOR(CanFreeAndRealloc, 8192) {
  constexpr size_t ALLOC_SIZE = 256;

  void *ptr1 = allocator.allocate(ALLOC_SIZE);
  allocator.free(ptr1);
  void *ptr2 = allocator.allocate(ALLOC_SIZE);

  EXPECT_EQ(ptr1, ptr2);
}

TEST_FOR_EACH_ALLOCATOR(ReturnsNullWhenAllocationTooLarge, 8192) {
  EXPECT_EQ(allocator.allocate(N), static_cast<void *>(nullptr));
}

TEST(LlvmLibcFlatTlsfGlobal, ReturnsNullWhenFull) {
  constexpr size_t N = 8192;
  alignas(8) byte buf[N];

  FlatTlsfHeap allocator(buf);

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

TEST_FOR_EACH_ALLOCATOR(ReturnedPointersAreAligned, 8192) {
  void *ptr1 = allocator.allocate(1);

  uintptr_t ptr1_start = reinterpret_cast<uintptr_t>(ptr1);
  EXPECT_EQ(ptr1_start % CHUNK_UNIT, static_cast<size_t>(0));

  void *ptr2 = allocator.allocate(1);
  uintptr_t ptr2_start = reinterpret_cast<uintptr_t>(ptr2);

  EXPECT_EQ(ptr2_start % CHUNK_UNIT, static_cast<size_t>(0));
}

TEST_FOR_EACH_ALLOCATOR(CanRealloc, 8192) {
  constexpr size_t ALLOC_SIZE = 256;
  constexpr size_t kNewAllocSize = 512;

  void *ptr1 = allocator.allocate(ALLOC_SIZE);
  void *ptr2 = allocator.realloc(ptr1, kNewAllocSize);

  ASSERT_NE(ptr1, static_cast<void *>(nullptr));
  ASSERT_NE(ptr2, static_cast<void *>(nullptr));
}

TEST_FOR_EACH_ALLOCATOR(ReallocHasSameContent, 8192) {
  constexpr size_t ALLOC_SIZE = sizeof(int);
  constexpr size_t kNewAllocSize = sizeof(int) * 2;
  byte data1[ALLOC_SIZE];
  byte data2[ALLOC_SIZE];

  int *ptr1 = reinterpret_cast<int *>(allocator.allocate(ALLOC_SIZE));
  *ptr1 = 42;
  inline_memcpy(data1, ptr1, ALLOC_SIZE);
  int *ptr2 = reinterpret_cast<int *>(allocator.realloc(ptr1, kNewAllocSize));
  inline_memcpy(data2, ptr2, ALLOC_SIZE);

  ASSERT_NE(ptr1, static_cast<int *>(nullptr));
  ASSERT_NE(ptr2, static_cast<int *>(nullptr));

  // Verify that data inside the allocated and reallocated chunks are the same.
  bool match = true;
  for (size_t i = 0; i < ALLOC_SIZE; ++i) {
    if (data1[i] != data2[i]) {
      match = false;
      break;
    }
  }
  EXPECT_TRUE(match);
}

TEST_FOR_EACH_ALLOCATOR(ReturnsNullReallocFreedPointer, 8192) {
  constexpr size_t ALLOC_SIZE = 256;
  constexpr size_t kNewAllocSize = 128;

  void *ptr1 = allocator.allocate(ALLOC_SIZE);
  allocator.free(ptr1);
  void *ptr2 = allocator.realloc(ptr1, kNewAllocSize);

  EXPECT_EQ(static_cast<void *>(nullptr), ptr2);
}

TEST_FOR_EACH_ALLOCATOR(ReallocSmallerSize, 8192) {
  constexpr size_t ALLOC_SIZE = 256;
  constexpr size_t kNewAllocSize = 128;

  void *ptr1 = allocator.allocate(ALLOC_SIZE);
  void *ptr2 = allocator.realloc(ptr1, kNewAllocSize);

  EXPECT_EQ(ptr1, ptr2);
}

TEST_FOR_EACH_ALLOCATOR(ReallocTooLarge, 8192) {
  constexpr size_t ALLOC_SIZE = 256;
  size_t kNewAllocSize = N * 2;

  void *ptr1 = allocator.allocate(ALLOC_SIZE);
  void *ptr2 = allocator.realloc(ptr1, kNewAllocSize);

  EXPECT_NE(static_cast<void *>(nullptr), ptr1);
  EXPECT_EQ(static_cast<void *>(nullptr), ptr2);
}

TEST_FOR_EACH_ALLOCATOR(CanCalloc, 8192) {
  constexpr size_t ALLOC_SIZE = 64;
  constexpr size_t NUM = 4;
  constexpr int size = NUM * ALLOC_SIZE;
  constexpr byte zero{0};

  byte *ptr1 = reinterpret_cast<byte *>(allocator.calloc(NUM, ALLOC_SIZE));

  for (int i = 0; i < size; i++) {
    EXPECT_EQ(ptr1[i], zero);
  }
}

TEST_FOR_EACH_ALLOCATOR(CanCallocWeirdSize, 8192) {
  constexpr size_t ALLOC_SIZE = 143;
  constexpr size_t NUM = 3;
  constexpr int size = NUM * ALLOC_SIZE;
  constexpr byte zero{0};

  byte *ptr1 = reinterpret_cast<byte *>(allocator.calloc(NUM, ALLOC_SIZE));

  for (int i = 0; i < size; i++) {
    EXPECT_EQ(ptr1[i], zero);
  }
}

TEST_FOR_EACH_ALLOCATOR(CallocTooLarge, 8192) {
  size_t ALLOC_SIZE = N + 1;
  EXPECT_EQ(allocator.calloc(1, ALLOC_SIZE), static_cast<void *>(nullptr));
}

TEST_FOR_EACH_ALLOCATOR(AllocateZero, 8192) {
  void *ptr = allocator.allocate(0);
  ASSERT_EQ(ptr, static_cast<void *>(nullptr));
}

TEST_FOR_EACH_ALLOCATOR(AlignedAlloc, 8192) {
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

TEST(LlvmLibcFlatTlsfGlobal, AlignedAllocUnalignedBuffer) {
  alignas(8) byte buf[4096] = {byte(0)};

  FlatTlsfHeap allocator(span<byte>(buf).subspan(1));

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

TEST_FOR_EACH_ALLOCATOR(InvalidAlignedAllocAlignment, 8192) {
  constexpr size_t ALIGNMENTS[] = {4, 8, 16, 32, 64, 128, 256};
  for (size_t alignment : ALIGNMENTS) {
    void *ptr = allocator.aligned_allocate(alignment - 1, alignment - 1);
    EXPECT_EQ(ptr, static_cast<void *>(nullptr));
  }

  for (size_t alignment : ALIGNMENTS) {
    void *ptr = allocator.aligned_allocate(alignment, alignment + 1);
    EXPECT_EQ(ptr, static_cast<void *>(nullptr));
  }

  void *ptr = allocator.aligned_allocate(1, 0);
  EXPECT_EQ(ptr, static_cast<void *>(nullptr));

  ptr = allocator.aligned_allocate(0, 8);
  EXPECT_EQ(ptr, static_cast<void *>(nullptr));
}

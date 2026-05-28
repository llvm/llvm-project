//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for flat_tlsf Heap.
///
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/algorithm.h"
#include "src/__support/CPP/limits.h"
#include "src/__support/flat_tlsf/binning.h"
#include "src/__support/flat_tlsf/chunk.h"
#include "src/__support/flat_tlsf/common.h"
#include "src/__support/flat_tlsf/heap.h"
#include "src/__support/flat_tlsf/node.h"
#include "src/__support/flat_tlsf/tag.h"
#include "src/string/memory_utils/inline_memset.h"
#include "test/UnitTest/Test.h"
#include <stddef.h> // For max_align_t, size_t

using LIBC_NAMESPACE::cpp::max;
using LIBC_NAMESPACE::cpp::min;
using LIBC_NAMESPACE::cpp::numeric_limits;
using LIBC_NAMESPACE::flat_tlsf::Binning;
using LIBC_NAMESPACE::flat_tlsf::Byte;
using LIBC_NAMESPACE::flat_tlsf::CHUNK_UNIT;
using LIBC_NAMESPACE::flat_tlsf::GAP_BIN_OFFSET;
using LIBC_NAMESPACE::flat_tlsf::GAP_HIGH_SIZE_OFFSET;
using LIBC_NAMESPACE::flat_tlsf::GAP_LOW_SIZE_OFFSET;
using LIBC_NAMESPACE::flat_tlsf::GAP_NODE_OFFSET;
using LIBC_NAMESPACE::flat_tlsf::Heap;
using LIBC_NAMESPACE::flat_tlsf::Node;
namespace chunk = LIBC_NAMESPACE::flat_tlsf::chunk;
namespace tag = LIBC_NAMESPACE::flat_tlsf::tag;
using LIBC_NAMESPACE::inline_memset;

namespace {

struct Layout {
  size_t size;
  size_t align;
};

LIBC_INLINE constexpr size_t min_first_heap_size() {
  size_t size = chunk::required_chunk_size(Binning::BIN_COUNT * sizeof(Node *));
  size_t max_overhead = CHUNK_UNIT + alignof(size_t) - 1;
  return size + max_overhead;
}

LIBC_INLINE constexpr Layout min_first_heap_layout() {
  size_t size = Binning::BIN_COUNT * sizeof(size_t);
  size_t max_overhead = CHUNK_UNIT;
  return {size + max_overhead, alignof(size_t)};
}

TEST(LlvmLibcFlatTlsfHeapTest, VerifyGapProperties) {
  Heap heap;

  constexpr Layout meta_layout = min_first_heap_layout();
  alignas(meta_layout.align) Byte meta_mem[meta_layout.size] = {};
  Byte *meta_heap_end = heap.claim(meta_mem, meta_layout.size);
  ASSERT_NE(meta_heap_end, static_cast<Byte *>(nullptr));

  Byte gap_mem[999] = {};
  Byte *gap_end = heap.claim(gap_mem, 999);
  ASSERT_NE(gap_end, static_cast<Byte *>(nullptr));

  ASSERT_LE(gap_end, gap_mem + 999);
  ASSERT_GT(gap_end + CHUNK_UNIT, gap_mem + 999);

  Byte *gap_base = chunk::align_up(gap_mem + sizeof(Byte));
  size_t gap_size = static_cast<size_t>(gap_end - gap_base);
  ASSERT_LE(gap_size, size_t{999});
  ASSERT_GT(gap_size, size_t{999} - CHUNK_UNIT * 2);

  uint32_t gap_bin = min(Binning::size_to_bin(gap_size),
                         static_cast<uint32_t>(Binning::BIN_COUNT - 1));

  Node *gap_node_ptr = heap.get_gap_list_head(gap_bin);
  ASSERT_NE(gap_node_ptr, static_cast<Node *>(nullptr));
  ASSERT_EQ(gap_node_ptr, chunk::gap_base_to_node(gap_base));

  Node gap_node = *gap_node_ptr;
  ASSERT_EQ(gap_node.next, static_cast<Node *>(nullptr));
  ASSERT_EQ(gap_node.next_of_prev, heap.get_gap_list_ptr(gap_bin));

  ASSERT_EQ(gap_bin,
            chunk::read_word<uint32_t>(chunk::gap_base_to_bin(gap_base)));
  ASSERT_EQ(gap_size,
            chunk::read_big_endian<size_t>(chunk::gap_base_to_size(gap_base)));

  ASSERT_EQ(chunk::read_big_endian<size_t>(chunk::gap_base_to_size(gap_base)),
            gap_size);
  ASSERT_EQ(chunk::gap_end_to_size_and_flag(gap_end),
            reinterpret_cast<size_t *>(gap_end - sizeof(size_t)));
  ASSERT_EQ(
      chunk::read_big_endian<size_t>(chunk::gap_end_to_size_and_flag(gap_end)),
      gap_size);

  heap.test_deregister_gap(gap_base, gap_size);
}

TEST(LlvmLibcFlatTlsfHeapTest, AllocDeallocTest) {
  Byte arena[5000] = {};
  Heap heap;
  ASSERT_NE(heap.claim(arena, 5000), static_cast<Byte *>(nullptr));

  size_t size = 2435;
  size_t align = 8;
  Byte *allocation = heap.allocate(size, align);
  ASSERT_NE(allocation, static_cast<Byte *>(nullptr));

  for (size_t i = 0; i < size; ++i) {
    allocation[i] = static_cast<Byte>(0xCD);
  }

  heap.deallocate(allocation, size);
}

TEST(LlvmLibcFlatTlsfHeapTest, AllocFailTest) {
  constexpr size_t arena_size = min_first_heap_size() + 100 + CHUNK_UNIT;
  Byte arena[arena_size] = {};
  Heap heap;
  ASSERT_NE(heap.claim(arena, arena_size), static_cast<Byte *>(nullptr));

  Byte *a1 = heap.allocate(8, 8);
  ASSERT_NE(a1, static_cast<Byte *>(nullptr));

  size_t large_size = 1234 + CHUNK_UNIT;
  Byte *a2 = heap.allocate(large_size, 8);
  ASSERT_EQ(a2, static_cast<Byte *>(nullptr));
}

TEST(LlvmLibcFlatTlsfHeapTest, AllocOverflowTest) {
  alignas(64) Byte arena[5000] = {};
  Heap heap;
  ASSERT_NE(heap.claim(arena, 5000), static_cast<Byte *>(nullptr));

  // Test malloc overflow
  ASSERT_EQ(heap.malloc(numeric_limits<size_t>::max()),
            static_cast<void *>(nullptr));

  // Test aligned_alloc overflow
  ASSERT_EQ(heap.aligned_alloc(8, numeric_limits<size_t>::max() - 100),
            static_cast<void *>(nullptr));

  // Test realloc overflow
  void *ptr = heap.malloc(10);
  ASSERT_NE(ptr, static_cast<void *>(nullptr));
  ASSERT_EQ(heap.realloc(ptr, numeric_limits<size_t>::max()),
            static_cast<void *>(nullptr));
}

TEST(LlvmLibcFlatTlsfHeapTest, ClaimHeapThatsTooSmall) {
  alignas(8) Byte tiny_heap[200] = {};
  Heap heap;
  ASSERT_EQ(heap.claim(tiny_heap, 200), static_cast<Byte *>(nullptr));

  ASSERT_EQ(heap.get_gap_list(), static_cast<Node **>(nullptr));
  ASSERT_GE(heap.get_available().bit_scan_after(0),
            static_cast<uint32_t>(Binning::BIN_COUNT));
}

TEST(LlvmLibcFlatTlsfHeapTest, ClaimSmallHeapAfterMetadataIsAllocated) {
  constexpr Layout meta_layout = min_first_heap_layout();
  alignas(meta_layout.align) Byte big_heap[meta_layout.size] = {};

  Heap heap;
  ASSERT_NE(heap.claim(big_heap, meta_layout.size),
            static_cast<Byte *>(nullptr));

  ASSERT_NE(heap.get_gap_list(), static_cast<Node **>(nullptr));
  ASSERT_GE(heap.get_available().bit_scan_after(0),
            static_cast<uint32_t>(Binning::BIN_COUNT));

  alignas(8) Byte tiny_heap[300] = {};
  ASSERT_NE(heap.claim(tiny_heap, 300), static_cast<Byte *>(nullptr));
}

TEST(LlvmLibcFlatTlsfHeapTest, MallocFreeBasic) {
  Byte arena[5000] = {};
  Heap heap;
  ASSERT_NE(heap.claim(arena, 5000), static_cast<Byte *>(nullptr));

  // Test simple malloc/free
  void *p1 = heap.malloc(100);
  ASSERT_NE(p1, static_cast<void *>(nullptr));
  ASSERT_EQ(reinterpret_cast<uintptr_t>(p1) % alignof(max_align_t), size_t{0});

  inline_memset(p1, 0xAB, 100);
  heap.free(p1);

  // Test aligned_alloc
  void *p2 = heap.aligned_alloc(64, 100);
  ASSERT_NE(p2, static_cast<void *>(nullptr));
  ASSERT_EQ(reinterpret_cast<uintptr_t>(p2) % 64, size_t{0});
  inline_memset(p2, 0xBC, 100);
  heap.free(p2);

  // Test free(nullptr) is a safe no-op
  heap.free(nullptr);
}

TEST(LlvmLibcFlatTlsfHeapTest, ReallocBasic) {
  Byte arena[5000] = {};
  Heap heap;
  ASSERT_NE(heap.claim(arena, 5000), static_cast<Byte *>(nullptr));

  // realloc(nullptr, size) is equivalent to malloc
  void *p1 = heap.realloc(nullptr, 100);
  ASSERT_NE(p1, static_cast<void *>(nullptr));
  inline_memset(p1, 0x11, 100);

  // Grow in place (or via move)
  void *p2 = heap.realloc(p1, 200);
  ASSERT_NE(p2, static_cast<void *>(nullptr));
  Byte *p2_bytes = static_cast<Byte *>(p2);
  for (size_t i = 0; i < 100; ++i) {
    ASSERT_EQ(p2_bytes[i], static_cast<Byte>(0x11));
  }

  // Shrink in place
  void *p3 = heap.realloc(p2, 50);
  ASSERT_EQ(p3, p2); // Shrink should always be in-place
  Byte *p3_bytes = static_cast<Byte *>(p3);
  for (size_t i = 0; i < 50; ++i) {
    ASSERT_EQ(p3_bytes[i], static_cast<Byte>(0x11));
  }

  // realloc(ptr, 0) is equivalent to free
  void *p4 = heap.realloc(p3, 0);
  ASSERT_EQ(p4, static_cast<void *>(nullptr));
}

} // namespace

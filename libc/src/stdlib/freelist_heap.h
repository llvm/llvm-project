//===-- Interface for freelist_heap ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDLIB_FREELIST_HEAP_H
#define LLVM_LIBC_SRC_STDLIB_FREELIST_HEAP_H

#include <stddef.h>

#include "block.h"
#include "freelist.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/CPP/span.h"
#include "src/__support/libc_assert.h"
#include "src/string/memory_utils/inline_memcpy.h"
#include "src/string/memory_utils/inline_memset.h"

namespace LIBC_NAMESPACE {

using cpp::optional;
using cpp::span;

static constexpr cpp::array<size_t, 6> DEFAULT_BUCKETS{16,  32,  64,
                                                       128, 256, 512};

template <size_t NUM_BUCKETS = DEFAULT_BUCKETS.size()> class FreeListHeap {
public:
  using BlockType = Block<>;

  struct HeapStats {
    size_t total_bytes;
    size_t bytes_allocated;
    size_t cumulative_allocated;
    size_t cumulative_freed;
    size_t total_allocate_calls;
    size_t total_free_calls;
  };
  FreeListHeap(span<cpp::byte> region);

  void *allocate(size_t size);
  void free(void *ptr);
  void *realloc(void *ptr, size_t size);
  void *calloc(size_t num, size_t size);

  const HeapStats &heap_stats() const { return heap_stats_; }

private:
  span<cpp::byte> block_to_span(BlockType *block) {
    return span<cpp::byte>(block->usable_space(), block->inner_size());
  }

  span<cpp::byte> region_;
  FreeList<NUM_BUCKETS> freelist_;
  HeapStats heap_stats_;
};

template <size_t NUM_BUCKETS>
FreeListHeap<NUM_BUCKETS>::FreeListHeap(span<cpp::byte> region)
    : region_(region), freelist_(DEFAULT_BUCKETS), heap_stats_() {
  auto result = BlockType::init(region);
  BlockType *block = *result;

  freelist_.add_chunk(block_to_span(block));

  heap_stats_.total_bytes = region.size();
}

template <size_t NUM_BUCKETS>
void *FreeListHeap<NUM_BUCKETS>::allocate(size_t size) {
  // Find a chunk in the freelist. Split it if needed, then return
  auto chunk = freelist_.find_chunk(size);

  if (chunk.data() == nullptr)
    return nullptr;
  freelist_.remove_chunk(chunk);

  BlockType *chunk_block = BlockType::from_usable_space(chunk.data());

  // Split that chunk. If there's a leftover chunk, add it to the freelist
  optional<BlockType *> result = BlockType::split(chunk_block, size);
  if (result)
    freelist_.add_chunk(block_to_span(*result));

  chunk_block->mark_used();

  heap_stats_.bytes_allocated += size;
  heap_stats_.cumulative_allocated += size;
  heap_stats_.total_allocate_calls += 1;

  return chunk_block->usable_space();
}

template <size_t NUM_BUCKETS> void FreeListHeap<NUM_BUCKETS>::free(void *ptr) {
  cpp::byte *bytes = static_cast<cpp::byte *>(ptr);

  LIBC_ASSERT(bytes >= region_.data() && bytes < region_.data() + region_.size() && "Invalid pointer");

  BlockType *chunk_block = BlockType::from_usable_space(bytes);

  size_t size_freed = chunk_block->inner_size();
  LIBC_ASSERT(chunk_block->used() && "The block is not in-use");
  chunk_block->mark_free();

  // Can we combine with the left or right blocks?
  BlockType *prev = chunk_block->prev();
  BlockType *next = nullptr;

  if (!chunk_block->last())
    next = chunk_block->next();

  if (prev != nullptr && !prev->used()) {
    // Remove from freelist and merge
    freelist_.remove_chunk(block_to_span(prev));
    chunk_block = chunk_block->prev();
    BlockType::merge_next(chunk_block);
  }

  if (next != nullptr && !next->used()) {
    freelist_.remove_chunk(block_to_span(next));
    BlockType::merge_next(chunk_block);
  }
  // Add back to the freelist
  freelist_.add_chunk(block_to_span(chunk_block));

  heap_stats_.bytes_allocated -= size_freed;
  heap_stats_.cumulative_freed += size_freed;
  heap_stats_.total_free_calls += 1;
}

// Follows contract of the C standard realloc() function
// If ptr is free'd, will return nullptr.
template <size_t NUM_BUCKETS>
void *FreeListHeap<NUM_BUCKETS>::realloc(void *ptr, size_t size) {
  if (size == 0) {
    free(ptr);
    return nullptr;
  }

  // If the pointer is nullptr, allocate a new memory.
  if (ptr == nullptr)
    return allocate(size);

  cpp::byte *bytes = static_cast<cpp::byte *>(ptr);

  if (bytes < region_.data() || bytes >= region_.data() + region_.size())
    return nullptr;

  BlockType *chunk_block = BlockType::from_usable_space(bytes);
  if (!chunk_block->used())
    return nullptr;
  size_t old_size = chunk_block->inner_size();

  // Do nothing and return ptr if the required memory size is smaller than
  // the current size.
  if (old_size >= size)
    return ptr;

  void *new_ptr = allocate(size);
  // Don't invalidate ptr if allocate(size) fails to initilize the memory.
  if (new_ptr == nullptr)
    return nullptr;
  LIBC_NAMESPACE::inline_memcpy(new_ptr, ptr, old_size);

  free(ptr);
  return new_ptr;
}

template <size_t NUM_BUCKETS>
void *FreeListHeap<NUM_BUCKETS>::calloc(size_t num, size_t size) {
  void *ptr = allocate(num * size);
  if (ptr != nullptr)
    LIBC_NAMESPACE::inline_memset(ptr, 0, num * size);
  return ptr;
}

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_STDLIB_FREELIST_HEAP_H

//===-- Interface for freelist_heap ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FREELIST_HEAP_H
#define LLVM_LIBC_SRC___SUPPORT_FREELIST_HEAP_H

#include <stddef.h>

#include "block.h"
#include "freelist.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/CPP/span.h"
#include "src/__support/libc_assert.h"
#include "src/__support/macros/config.h"
#include "src/string/memory_utils/inline_memcpy.h"
#include "src/string/memory_utils/inline_memset.h"

namespace LIBC_NAMESPACE_DECL {

extern "C" cpp::byte _end;
extern "C" cpp::byte __llvm_libc_heap_limit;

using cpp::optional;
using cpp::span;

inline constexpr bool IsPow2(size_t x) { return x && (x & (x - 1)) == 0; }

static constexpr cpp::array<size_t, 6> DEFAULT_BUCKETS{16,  32,  64,
                                                       128, 256, 512};

template <size_t NUM_BUCKETS = DEFAULT_BUCKETS.size()> class FreeListHeap {
public:
  using BlockType = Block<>;
  using FreeListType = FreeList<NUM_BUCKETS>;

  static constexpr size_t MIN_ALIGNMENT =
      cpp::max(BlockType::ALIGNMENT, alignof(max_align_t));

  constexpr FreeListHeap() : begin_(&_end), end_(&__llvm_libc_heap_limit) {}

  constexpr FreeListHeap(span<cpp::byte> region)
      : begin_(region.begin()), end_(region.end()) {}

  void *allocate(size_t size);
  void *aligned_allocate(size_t alignment, size_t size);
  // NOTE: All pointers passed to free must come from one of the other
  // allocation functions: `allocate`, `aligned_allocate`, `realloc`, `calloc`.
  void free(void *ptr);
  void *realloc(void *ptr, size_t size);
  void *calloc(size_t num, size_t size);

  cpp::span<cpp::byte> region() const { return {begin_, end_}; }

private:
  void init();

  void *allocate_impl(size_t alignment, size_t size);

  span<cpp::byte> block_to_span(BlockType *block) {
    return span<cpp::byte>(block->usable_space(), block->inner_size());
  }

  bool is_valid_ptr(void *ptr) { return ptr >= begin_ && ptr < end_; }

  bool is_initialized_ = false;
  cpp::byte *begin_;
  cpp::byte *end_;
  FreeListType freelist_{DEFAULT_BUCKETS};
};

template <size_t BUFF_SIZE, size_t NUM_BUCKETS = DEFAULT_BUCKETS.size()>
class FreeListHeapBuffer : public FreeListHeap<NUM_BUCKETS> {
  using parent = FreeListHeap<NUM_BUCKETS>;
  using FreeListNode = typename parent::FreeListType::FreeListNode;

public:
  constexpr FreeListHeapBuffer()
      : FreeListHeap<NUM_BUCKETS>{buffer}, buffer{} {}

private:
  cpp::byte buffer[BUFF_SIZE];
};

template <size_t NUM_BUCKETS> void FreeListHeap<NUM_BUCKETS>::init() {
  LIBC_ASSERT(!is_initialized_ && "duplicate initialization");
  auto result = BlockType::init(region());
  BlockType *block = *result;
  freelist_.add_chunk(block_to_span(block));
  is_initialized_ = true;
}

template <size_t NUM_BUCKETS>
void *FreeListHeap<NUM_BUCKETS>::allocate_impl(size_t alignment, size_t size) {
  if (size == 0)
    return nullptr;

  if (!is_initialized_)
    init();

  // Find a chunk in the freelist. Split it if needed, then return.
  auto chunk =
      freelist_.find_chunk_if([alignment, size](span<cpp::byte> chunk) {
        BlockType *block = BlockType::from_usable_space(chunk.data());
        return block->can_allocate(alignment, size);
      });

  if (chunk.data() == nullptr)
    return nullptr;
  freelist_.remove_chunk(chunk);

  BlockType *chunk_block = BlockType::from_usable_space(chunk.data());
  LIBC_ASSERT(!chunk_block->used());

  // Split that chunk. If there's a leftover chunk, add it to the freelist
  auto block_info = BlockType::allocate(chunk_block, alignment, size);
  if (block_info.next)
    freelist_.add_chunk(block_to_span(block_info.next));
  if (block_info.prev)
    freelist_.add_chunk(block_to_span(block_info.prev));
  chunk_block = block_info.block;

  chunk_block->mark_used();

  return chunk_block->usable_space();
}

template <size_t NUM_BUCKETS>
void *FreeListHeap<NUM_BUCKETS>::allocate(size_t size) {
  return allocate_impl(MIN_ALIGNMENT, size);
}

template <size_t NUM_BUCKETS>
void *FreeListHeap<NUM_BUCKETS>::aligned_allocate(size_t alignment,
                                                  size_t size) {
  // The alignment must be an integral power of two.
  if (!IsPow2(alignment))
    return nullptr;

  // The size parameter must be an integral multiple of alignment.
  if (size % alignment != 0)
    return nullptr;

  return allocate_impl(alignment, size);
}

template <size_t NUM_BUCKETS> void FreeListHeap<NUM_BUCKETS>::free(void *ptr) {
  cpp::byte *bytes = static_cast<cpp::byte *>(ptr);

  LIBC_ASSERT(is_valid_ptr(bytes) && "Invalid pointer");

  BlockType *chunk_block = BlockType::from_usable_space(bytes);
  LIBC_ASSERT(chunk_block->next() && "sentinel last block cannot be freed");
  LIBC_ASSERT(chunk_block->used() && "The block is not in-use");
  chunk_block->mark_free();

  // Can we combine with the left or right blocks?
  BlockType *prev_free = chunk_block->prev_free();
  BlockType *next = chunk_block->next();

  if (prev_free != nullptr) {
    // Remove from freelist and merge
    freelist_.remove_chunk(block_to_span(prev_free));
    chunk_block = prev_free;
    chunk_block->merge_next();
  }
  if (!next->used()) {
    freelist_.remove_chunk(block_to_span(next));
    chunk_block->merge_next();
  }
  // Add back to the freelist
  freelist_.add_chunk(block_to_span(chunk_block));
}

// Follows constract of the C standard realloc() function
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

  if (!is_valid_ptr(bytes))
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

extern FreeListHeap<> *freelist_heap;

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_FREELIST_HEAP_H

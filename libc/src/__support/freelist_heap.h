//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for freelist_heap.
///
//===----------------------------------------------------------------------===//

#include "block.h"
#include "freestore.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/CPP/span.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/libc_assert.h"
#include "src/__support/macros/config.h"
#include "src/string/memory_utils/inline_memcpy.h"
#include "src/string/memory_utils/inline_memset.h"

#ifndef LLVM_LIBC_SRC___SUPPORT_FREELIST_HEAP_H
#define LLVM_LIBC_SRC___SUPPORT_FREELIST_HEAP_H

namespace LIBC_NAMESPACE_DECL {

extern "C" cpp::byte _end;
extern "C" cpp::byte __llvm_libc_heap_limit;

using cpp::optional;
using cpp::span;

LIBC_INLINE constexpr bool IsPow2(size_t x) { return x && (x & (x - 1)) == 0; }

template <typename T> LIBC_INLINE void init_free_store(T &, size_t) {}

template <>
LIBC_INLINE void init_free_store<FreeStore>(FreeStore &store, size_t size) {
  store.set_range({0, cpp::bit_ceil(size)});
}

template <typename FreeStoreType = FreeStore> class FreeListHeap {
public:
  constexpr FreeListHeap() : begin(&_end), end(&__llvm_libc_heap_limit) {}

  constexpr FreeListHeap(span<cpp::byte> region)
      : begin(region.begin()), end(region.end()) {}

  void *allocate(size_t size);
  void *aligned_allocate(size_t alignment, size_t size);
  // NOTE: All pointers passed to free must come from one of the other
  // allocation functions: `allocate`, `aligned_allocate`, `realloc`, `calloc`.
  void free(void *ptr);
  void *realloc(void *ptr, size_t size);
  void *calloc(size_t num, size_t size);

  cpp::span<cpp::byte> region() const { return {begin, end}; }

private:
  void init();

  void *allocate_impl(size_t alignment, size_t size);

  bool shrink_in_place(Block *block, size_t size);

  span<cpp::byte> block_to_span(Block *block) {
    return span<cpp::byte>(block->usable_space(), block->inner_size());
  }

  bool is_valid_ptr(void *ptr) { return ptr >= begin && ptr < end; }

  cpp::byte *begin;
  cpp::byte *end;
  bool is_initialized = false;
  FreeStoreType free_store;
};

// Deduction guide for FreeListHeap to allow bracket-less instantiation
// e.g. FreeListHeap allocator(buf); -> FreeListHeap<FreeStore>
FreeListHeap(span<cpp::byte>) -> FreeListHeap<FreeStore>;

template <size_t BUFF_SIZE, typename FreeStoreType = FreeStore>
class FreeListHeapBuffer : public FreeListHeap<FreeStoreType> {
public:
  constexpr FreeListHeapBuffer()
      : FreeListHeap<FreeStoreType>{buffer}, buffer{} {}

private:
  cpp::byte buffer[BUFF_SIZE];
};

// Specialization for the default FreeStore to allow conversion to FreeListHeap*
template <size_t BUFF_SIZE>
class FreeListHeapBuffer<BUFF_SIZE, FreeStore> : public FreeListHeap<> {
public:
  constexpr FreeListHeapBuffer() : FreeListHeap<>{buffer}, buffer{} {}

private:
  cpp::byte buffer[BUFF_SIZE];
};

extern FreeListHeap<FreeStore> *freelist_heap;

template <typename FreeStoreType>
LIBC_INLINE void FreeListHeap<FreeStoreType>::init() {
  LIBC_ASSERT(!is_initialized && "duplicate initialization");
  auto result = Block::init(region());
  Block *block = *result;
  init_free_store<FreeStoreType>(free_store, block->inner_size());
  free_store.insert(block);
  is_initialized = true;
}

template <typename FreeStoreType>
LIBC_INLINE void *FreeListHeap<FreeStoreType>::allocate_impl(size_t alignment,
                                                             size_t size) {
  if (size == 0)
    return nullptr;

  if (!is_initialized)
    init();

  size_t request_size = Block::min_size_for_allocation(alignment, size);
  if (!request_size)
    return nullptr;

  Block *block = free_store.find_and_remove_fit(request_size);
  if (!block)
    return nullptr;

  auto block_info = Block::allocate(block, alignment, size);
  if (block_info.next)
    free_store.insert(block_info.next);
  if (block_info.prev)
    free_store.insert(block_info.prev);

  block_info.block->mark_used();
  return block_info.block->usable_space();
}

template <typename FreeStoreType>
LIBC_INLINE void *FreeListHeap<FreeStoreType>::allocate(size_t size) {
  return allocate_impl(Block::MIN_ALIGN, size);
}

template <typename FreeStoreType>
LIBC_INLINE void *
FreeListHeap<FreeStoreType>::aligned_allocate(size_t alignment, size_t size) {
  // The alignment must be an integral power of two.
  if (!IsPow2(alignment))
    return nullptr;

  // The size parameter must be an integral multiple of alignment.
  if (size % alignment != 0)
    return nullptr;

  // The minimum alignment supported by Block is MIN_ALIGN.
  alignment = cpp::max(alignment, Block::MIN_ALIGN);

  return allocate_impl(alignment, size);
}

template <typename FreeStoreType>
LIBC_INLINE void FreeListHeap<FreeStoreType>::free(void *ptr) {
  if (ptr == nullptr)
    return;

  cpp::byte *bytes = static_cast<cpp::byte *>(ptr);

  LIBC_ASSERT(is_valid_ptr(bytes) && "Invalid pointer");

  Block *block = Block::from_usable_space(bytes);
  LIBC_ASSERT(block->next() && "sentinel last block cannot be freed");
  LIBC_ASSERT(block->used() && "double free");
  block->mark_free();

  // Can we combine with the left or right blocks?
  Block *prev_free = block->prev_free();
  Block *next = block->next();

  if (prev_free != nullptr) {
    // Remove from free store and merge.
    free_store.remove(prev_free);
    block = prev_free;
    block->merge_next();
  }
  if (!next->used()) {
    free_store.remove(next);
    block->merge_next();
  }
  // Add back to the freelist
  free_store.insert(block);
}

template <typename FreeStoreType>
LIBC_INLINE bool FreeListHeap<FreeStoreType>::shrink_in_place(Block *block,
                                                              size_t size) {
  size_t min_outer_size = Block::outer_size(cpp::max(size, sizeof(size_t)));
  uintptr_t next_block_start = Block::next_possible_block_start(
      reinterpret_cast<uintptr_t>(block) + min_outer_size, Block::MIN_ALIGN);
  size_t new_outer_size = next_block_start - reinterpret_cast<uintptr_t>(block);
  // only split the block if the trailing part can be inserted into freelist
  if (block->outer_size() >= new_outer_size &&
      block->outer_size() - new_outer_size >= FreeStoreType::MIN_OUTER_SIZE) {
    // We must temporarily mark the block as free to allow splitting.
    // A block's usable space overlaps with the next block's prev_ field. When
    // the next block is created via splitting, its header constructor writes to
    // its prev_ field (initializing it to 0). Since the original block is
    // currently in use, this write will corrupt the last sizeof(size_t) bytes
    // of active user data. We back up these bytes here and restore them after
    // the split is completed.
    cpp::byte *overlap_ptr =
        reinterpret_cast<cpp::byte *>(block) + new_outer_size;
    size_t backup;
    LIBC_NAMESPACE::inline_memcpy(&backup, overlap_ptr, sizeof(size_t));
    optional<Block *> next = block->split(size, Block::MIN_ALIGN);

    LIBC_NAMESPACE::inline_memcpy(overlap_ptr, &backup, sizeof(size_t));

    // register the new block on successful split
    if (next.has_value()) {
      Block *next_block = *next;
      Block *right = next_block->next();
      if (right != nullptr && !right->used()) {
        free_store.remove(right);
        next_block->merge_next();
      }
      free_store.insert(next_block);
    }
    return true;
  }
  return false;
}

// Follows contract of the C standard realloc() function
// If ptr is free'd, will return nullptr.
template <typename FreeStoreType>
LIBC_INLINE void *FreeListHeap<FreeStoreType>::realloc(void *ptr, size_t size) {
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

  Block *block = Block::from_usable_space(bytes);
  if (!block->used())
    return nullptr;
  size_t old_size = block->inner_size();

  if (old_size >= size) {
    shrink_in_place(block, size);
    return ptr;
  }

  void *new_ptr = allocate(size);
  // Don't invalidate ptr if allocate(size) fails to initialize the memory.
  if (new_ptr == nullptr)
    return nullptr;
  LIBC_NAMESPACE::inline_memcpy(new_ptr, ptr, old_size);

  free(ptr);
  return new_ptr;
}

template <typename FreeStoreType>
LIBC_INLINE void *FreeListHeap<FreeStoreType>::calloc(size_t num, size_t size) {
  size_t bytes;
  if (__builtin_mul_overflow(num, size, &bytes))
    return nullptr;
  void *ptr = allocate(bytes);
  if (ptr != nullptr)
    LIBC_NAMESPACE::inline_memset(ptr, 0, bytes);
  return ptr;
}

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_FREELIST_HEAP_H

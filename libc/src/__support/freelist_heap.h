//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Interface for freelist_heap.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FREELIST_HEAP_H
#define LLVM_LIBC_SRC___SUPPORT_FREELIST_HEAP_H

#include <stddef.h>

#include "block.h"
#include "freestore.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/CPP/span.h"
#include "src/__support/libc_assert.h"
#include "src/__support/macros/config.h"
#include "src/__support/math_extras.h"
#include "src/string/memory_utils/inline_memcpy.h"
#include "src/string/memory_utils/inline_memset.h"

namespace LIBC_NAMESPACE_DECL {

extern "C" cpp::byte _end;
extern "C" cpp::byte __llvm_libc_heap_limit;

using cpp::optional;
using cpp::span;

LIBC_INLINE constexpr bool IsPow2(size_t x) { return x && (x & (x - 1)) == 0; }

class FreeListHeap {
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
  size_t allocation_size(const void *ptr) const;
  LIBC_INLINE void integrity_check() const {
#ifdef LIBC_COPT_BAREMETAL_HEAP_ENABLE_FREESTORE_ROTATION
    free_stores[0].integrity_check();
    free_stores[1].integrity_check();
#else
    free_store.integrity_check();
#endif
  }

  cpp::span<cpp::byte> region() const { return {begin, end}; }

private:
  void init();

  void *allocate_impl(size_t alignment, size_t size);

  span<cpp::byte> block_to_span(BlockRef block) {
    return span<cpp::byte>(block.usable_space(), block.inner_size());
  }

  bool shrink_in_place(BlockRef block, size_t size);

  bool is_valid_ptr(const void *ptr) const { return ptr >= begin && ptr < end; }

#ifdef LIBC_COPT_BAREMETAL_HEAP_ENABLE_FREESTORE_ROTATION
  LIBC_INLINE int get_store_index(BlockRef block) {
    return block.next().prev_free_store_index();
  }

  LIBC_INLINE void coalesce_and_insert(BlockRef block, int store_idx) {
    block.mark_free(store_idx);
    BlockRef prev = block.prev_free();
    if (prev) {
      if (FreeStore::too_small(prev)) {
        block = prev;
        block.merge_next();
      } else {
        int prev_store = get_store_index(prev);
        if (prev_store == store_idx) {
          free_stores[prev_store].remove(prev);
          block = prev;
          block.merge_next();
        }
      }
    }

    BlockRef next = block.next();
    if (!next.used()) {
      if (FreeStore::too_small(next)) {
        block.merge_next();
      } else {
        int next_store = get_store_index(next);
        if (next_store == store_idx) {
          free_stores[next_store].remove(next);
          block.merge_next();
        }
      }
    }
    block.mark_free(store_idx);
    free_stores[store_idx].insert(block);
  }

  LIBC_INLINE void rotate() {
    unsigned prev_active = active;
    active = 1 - active;
    while (BlockRef block = free_stores[prev_active].remove_any())
      coalesce_and_insert(block, active);
  }
#endif

  cpp::byte *begin;
  cpp::byte *end;
  bool is_initialized = false;
#ifdef LIBC_COPT_BAREMETAL_HEAP_ENABLE_FREESTORE_ROTATION
  FreeStore free_stores[2];
  unsigned active = 0;
  LIBC_INLINE FreeStore &active_free_store() { return free_stores[active]; }
#else
  FreeStore free_store;
  LIBC_INLINE FreeStore &active_free_store() { return free_store; }
#endif
};

template <size_t BUFF_SIZE> class FreeListHeapBuffer : public FreeListHeap {
public:
  LIBC_INLINE constexpr FreeListHeapBuffer() : FreeListHeap{buffer}, buffer{} {}

private:
  cpp::byte buffer[BUFF_SIZE];
};

LIBC_INLINE void FreeListHeap::init() {
  LIBC_ASSERT(!is_initialized && "duplicate initialization");
  auto result = BlockRef::init(region());
  BlockRef block = *result;
#ifdef LIBC_COPT_BAREMETAL_HEAP_ENABLE_FREESTORE_ROTATION
  free_stores[0].set_range({0, cpp::bit_ceil(block.inner_size())});
  free_stores[1].set_range({0, cpp::bit_ceil(block.inner_size())});
  coalesce_and_insert(block, active);
#else
  free_store.set_range({0, cpp::bit_ceil(block.inner_size())});
  free_store.insert(block);
#endif
  is_initialized = true;
}

LIBC_INLINE void *FreeListHeap::allocate_impl(size_t alignment, size_t size) {
  if (size == 0)
    return nullptr;

  if (!is_initialized)
    init();

  size_t request_size = BlockRef::min_size_for_allocation(alignment, size);
  if (!request_size)
    return nullptr;

  BlockRef block = active_free_store().remove_best_fit(request_size);
#ifdef LIBC_COPT_BAREMETAL_HEAP_ENABLE_FREESTORE_ROTATION
  if (!block) {
    rotate();
    block = active_free_store().remove_best_fit(request_size);
  }
#endif
  if (!block)
    return nullptr;

  auto block_info = BlockRef::allocate(block, alignment, size);
  block_info.block.mark_used();
#ifdef LIBC_COPT_BAREMETAL_HEAP_ENABLE_FREESTORE_ROTATION
  if (block_info.next)
    coalesce_and_insert(block_info.next, active);
  if (block_info.prev)
    coalesce_and_insert(block_info.prev, active);
#else
  if (block_info.next)
    free_store.insert(block_info.next);
  if (block_info.prev)
    free_store.insert(block_info.prev);
#endif
  return block_info.block.usable_space();
}

LIBC_INLINE void *FreeListHeap::allocate(size_t size) {
  return allocate_impl(BlockRef::MIN_ALIGN, size);
}

LIBC_INLINE void *FreeListHeap::aligned_allocate(size_t alignment,
                                                 size_t size) {
  // The alignment must be an integral power of two.
  if (!IsPow2(alignment))
    return nullptr;

  // The size parameter must be an integral multiple of alignment.
  if (size % alignment != 0)
    return nullptr;

  // The minimum alignment supported by BlockRef is MIN_ALIGN.
  alignment = cpp::max(alignment, BlockRef::MIN_ALIGN);

  return allocate_impl(alignment, size);
}

LIBC_INLINE void FreeListHeap::free(void *ptr) {
  if (ptr == nullptr)
    return;

  cpp::byte *bytes = static_cast<cpp::byte *>(ptr);

  LIBC_ASSERT(is_valid_ptr(bytes) && "Invalid pointer");

  BlockRef block = BlockRef::from_usable_space(bytes);
  LIBC_ASSERT(block.next() && "sentinel last block cannot be freed");
  LIBC_ASSERT(block.used() && "double free");
#ifdef LIBC_COPT_BAREMETAL_HEAP_ENABLE_FREESTORE_ROTATION
  coalesce_and_insert(block, 1 - active);
#else
  block.mark_free();

  // Can we combine with the left or right blocks?
  BlockRef prev_free = block.prev_free();
  BlockRef next = block.next();

  if (prev_free) {
    // Remove from free store and merge.
    free_store.remove(prev_free);
    block = prev_free;
    block.merge_next();
  }
  if (!next.used()) {
    free_store.remove(next);
    block.merge_next();
  }
  // Add back to the freelist
  free_store.insert(block);
#endif
}

LIBC_INLINE size_t FreeListHeap::allocation_size(const void *ptr) const {
  if (!is_valid_ptr(ptr))
    return 0;
  BlockRef block = BlockRef::from_usable_space(ptr);
  if (!block.used())
    return 0;
  return block.inner_size();
}

LIBC_INLINE bool FreeListHeap::shrink_in_place(BlockRef block, size_t size) {
  size_t min_outer_size = BlockRef::outer_size(cpp::max(size, sizeof(size_t)));
  uintptr_t next_block_start = BlockRef::next_possible_block_start(
      block.addr() + min_outer_size, BlockRef::MIN_ALIGN);
  size_t new_outer_size = next_block_start - block.addr();
  if (block.outer_size() >= new_outer_size) {
    optional<BlockRef> next = block.split(size);
    // register the new block on successful split
    if (next.has_value()) {
      BlockRef next_block = *next;
#ifdef LIBC_COPT_BAREMETAL_HEAP_ENABLE_FREESTORE_ROTATION
      coalesce_and_insert(next_block, 1 - active);
#else
      BlockRef right = next_block.next();
      // Since the original block was not the last block (the sentinel last
      // block is never split), the split-off remainder block `next_block` is
      // also not the last block. Thus, its next block `right` is guaranteed
      // to be non-null.
      LIBC_ASSERT(right && "right block must be non-null");
      if (!right.used()) {
        free_store.remove(right);
        next_block.merge_next();
      }
      free_store.insert(next_block);
#endif
    }
    return true;
  }
  return false;
}

// Follows constract of the C standard realloc() function
// If ptr is free'd, will return nullptr.
LIBC_INLINE void *FreeListHeap::realloc(void *ptr, size_t size) {
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

  BlockRef block = BlockRef::from_usable_space(bytes);
  if (!block.used())
    return nullptr;
  size_t old_size = block.inner_size();

  if (old_size >= size) {
    shrink_in_place(block, size);
    return ptr;
  }

  void *new_ptr = allocate(size);
  // Don't invalidate ptr if allocate(size) fails to initilize the memory.
  if (new_ptr == nullptr)
    return nullptr;
  LIBC_NAMESPACE::inline_memcpy(new_ptr, ptr, old_size);

  free(ptr);
  return new_ptr;
}

LIBC_INLINE void *FreeListHeap::calloc(size_t num, size_t size) {
  size_t bytes;
  if (__builtin_mul_overflow(num, size, &bytes))
    return nullptr;
  void *ptr = allocate(bytes);
  if (ptr != nullptr)
    LIBC_NAMESPACE::inline_memset(ptr, 0, bytes);
  return ptr;
}

extern FreeListHeap *freelist_heap;

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_FREELIST_HEAP_H

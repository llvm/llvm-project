//===-- sanitizer_stack_store.cpp -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A fast memory allocator that does not support free() nor realloc().
// All allocations are forever.
//===----------------------------------------------------------------------===//

#include "sanitizer_stack_store.h"

#include "sanitizer_atomic.h"
#include "sanitizer_common.h"
namespace __sanitizer {

uptr *StackStore::tryAlloc(uptr count) {
  // Optimisic lock-free allocation, essentially try to bump the region ptr.
  for (;;) {
    uptr cmp = atomic_load(&region_pos, memory_order_acquire);
    uptr end = atomic_load(&region_end, memory_order_acquire);
    uptr size = count * sizeof(uptr);
    if (cmp == 0 || cmp + size > end)
      return nullptr;
    if (atomic_compare_exchange_weak(&region_pos, &cmp, cmp + size,
                                     memory_order_acquire))
      return reinterpret_cast<uptr *>(cmp);
  }
}

uptr *StackStore::alloc(uptr count) {
  // First, try to allocate optimisitically.
  uptr *s = tryAlloc(count);
  if (LIKELY(s))
    return s;
  return refillAndAlloc(count);
}

uptr *StackStore::refillAndAlloc(uptr count) {
  // If failed, lock, retry and alloc new superblock.
  SpinMutexLock l(&mtx);
  for (;;) {
    uptr *s = tryAlloc(count);
    if (s)
      return s;
    atomic_store(&region_pos, 0, memory_order_relaxed);
    uptr size = count * sizeof(uptr) + sizeof(BlockInfo);
    uptr allocsz = RoundUpTo(Max<uptr>(size, 64u * 1024u), GetPageSizeCached());
    uptr mem = (uptr)MmapOrDie(allocsz, "stack depot");
    BlockInfo *new_block = (BlockInfo *)(mem + allocsz) - 1;
    new_block->next = curr;
    new_block->ptr = mem;
    new_block->size = allocsz;
    curr = new_block;

    atomic_fetch_add(&mapped_size, allocsz, memory_order_relaxed);

    allocsz -= sizeof(BlockInfo);
    atomic_store(&region_end, mem + allocsz, memory_order_release);
    atomic_store(&region_pos, mem, memory_order_release);
  }
}

void StackStore::TestOnlyUnmap() {
  while (curr) {
    uptr mem = curr->ptr;
    uptr allocsz = curr->size;
    curr = curr->next;
    UnmapOrDie((void *)mem, allocsz);
  }
  internal_memset(this, 0, sizeof(*this));
}

}  // namespace __sanitizer

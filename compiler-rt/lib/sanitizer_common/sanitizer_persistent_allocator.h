//===-- sanitizer_persistent_allocator.h ------------------------*- C++ -*-===//
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

#ifndef SANITIZER_PERSISTENT_ALLOCATOR_H
#define SANITIZER_PERSISTENT_ALLOCATOR_H

#include "sanitizer_internal_defs.h"
#include "sanitizer_mutex.h"
#include "sanitizer_atomic.h"
#include "sanitizer_common.h"

namespace __sanitizer {

template <typename T>
class PersistentAllocator {
 public:
  T *alloc(uptr count = 1);
  uptr allocated() const { return atomic_load_relaxed(&mapped_size); }

  void TestOnlyUnmap();

 private:
  T *tryAlloc(uptr count);
  T *refillAndAlloc(uptr count);
  mutable StaticSpinMutex mtx;  // Protects alloc of new blocks.
  atomic_uintptr_t region_pos;  // Region allocator for Node's.
  atomic_uintptr_t region_end;
  atomic_uintptr_t mapped_size;

  struct BlockInfo {
    const BlockInfo *next;
    uptr ptr;
    uptr size;
  };
  const BlockInfo *curr;
};

template <typename T>
inline T *PersistentAllocator<T>::tryAlloc(uptr count) {
  // Optimisic lock-free allocation, essentially try to bump the region ptr.
  for (;;) {
    uptr cmp = atomic_load(&region_pos, memory_order_acquire);
    uptr end = atomic_load(&region_end, memory_order_acquire);
    uptr size = count * sizeof(T);
    if (cmp == 0 || cmp + size > end)
      return nullptr;
    if (atomic_compare_exchange_weak(&region_pos, &cmp, cmp + size,
                                     memory_order_acquire))
      return reinterpret_cast<T *>(cmp);
  }
}

template <typename T>
inline T *PersistentAllocator<T>::alloc(uptr count) {
  // First, try to allocate optimisitically.
  T *s = tryAlloc(count);
  if (LIKELY(s))
    return s;
  return refillAndAlloc(count);
}

template <typename T>
inline T *PersistentAllocator<T>::refillAndAlloc(uptr count) {
  // If failed, lock, retry and alloc new superblock.
  SpinMutexLock l(&mtx);
  for (;;) {
    T *s = tryAlloc(count);
    if (s)
      return s;
    atomic_store(&region_pos, 0, memory_order_relaxed);
    uptr size = count * sizeof(T) + sizeof(BlockInfo);
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

template <typename T>
void PersistentAllocator<T>::TestOnlyUnmap() {
  while (curr) {
    uptr mem = curr->ptr;
    uptr allocsz = curr->size;
    curr = curr->next;
    UnmapOrDie((void *)mem, allocsz);
  }
  internal_memset(this, 0, sizeof(*this));
}

} // namespace __sanitizer

#endif // SANITIZER_PERSISTENT_ALLOCATOR_H

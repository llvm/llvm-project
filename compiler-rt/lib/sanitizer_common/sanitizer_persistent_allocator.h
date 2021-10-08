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

class PersistentAllocator {
 public:
  void *alloc(uptr size);
  uptr allocated() const {
    SpinMutexLock l(&mtx);
    return atomic_load_relaxed(&mapped_size) +
           atomic_load_relaxed(&region_pos) - atomic_load_relaxed(&region_end);
  }

 private:
  void *tryAlloc(uptr size);
  void *refillAndAlloc(uptr size);
  mutable StaticSpinMutex mtx;  // Protects alloc of new blocks.
  atomic_uintptr_t region_pos;  // Region allocator for Node's.
  atomic_uintptr_t region_end;
  atomic_uintptr_t mapped_size;
};

inline void *PersistentAllocator::tryAlloc(uptr size) {
  // Optimisic lock-free allocation, essentially try to bump the region ptr.
  for (;;) {
    uptr cmp = atomic_load(&region_pos, memory_order_acquire);
    uptr end = atomic_load(&region_end, memory_order_acquire);
    if (cmp == 0 || cmp + size > end) return nullptr;
    if (atomic_compare_exchange_weak(&region_pos, &cmp, cmp + size,
                                     memory_order_acquire))
      return (void *)cmp;
  }
}

inline void *PersistentAllocator::alloc(uptr size) {
  // First, try to allocate optimisitically.
  void *s = tryAlloc(size);
  if (LIKELY(s))
    return s;
  return refillAndAlloc(size);
}

inline void *PersistentAllocator::refillAndAlloc(uptr size) {
  // If failed, lock, retry and alloc new superblock.
  SpinMutexLock l(&mtx);
  for (;;) {
    void *s = tryAlloc(size);
    if (s)
      return s;
    atomic_store(&region_pos, 0, memory_order_relaxed);
    uptr allocsz = 64 * 1024;
    if (allocsz < size)
      allocsz = size;
    uptr mem = (uptr)MmapOrDie(allocsz, "stack depot");
    atomic_fetch_add(&mapped_size, allocsz, memory_order_relaxed);
    atomic_store(&region_end, mem + allocsz, memory_order_release);
    atomic_store(&region_pos, mem, memory_order_release);
  }
}

} // namespace __sanitizer

#endif // SANITIZER_PERSISTENT_ALLOCATOR_H

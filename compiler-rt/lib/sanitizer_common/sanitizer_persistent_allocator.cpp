//===-- sanitizer_persistent_allocator.cpp ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries.
//===----------------------------------------------------------------------===//
#include "sanitizer_persistent_allocator.h"

namespace __sanitizer {

void *PersistentAllocator::refillAndAlloc(uptr size) {
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

}  // namespace __sanitizer

//===-- sanitizer_stack_store.cpp -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "sanitizer_stack_store.h"

#include "sanitizer_atomic.h"
#include "sanitizer_common.h"
#include "sanitizer_stacktrace.h"

namespace __sanitizer {

static constexpr u32 kStackSizeBits = 16;

StackStore::Id StackStore::Store(const StackTrace &trace) {
  uptr *stack_trace = Alloc(trace.size + 1);
  CHECK_LT(trace.size, 1 << kStackSizeBits);
  *stack_trace = trace.size + (trace.tag << kStackSizeBits);
  internal_memcpy(stack_trace + 1, trace.trace, trace.size * sizeof(uptr));
  return reinterpret_cast<StackStore::Id>(stack_trace);
}

StackTrace StackStore::Load(Id id) {
  const uptr *stack_trace = reinterpret_cast<const uptr *>(id);
  uptr size = *stack_trace & ((1 << kStackSizeBits) - 1);
  uptr tag = *stack_trace >> kStackSizeBits;
  return StackTrace(stack_trace + 1, size, tag);
}

uptr *StackStore::TryAlloc(uptr count) {
  // Optimisic lock-free allocation, essentially try to bump the region ptr.
  for (;;) {
    uptr cmp = atomic_load(&region_pos_, memory_order_acquire);
    uptr end = atomic_load(&region_end_, memory_order_acquire);
    uptr size = count * sizeof(uptr);
    if (cmp == 0 || cmp + size > end)
      return nullptr;
    if (atomic_compare_exchange_weak(&region_pos_, &cmp, cmp + size,
                                     memory_order_acquire))
      return reinterpret_cast<uptr *>(cmp);
  }
}

uptr *StackStore::Alloc(uptr count) {
  // First, try to allocate optimisitically.
  uptr *s = TryAlloc(count);
  if (LIKELY(s))
    return s;
  return RefillAndAlloc(count);
}

uptr *StackStore::RefillAndAlloc(uptr count) {
  // If failed, lock, retry and alloc new superblock.
  SpinMutexLock l(&mtx_);
  for (;;) {
    uptr *s = TryAlloc(count);
    if (s)
      return s;
    atomic_store(&region_pos_, 0, memory_order_relaxed);
    uptr size = count * sizeof(uptr) + sizeof(BlockInfo);
    uptr allocsz = RoundUpTo(Max<uptr>(size, 64u * 1024u), GetPageSizeCached());
    uptr mem = (uptr)MmapOrDie(allocsz, "stack depot");
    BlockInfo *new_block = (BlockInfo *)(mem + allocsz) - 1;
    new_block->next = curr_;
    new_block->ptr = mem;
    new_block->size = allocsz;
    curr_ = new_block;

    atomic_fetch_add(&mapped_size_, allocsz, memory_order_relaxed);

    allocsz -= sizeof(BlockInfo);
    atomic_store(&region_end_, mem + allocsz, memory_order_release);
    atomic_store(&region_pos_, mem, memory_order_release);
  }
}

void StackStore::TestOnlyUnmap() {
  while (curr_) {
    uptr mem = curr_->ptr;
    uptr allocsz = curr_->size;
    curr_ = curr_->next;
    UnmapOrDie((void *)mem, allocsz);
  }
  internal_memset(this, 0, sizeof(*this));
}

}  // namespace __sanitizer

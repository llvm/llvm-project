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

namespace {
struct StackTraceHeader {
  static constexpr u32 kStackSizeBits = 8;

  u8 size;
  u8 tag;
  explicit StackTraceHeader(const StackTrace &trace)
      : size(Min<uptr>(trace.size, (1u << 8) - 1)), tag(trace.tag) {
    CHECK_EQ(trace.tag, static_cast<uptr>(tag));
  }
  explicit StackTraceHeader(uptr h)
      : size(h & ((1 << kStackSizeBits) - 1)), tag(h >> kStackSizeBits) {}

  uptr ToUptr() const {
    return static_cast<uptr>(size) | (static_cast<uptr>(tag) << kStackSizeBits);
  }
};
}  // namespace

StackStore::Id StackStore::Store(const StackTrace &trace) {
  if (!trace.size && !trace.tag)
    return 0;
  StackTraceHeader h(trace);
  uptr *stack_trace = Alloc(h.size + 1);
  *stack_trace = h.ToUptr();
  internal_memcpy(stack_trace + 1, trace.trace, h.size * sizeof(uptr));
  return reinterpret_cast<StackStore::Id>(stack_trace);
}

StackTrace StackStore::Load(Id id) const {
  if (!id)
    return {};
  const uptr *stack_trace = reinterpret_cast<const uptr *>(id);
  StackTraceHeader h(*stack_trace);
  return StackTrace(stack_trace + 1, h.size, h.tag);
}

uptr StackStore::Allocated() const {
  return atomic_load_relaxed(&mapped_size_);
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

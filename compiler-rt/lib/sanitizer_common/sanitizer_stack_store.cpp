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
  uptr idx;
  uptr *stack_trace = Alloc(h.size + 1, &idx);
  *stack_trace = h.ToUptr();
  internal_memcpy(stack_trace + 1, trace.trace, h.size * sizeof(uptr));
  return OffsetToId(idx);
}

StackTrace StackStore::Load(Id id) const {
  if (!id)
    return {};
  uptr idx = IdToOffset(id);
  uptr block_idx = GetBlockIdx(idx);
  CHECK_LT(block_idx, ARRAY_SIZE(blocks_));
  const uptr *stack_trace = blocks_[block_idx].Get();
  if (!stack_trace)
    return {};
  stack_trace += GetInBlockIdx(idx);
  StackTraceHeader h(*stack_trace);
  return StackTrace(stack_trace + 1, h.size, h.tag);
}

uptr StackStore::Allocated() const {
  return RoundUpTo(atomic_load_relaxed(&total_frames_) * sizeof(uptr),
                   GetPageSizeCached()) +
         sizeof(*this);
}

uptr *StackStore::Alloc(uptr count, uptr *idx) {
  for (;;) {
    // Optimisic lock-free allocation, essentially try to bump the
    // total_frames_.
    uptr start = atomic_fetch_add(&total_frames_, count, memory_order_relaxed);
    uptr block_idx = GetBlockIdx(start);
    if (LIKELY(block_idx == GetBlockIdx(start + count - 1))) {
      // Fits into the a single block.
      CHECK_LT(block_idx, ARRAY_SIZE(blocks_));
      *idx = start;
      return blocks_[block_idx].GetOrCreate() + GetInBlockIdx(start);
    }

    // Retry. We can't use range allocated in two different blocks.
  }
}

void StackStore::TestOnlyUnmap() {
  for (BlockInfo &b : blocks_) b.TestOnlyUnmap();
  internal_memset(this, 0, sizeof(*this));
}

uptr *StackStore::BlockInfo::Get() const {
  // Idiomatic double-checked locking uses memory_order_acquire here. But
  // relaxed is find for us, justification is similar to
  // TwoLevelMap::GetOrCreate.
  return reinterpret_cast<uptr *>(atomic_load_relaxed(&data_));
}

uptr *StackStore::BlockInfo::Create() {
  SpinMutexLock l(&mtx_);
  uptr *ptr = Get();
  if (!ptr) {
    ptr = reinterpret_cast<uptr *>(
        MmapNoReserveOrDie(kBlockSizeBytes, "StackStore"));
    atomic_store(&data_, reinterpret_cast<uptr>(ptr), memory_order_release);
  }
  return ptr;
}

uptr *StackStore::BlockInfo::GetOrCreate() {
  uptr *ptr = Get();
  if (LIKELY(ptr))
    return ptr;
  return Create();
}

void StackStore::BlockInfo::TestOnlyUnmap() {
  if (uptr *ptr = Get())
    UnmapOrDie(ptr, StackStore::kBlockSizeBytes);
}

}  // namespace __sanitizer

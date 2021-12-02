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

StackStore::Id StackStore::Store(const StackTrace &trace, uptr *pack) {
  if (!trace.size && !trace.tag)
    return 0;
  StackTraceHeader h(trace);
  uptr idx = 0;
  *pack = 0;
  uptr *stack_trace = Alloc(h.size + 1, &idx, pack);
  *stack_trace = h.ToUptr();
  internal_memcpy(stack_trace + 1, trace.trace, h.size * sizeof(uptr));
  *pack += blocks_[GetBlockIdx(idx)].Stored(h.size + 1);
  return OffsetToId(idx);
}

StackTrace StackStore::Load(Id id) {
  if (!id)
    return {};
  uptr idx = IdToOffset(id);
  uptr block_idx = GetBlockIdx(idx);
  CHECK_LT(block_idx, ARRAY_SIZE(blocks_));
  const uptr *stack_trace = blocks_[block_idx].GetOrUnpack();
  if (!stack_trace)
    return {};
  stack_trace += GetInBlockIdx(idx);
  StackTraceHeader h(*stack_trace);
  return StackTrace(stack_trace + 1, h.size, h.tag);
}

uptr StackStore::Allocated() const {
  uptr next_block = GetBlockIdx(
      RoundUpTo(atomic_load_relaxed(&total_frames_), kBlockSizeFrames));
  uptr res = 0;
  for (uptr i = 0; i < next_block; ++i) res += blocks_[i].Allocated();
  return res + sizeof(*this);
}

uptr *StackStore::Alloc(uptr count, uptr *idx, uptr *pack) {
  for (;;) {
    // Optimisic lock-free allocation, essentially try to bump the
    // total_frames_.
    uptr start = atomic_fetch_add(&total_frames_, count, memory_order_relaxed);
    uptr block_idx = GetBlockIdx(start);
    uptr last_idx = GetBlockIdx(start + count - 1);
    if (LIKELY(block_idx == last_idx)) {
      // Fits into the a single block.
      CHECK_LT(block_idx, ARRAY_SIZE(blocks_));
      *idx = start;
      return blocks_[block_idx].GetOrCreate() + GetInBlockIdx(start);
    }

    // Retry. We can't use range allocated in two different blocks.
    CHECK_LE(count, kBlockSizeFrames);
    uptr in_first = kBlockSizeFrames - GetInBlockIdx(start);
    // Mark tail/head of these blocks as "stored".to avoid waiting before we can
    // Pack().
    *pack += blocks_[block_idx].Stored(in_first);
    *pack += blocks_[last_idx].Stored(count - in_first);
  }
}

uptr StackStore::Pack(Compression type) {
  uptr res = 0;
  for (BlockInfo &b : blocks_) res += b.Pack(type);
  return res;
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

uptr *StackStore::BlockInfo::GetOrUnpack() {
  SpinMutexLock l(&mtx_);
  switch (state) {
    case State::Storing:
      state = State::Unpacked;
      FALLTHROUGH;
    case State::Unpacked:
      return Get();
    case State::Packed:
      break;
  }

  uptr *ptr = Get();
  CHECK_NE(nullptr, ptr);
  // Fake unpacking.
  for (uptr i = 0; i < kBlockSizeFrames; ++i) ptr[i] = ~ptr[i];
  state = State::Unpacked;
  return Get();
}

uptr StackStore::BlockInfo::Pack(Compression type) {
  if (type == Compression::None)
    return 0;

  SpinMutexLock l(&mtx_);
  switch (state) {
    case State::Unpacked:
    case State::Packed:
      return 0;
    case State::Storing:
      break;
  }

  uptr *ptr = Get();
  if (!ptr || !Stored(0))
    return 0;

  // Fake packing.
  for (uptr i = 0; i < kBlockSizeFrames; ++i) ptr[i] = ~ptr[i];
  state = State::Packed;
  return kBlockSizeBytes - kBlockSizeBytes / 10;
}

uptr StackStore::BlockInfo::Allocated() const {
  SpinMutexLock l(&mtx_);
  switch (state) {
    case State::Packed:
      return kBlockSizeBytes / 10;
    case State::Unpacked:
    case State::Storing:
      return kBlockSizeBytes;
  }
}

void StackStore::BlockInfo::TestOnlyUnmap() {
  if (uptr *ptr = Get())
    UnmapOrDie(ptr, StackStore::kBlockSizeBytes);
}

bool StackStore::BlockInfo::Stored(uptr n) {
  return n + atomic_fetch_add(&stored_, n, memory_order_release) ==
         kBlockSizeFrames;
}

bool StackStore::BlockInfo::IsPacked() const {
  SpinMutexLock l(&mtx_);
  return state == State::Packed;
}

}  // namespace __sanitizer

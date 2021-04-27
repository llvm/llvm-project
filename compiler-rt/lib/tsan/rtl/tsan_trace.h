//===-- tsan_trace.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
//===----------------------------------------------------------------------===//
#ifndef TSAN_TRACE_H
#define TSAN_TRACE_H

#include "tsan_defs.h"
#include "tsan_ilist.h"
#include "tsan_mutex.h"
#include "tsan_mutexset.h"
#include "tsan_stack_trace.h"

namespace __tsan {

enum EventType {
  EventTypeAccessExt,
  EventTypeAccessRange,
  EventTypeLock,
  EventTypeRLock,
  EventTypeUnlock,
  EventTypeTime,
};

struct Event {
  u64 is_access : 1;
  u64 is_func : 1;
  u64 type : 3;
  u64 _ : 59;
};
static_assert(sizeof(Event) == 8, "bad Event size");
static constexpr Event NopEvent = {1, 0, 0, 0};

constexpr uptr kCompressedAddrBits = 44;

struct EventAccess {
  static constexpr uptr kPCBits = 15;

  u64 is_access : 1;
  u64 isRead : 1;
  u64 isAtomic : 1;
  u64 sizeLog : 2;
  u64 pcDelta : kPCBits;
  u64 addr : kCompressedAddrBits;
};
static_assert(sizeof(EventAccess) == 8, "bad EventAccess size");

struct EventFunc {
  u64 is_access : 1;
  u64 is_func : 1;
  u64 pc : 62;
};
static_assert(sizeof(EventFunc) == 8, "bad EventFunc size");

struct EventAccessExt {
  u64 is_access : 1;
  u64 is_func : 1;
  u64 type : 3;
  u64 isRead : 1;
  u64 isAtomic : 1;
  u64 sizeLog : 2;
  u64 _ : 11;
  u64 addr : kCompressedAddrBits;
  u64 pc;
};
static_assert(sizeof(EventAccessExt) == 16, "bad EventAccessExt size");

struct EventAccessRange {
  static constexpr uptr kSizeLoBits = 13;

  u64 is_access : 1;
  u64 is_func : 1;
  u64 type : 3;
  u64 isRead : 1;
  u64 isFreed : 1;
  u64 sizeLo : kSizeLoBits;
  u64 pc : kCompressedAddrBits;
  u64 addr : kCompressedAddrBits;
  u64 sizeHi : 64 - kCompressedAddrBits;
};
static_assert(sizeof(EventAccessRange) == 16, "bad EventAccessRange size");

struct EventLock {
  static constexpr uptr kStackIDLoBits = 15;

  u64 is_access : 1;
  u64 is_func : 1;
  u64 type : 3;
  u64 pc : kCompressedAddrBits;
  u64 stackIDLo : kStackIDLoBits;
  u64 stackIDHi : sizeof(StackID) * kByteBits - kStackIDLoBits;
  u64 _ : 3;
  u64 addr : kCompressedAddrBits;
};
static_assert(sizeof(EventLock) == 16, "bad EventLock size");

struct EventUnlock {
  u64 is_access : 1;
  u64 is_func : 1;
  u64 type : 3;
  u64 _ : 15;
  u64 addr : kCompressedAddrBits;
};
static_assert(sizeof(EventUnlock) == 8, "bad EventUnlock size");

struct EventTime {
  u64 is_access : 1;
  u64 is_func : 1;
  u64 type : 3;
  u64 sid : 8;
  u64 epoch : kEpochBits;
  u64 _ : 64 - 13 - kEpochBits;
};
static_assert(sizeof(EventTime) == 8, "bad EventTime size");

struct TraceHeader {
  Trace* trace = nullptr;
  INode trace_parts; // in Trace::parts
  INode global; // in Contex::trace_part_recycle
  VarSizeStackTrace start_stack;
  MutexSet start_mset;
  uptr prev_pc = 0;
#if !SANITIZER_GO
  BufferedStackTrace stack0;  // Start stack for the trace.
#else
  VarSizeStackTrace stack0;
#endif
};

struct TracePart : TraceHeader {
  static constexpr uptr kByteSize = 256 << 10;
  static constexpr uptr kSize =
      (kByteSize - sizeof(TraceHeader)) / sizeof(Event);
  // Note: TracePos assumes this to be the last field.
  Event events[kSize];

  TracePart() {
  }
};
static_assert(sizeof(TracePart) == TracePart::kByteSize, "bad TracePart size");

struct Trace {
  Mutex mtx;
  IList<TraceHeader, &TraceHeader::trace_parts, TracePart> parts;
  TracePart* local_head; // first node non-queued into ctx->trace_part_recycle
  Event* final_pos = nullptr;
  uptr parts_allocated = 0;

  Trace() : mtx(MutexTypeTrace) {
  }
};

}  // namespace __tsan

#endif  // TSAN_TRACE_H

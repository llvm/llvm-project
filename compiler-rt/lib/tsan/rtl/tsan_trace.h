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
#include "tsan_mutexset.h"
#include "tsan_stack_trace.h"

namespace __tsan {

const int kTracePartSizeBits = 13;
const int kTracePartSize = 1 << kTracePartSizeBits;
const int kTraceParts = 2 * 1024 * 1024 / kTracePartSize;
const int kTraceSize = kTracePartSize * kTraceParts;

// Must fit into 3 bits.
enum EventType {
  EventTypeMop,
  EventTypeFuncEnter,
  EventTypeFuncExit,
  EventTypeLock,
  EventTypeUnlock,
  EventTypeRLock,
  EventTypeRUnlock
};

// Represents a thread event (from most significant bit):
// u64 typ  : 3;   // EventType.
// u64 addr : 61;  // Associated pc.
typedef u64 Event;

const uptr kEventPCBits = 61;

struct TraceHeader {
#if !SANITIZER_GO
  BufferedStackTrace stack0;  // Start stack for the trace.
#else
  VarSizeStackTrace stack0;
#endif
  u64        epoch0;  // Start epoch for the trace.
  MutexSet   mset0;

  TraceHeader() : stack0(), epoch0() {}
};

struct Trace {
  Mutex mtx;
#if !SANITIZER_GO
  // Must be last to catch overflow as paging fault.
  // Go shadow stack is dynamically allocated.
  uptr shadow_stack[kShadowStackSize];
#endif
  // Must be the last field, because we unmap the unused part in
  // CreateThreadContext.
  TraceHeader headers[kTraceParts];

  Trace() : mtx(MutexTypeTrace) {}
};

namespace v3 {

enum class EventType : u64 {
  kAccessExt,
  kAccessRange,
  kLock,
  kRLock,
  kUnlock,
  kTime,
};

// "Base" type for all events for type dispatch.
struct Event {
  // We use variable-length type encoding to give more bits to some event
  // types that need them. If is_access is set, this is EventAccess.
  // Otherwise, if is_func is set, this is EventFunc.
  // Otherwise type denotes the type.
  u64 is_access : 1;
  u64 is_func : 1;
  EventType type : 3;
  u64 _ : 59;
};
static_assert(sizeof(Event) == 8, "bad Event size");

// Nop event used as padding and does not affect state during replay.
static constexpr Event NopEvent = {1, 0, EventType::kAccessExt, 0};

// Compressed memory access can represent only some events with PCs
// close enough to each other. Otherwise we fall back to EventAccessExt.
struct EventAccess {
  static constexpr uptr kPCBits = 15;

  u64 is_access : 1;  // = 1
  u64 is_read : 1;
  u64 is_atomic : 1;
  u64 size_log : 2;
  u64 pc_delta : kPCBits;  // signed delta from the previous memory access PC
  u64 addr : kCompressedAddrBits;
};
static_assert(sizeof(EventAccess) == 8, "bad EventAccess size");

// Function entry (pc != 0) or exit (pc == 0).
struct EventFunc {
  u64 is_access : 1;  // = 0
  u64 is_func : 1;    // = 1
  u64 pc : 62;
};
static_assert(sizeof(EventFunc) == 8, "bad EventFunc size");

// Extended memory access with full PC.
struct EventAccessExt {
  u64 is_access : 1;   // = 0
  u64 is_func : 1;     // = 0
  EventType type : 3;  // = EventType::kAccessExt
  u64 is_read : 1;
  u64 is_atomic : 1;
  u64 size_log : 2;
  u64 _ : 11;
  u64 addr : kCompressedAddrBits;
  u64 pc;
};
static_assert(sizeof(EventAccessExt) == 16, "bad EventAccessExt size");

// Access to a memory range.
struct EventAccessRange {
  static constexpr uptr kSizeLoBits = 13;

  u64 is_access : 1;   // = 0
  u64 is_func : 1;     // = 0
  EventType type : 3;  // = EventType::kAccessRange
  u64 is_read : 1;
  u64 is_free : 1;
  u64 size_lo : kSizeLoBits;
  u64 pc : kCompressedAddrBits;
  u64 addr : kCompressedAddrBits;
  u64 size_hi : 64 - kCompressedAddrBits;
};
static_assert(sizeof(EventAccessRange) == 16, "bad EventAccessRange size");

// Mutex lock.
struct EventLock {
  static constexpr uptr kStackIDLoBits = 15;

  u64 is_access : 1;   // = 0
  u64 is_func : 1;     // = 0
  EventType type : 3;  // = EventType::kLock or EventType::kRLock
  u64 pc : kCompressedAddrBits;
  u64 stack_lo : kStackIDLoBits;
  u64 stack_hi : sizeof(StackID) * kByteBits - kStackIDLoBits;
  u64 _ : 3;
  u64 addr : kCompressedAddrBits;
};
static_assert(sizeof(EventLock) == 16, "bad EventLock size");

// Mutex unlock.
struct EventUnlock {
  u64 is_access : 1;   // = 0
  u64 is_func : 1;     // = 0
  EventType type : 3;  // = EventType::kUnlock
  u64 _ : 15;
  u64 addr : kCompressedAddrBits;
};
static_assert(sizeof(EventUnlock) == 8, "bad EventUnlock size");

// Time change event.
struct EventTime {
  u64 is_access : 1;   // = 0
  u64 is_func : 1;     // = 0
  EventType type : 3;  // = EventType::kTime
  u64 sid : sizeof(Sid) * kByteBits;
  u64 epoch : kEpochBits;
  u64 _ : 64 - 5 - sizeof(Sid) * kByteBits - kEpochBits;
};
static_assert(sizeof(EventTime) == 8, "bad EventTime size");

struct Trace;

struct TraceHeader {
  Trace* trace = nullptr;  // back-pointer to Trace containing this part
  INode trace_parts;       // in Trace::parts
};

struct TracePart : TraceHeader {
  static constexpr uptr kByteSize = 256 << 10;
  static constexpr uptr kSize =
      (kByteSize - sizeof(TraceHeader)) / sizeof(Event);
  // TraceAcquire does a fast event pointer overflow check by comparing
  // pointer into TracePart::events with kAlignment mask. Since TracePart's
  // are allocated page-aligned, this check detects end of the array
  // (it also have false positives in the middle that are filtered separately).
  // This also requires events to be the last field.
  static constexpr uptr kAlignment = 0xff0;
  Event events[kSize];

  TracePart() {}
};
static_assert(sizeof(TracePart) == TracePart::kByteSize, "bad TracePart size");

struct Trace {
  Mutex mtx;
  IList<TraceHeader, &TraceHeader::trace_parts, TracePart> parts;
  Event* final_pos =
      nullptr;  // final position in the last part for finished threads

  Trace() : mtx(MutexTypeTrace) {}
};

}  // namespace v3

}  // namespace __tsan

#endif  // TSAN_TRACE_H

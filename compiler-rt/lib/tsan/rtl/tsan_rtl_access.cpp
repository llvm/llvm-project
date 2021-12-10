//===-- tsan_rtl_access.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
// Definitions of memory access and function entry/exit entry points.
//===----------------------------------------------------------------------===//

#include "tsan_rtl.h"

namespace __tsan {

namespace v3 {

ALWAYS_INLINE USED bool TryTraceMemoryAccess(ThreadState *thr, uptr pc,
                                             uptr addr, uptr size,
                                             AccessType typ) {
  DCHECK(size == 1 || size == 2 || size == 4 || size == 8);
  if (!kCollectHistory)
    return true;
  EventAccess *ev;
  if (UNLIKELY(!TraceAcquire(thr, &ev)))
    return false;
  u64 size_log = size == 1 ? 0 : size == 2 ? 1 : size == 4 ? 2 : 3;
  uptr pc_delta = pc - thr->trace_prev_pc + (1 << (EventAccess::kPCBits - 1));
  thr->trace_prev_pc = pc;
  if (LIKELY(pc_delta < (1 << EventAccess::kPCBits))) {
    ev->is_access = 1;
    ev->is_read = !!(typ & kAccessRead);
    ev->is_atomic = !!(typ & kAccessAtomic);
    ev->size_log = size_log;
    ev->pc_delta = pc_delta;
    DCHECK_EQ(ev->pc_delta, pc_delta);
    ev->addr = CompressAddr(addr);
    TraceRelease(thr, ev);
    return true;
  }
  auto *evex = reinterpret_cast<EventAccessExt *>(ev);
  evex->is_access = 0;
  evex->is_func = 0;
  evex->type = EventType::kAccessExt;
  evex->is_read = !!(typ & kAccessRead);
  evex->is_atomic = !!(typ & kAccessAtomic);
  evex->size_log = size_log;
  evex->addr = CompressAddr(addr);
  evex->pc = pc;
  TraceRelease(thr, evex);
  return true;
}

ALWAYS_INLINE USED bool TryTraceMemoryAccessRange(ThreadState *thr, uptr pc,
                                                  uptr addr, uptr size,
                                                  AccessType typ) {
  if (!kCollectHistory)
    return true;
  EventAccessRange *ev;
  if (UNLIKELY(!TraceAcquire(thr, &ev)))
    return false;
  thr->trace_prev_pc = pc;
  ev->is_access = 0;
  ev->is_func = 0;
  ev->type = EventType::kAccessRange;
  ev->is_read = !!(typ & kAccessRead);
  ev->is_free = !!(typ & kAccessFree);
  ev->size_lo = size;
  ev->pc = CompressAddr(pc);
  ev->addr = CompressAddr(addr);
  ev->size_hi = size >> EventAccessRange::kSizeLoBits;
  TraceRelease(thr, ev);
  return true;
}

void TraceMemoryAccessRange(ThreadState *thr, uptr pc, uptr addr, uptr size,
                            AccessType typ) {
  if (LIKELY(TryTraceMemoryAccessRange(thr, pc, addr, size, typ)))
    return;
  TraceSwitchPart(thr);
  UNUSED bool res = TryTraceMemoryAccessRange(thr, pc, addr, size, typ);
  DCHECK(res);
}

void TraceFunc(ThreadState *thr, uptr pc) {
  if (LIKELY(TryTraceFunc(thr, pc)))
    return;
  TraceSwitchPart(thr);
  UNUSED bool res = TryTraceFunc(thr, pc);
  DCHECK(res);
}

void TraceMutexLock(ThreadState *thr, EventType type, uptr pc, uptr addr,
                    StackID stk) {
  DCHECK(type == EventType::kLock || type == EventType::kRLock);
  if (!kCollectHistory)
    return;
  EventLock ev;
  ev.is_access = 0;
  ev.is_func = 0;
  ev.type = type;
  ev.pc = CompressAddr(pc);
  ev.stack_lo = stk;
  ev.stack_hi = stk >> EventLock::kStackIDLoBits;
  ev._ = 0;
  ev.addr = CompressAddr(addr);
  TraceEvent(thr, ev);
}

void TraceMutexUnlock(ThreadState *thr, uptr addr) {
  if (!kCollectHistory)
    return;
  EventUnlock ev;
  ev.is_access = 0;
  ev.is_func = 0;
  ev.type = EventType::kUnlock;
  ev._ = 0;
  ev.addr = CompressAddr(addr);
  TraceEvent(thr, ev);
}

void TraceTime(ThreadState *thr) {
  if (!kCollectHistory)
    return;
  EventTime ev;
  ev.is_access = 0;
  ev.is_func = 0;
  ev.type = EventType::kTime;
  ev.sid = static_cast<u64>(thr->sid);
  ev.epoch = static_cast<u64>(thr->epoch);
  ev._ = 0;
  TraceEvent(thr, ev);
}

}  // namespace v3

ALWAYS_INLINE
Shadow LoadShadow(u64 *p) {
  u64 raw = atomic_load((atomic_uint64_t *)p, memory_order_relaxed);
  return Shadow(raw);
}

ALWAYS_INLINE
void StoreShadow(u64 *sp, u64 s) {
  atomic_store((atomic_uint64_t *)sp, s, memory_order_relaxed);
}

ALWAYS_INLINE
void StoreIfNotYetStored(u64 *sp, u64 *s) {
  StoreShadow(sp, *s);
  *s = 0;
}

extern "C" void __tsan_report_race();

ALWAYS_INLINE
void HandleRace(ThreadState *thr, u64 *shadow_mem, Shadow cur, Shadow old) {
  thr->racy_state[0] = cur.raw();
  thr->racy_state[1] = old.raw();
  thr->racy_shadow_addr = shadow_mem;
#if !SANITIZER_GO
  HACKY_CALL(__tsan_report_race);
#else
  ReportRace(thr);
#endif
}

static inline bool HappensBefore(Shadow old, ThreadState *thr) {
  return thr->clock.get(old.TidWithIgnore()) >= old.epoch();
}

ALWAYS_INLINE
void MemoryAccessImpl1(ThreadState *thr, uptr addr, int kAccessSizeLog,
                       bool kAccessIsWrite, bool kIsAtomic, u64 *shadow_mem,
                       Shadow cur) {
  // This potentially can live in an MMX/SSE scratch register.
  // The required intrinsics are:
  // __m128i _mm_move_epi64(__m128i*);
  // _mm_storel_epi64(u64*, __m128i);
  u64 store_word = cur.raw();
  bool stored = false;

  // scan all the shadow values and dispatch to 4 categories:
  // same, replace, candidate and race (see comments below).
  // we consider only 3 cases regarding access sizes:
  // equal, intersect and not intersect. initially I considered
  // larger and smaller as well, it allowed to replace some
  // 'candidates' with 'same' or 'replace', but I think
  // it's just not worth it (performance- and complexity-wise).

  Shadow old(0);

  // It release mode we manually unroll the loop,
  // because empirically gcc generates better code this way.
  // However, we can't afford unrolling in debug mode, because the function
  // consumes almost 4K of stack. Gtest gives only 4K of stack to death test
  // threads, which is not enough for the unrolled loop.
#if SANITIZER_DEBUG
  for (int idx = 0; idx < 4; idx++) {
#  include "tsan_update_shadow_word.inc"
  }
#else
  int idx = 0;
#  include "tsan_update_shadow_word.inc"
  idx = 1;
  if (stored) {
#  include "tsan_update_shadow_word.inc"
  } else {
#  include "tsan_update_shadow_word.inc"
  }
  idx = 2;
  if (stored) {
#  include "tsan_update_shadow_word.inc"
  } else {
#  include "tsan_update_shadow_word.inc"
  }
  idx = 3;
  if (stored) {
#  include "tsan_update_shadow_word.inc"
  } else {
#  include "tsan_update_shadow_word.inc"
  }
#endif

  // we did not find any races and had already stored
  // the current access info, so we are done
  if (LIKELY(stored))
    return;
  // choose a random candidate slot and replace it
  StoreShadow(shadow_mem + (cur.epoch() % kShadowCnt), store_word);
  return;
RACE:
  HandleRace(thr, shadow_mem, cur, old);
  return;
}

void UnalignedMemoryAccess(ThreadState *thr, uptr pc, uptr addr, uptr size,
                           AccessType typ) {
  DCHECK(!(typ & kAccessAtomic));
  const bool kAccessIsWrite = !(typ & kAccessRead);
  const bool kIsAtomic = false;
  while (size) {
    int size1 = 1;
    int kAccessSizeLog = kSizeLog1;
    if (size >= 8 && (addr & ~7) == ((addr + 7) & ~7)) {
      size1 = 8;
      kAccessSizeLog = kSizeLog8;
    } else if (size >= 4 && (addr & ~7) == ((addr + 3) & ~7)) {
      size1 = 4;
      kAccessSizeLog = kSizeLog4;
    } else if (size >= 2 && (addr & ~7) == ((addr + 1) & ~7)) {
      size1 = 2;
      kAccessSizeLog = kSizeLog2;
    }
    MemoryAccess(thr, pc, addr, kAccessSizeLog, kAccessIsWrite, kIsAtomic);
    addr += size1;
    size -= size1;
  }
}

ALWAYS_INLINE
bool ContainsSameAccessSlow(u64 *s, u64 a, u64 sync_epoch, bool is_write) {
  Shadow cur(a);
  for (uptr i = 0; i < kShadowCnt; i++) {
    Shadow old(LoadShadow(&s[i]));
    if (Shadow::Addr0AndSizeAreEqual(cur, old) &&
        old.TidWithIgnore() == cur.TidWithIgnore() &&
        old.epoch() > sync_epoch && old.IsAtomic() == cur.IsAtomic() &&
        old.IsRead() <= cur.IsRead())
      return true;
  }
  return false;
}

#if TSAN_VECTORIZE
#  define SHUF(v0, v1, i0, i1, i2, i3)                    \
    _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(v0), \
                                    _mm_castsi128_ps(v1), \
                                    (i0)*1 + (i1)*4 + (i2)*16 + (i3)*64))
ALWAYS_INLINE
bool ContainsSameAccessFast(u64 *s, u64 a, u64 sync_epoch, bool is_write) {
  // This is an optimized version of ContainsSameAccessSlow.
  // load current access into access[0:63]
  const m128 access = _mm_cvtsi64_si128(a);
  // duplicate high part of access in addr0:
  // addr0[0:31]        = access[32:63]
  // addr0[32:63]       = access[32:63]
  // addr0[64:95]       = access[32:63]
  // addr0[96:127]      = access[32:63]
  const m128 addr0 = SHUF(access, access, 1, 1, 1, 1);
  // load 4 shadow slots
  const m128 shadow0 = _mm_load_si128((__m128i *)s);
  const m128 shadow1 = _mm_load_si128((__m128i *)s + 1);
  // load high parts of 4 shadow slots into addr_vect:
  // addr_vect[0:31]    = shadow0[32:63]
  // addr_vect[32:63]   = shadow0[96:127]
  // addr_vect[64:95]   = shadow1[32:63]
  // addr_vect[96:127]  = shadow1[96:127]
  m128 addr_vect = SHUF(shadow0, shadow1, 1, 3, 1, 3);
  if (!is_write) {
    // set IsRead bit in addr_vect
    const m128 rw_mask1 = _mm_cvtsi64_si128(1 << 15);
    const m128 rw_mask = SHUF(rw_mask1, rw_mask1, 0, 0, 0, 0);
    addr_vect = _mm_or_si128(addr_vect, rw_mask);
  }
  // addr0 == addr_vect?
  const m128 addr_res = _mm_cmpeq_epi32(addr0, addr_vect);
  // epoch1[0:63]       = sync_epoch
  const m128 epoch1 = _mm_cvtsi64_si128(sync_epoch);
  // epoch[0:31]        = sync_epoch[0:31]
  // epoch[32:63]       = sync_epoch[0:31]
  // epoch[64:95]       = sync_epoch[0:31]
  // epoch[96:127]      = sync_epoch[0:31]
  const m128 epoch = SHUF(epoch1, epoch1, 0, 0, 0, 0);
  // load low parts of shadow cell epochs into epoch_vect:
  // epoch_vect[0:31]   = shadow0[0:31]
  // epoch_vect[32:63]  = shadow0[64:95]
  // epoch_vect[64:95]  = shadow1[0:31]
  // epoch_vect[96:127] = shadow1[64:95]
  const m128 epoch_vect = SHUF(shadow0, shadow1, 0, 2, 0, 2);
  // epoch_vect >= sync_epoch?
  const m128 epoch_res = _mm_cmpgt_epi32(epoch_vect, epoch);
  // addr_res & epoch_res
  const m128 res = _mm_and_si128(addr_res, epoch_res);
  // mask[0] = res[7]
  // mask[1] = res[15]
  // ...
  // mask[15] = res[127]
  const int mask = _mm_movemask_epi8(res);
  return mask != 0;
}
#endif

ALWAYS_INLINE
bool ContainsSameAccess(u64 *s, u64 a, u64 sync_epoch, bool is_write) {
#if TSAN_VECTORIZE
  bool res = ContainsSameAccessFast(s, a, sync_epoch, is_write);
  // NOTE: this check can fail if the shadow is concurrently mutated
  // by other threads. But it still can be useful if you modify
  // ContainsSameAccessFast and want to ensure that it's not completely broken.
  // DCHECK_EQ(res, ContainsSameAccessSlow(s, a, sync_epoch, is_write));
  return res;
#else
  return ContainsSameAccessSlow(s, a, sync_epoch, is_write);
#endif
}

ALWAYS_INLINE USED void MemoryAccess(ThreadState *thr, uptr pc, uptr addr,
                                     int kAccessSizeLog, bool kAccessIsWrite,
                                     bool kIsAtomic) {
  RawShadow *shadow_mem = MemToShadow(addr);
  DPrintf2(
      "#%d: MemoryAccess: @%p %p size=%d"
      " is_write=%d shadow_mem=%p {%zx, %zx, %zx, %zx}\n",
      (int)thr->fast_state.tid(), (void *)pc, (void *)addr,
      (int)(1 << kAccessSizeLog), kAccessIsWrite, shadow_mem,
      (uptr)shadow_mem[0], (uptr)shadow_mem[1], (uptr)shadow_mem[2],
      (uptr)shadow_mem[3]);
#if SANITIZER_DEBUG
  if (!IsAppMem(addr)) {
    Printf("Access to non app mem %zx\n", addr);
    DCHECK(IsAppMem(addr));
  }
  if (!IsShadowMem(shadow_mem)) {
    Printf("Bad shadow addr %p (%zx)\n", shadow_mem, addr);
    DCHECK(IsShadowMem(shadow_mem));
  }
#endif

  if (!SANITIZER_GO && !kAccessIsWrite && *shadow_mem == kShadowRodata) {
    // Access to .rodata section, no races here.
    // Measurements show that it can be 10-20% of all memory accesses.
    return;
  }

  FastState fast_state = thr->fast_state;
  if (UNLIKELY(fast_state.GetIgnoreBit())) {
    return;
  }

  Shadow cur(fast_state);
  cur.SetAddr0AndSizeLog(addr & 7, kAccessSizeLog);
  cur.SetWrite(kAccessIsWrite);
  cur.SetAtomic(kIsAtomic);

  if (LIKELY(ContainsSameAccess(shadow_mem, cur.raw(), thr->fast_synch_epoch,
                                kAccessIsWrite))) {
    return;
  }

  if (kCollectHistory) {
    fast_state.IncrementEpoch();
    thr->fast_state = fast_state;
    TraceAddEvent(thr, fast_state, EventTypeMop, pc);
    cur.IncrementEpoch();
  }

  MemoryAccessImpl1(thr, addr, kAccessSizeLog, kAccessIsWrite, kIsAtomic,
                    shadow_mem, cur);
}

// Called by MemoryAccessRange in tsan_rtl_thread.cpp
ALWAYS_INLINE USED void MemoryAccessImpl(ThreadState *thr, uptr addr,
                                         int kAccessSizeLog,
                                         bool kAccessIsWrite, bool kIsAtomic,
                                         u64 *shadow_mem, Shadow cur) {
  if (LIKELY(ContainsSameAccess(shadow_mem, cur.raw(), thr->fast_synch_epoch,
                                kAccessIsWrite))) {
    return;
  }

  MemoryAccessImpl1(thr, addr, kAccessSizeLog, kAccessIsWrite, kIsAtomic,
                    shadow_mem, cur);
}

static void MemoryRangeSet(ThreadState *thr, uptr pc, uptr addr, uptr size,
                           u64 val) {
  (void)thr;
  (void)pc;
  if (size == 0)
    return;
  // FIXME: fix me.
  uptr offset = addr % kShadowCell;
  if (offset) {
    offset = kShadowCell - offset;
    if (size <= offset)
      return;
    addr += offset;
    size -= offset;
  }
  DCHECK_EQ(addr % 8, 0);
  // If a user passes some insane arguments (memset(0)),
  // let it just crash as usual.
  if (!IsAppMem(addr) || !IsAppMem(addr + size - 1))
    return;
  // Don't want to touch lots of shadow memory.
  // If a program maps 10MB stack, there is no need reset the whole range.
  size = (size + (kShadowCell - 1)) & ~(kShadowCell - 1);
  // UnmapOrDie/MmapFixedNoReserve does not work on Windows.
  if (SANITIZER_WINDOWS || size < common_flags()->clear_shadow_mmap_threshold) {
    RawShadow *p = MemToShadow(addr);
    CHECK(IsShadowMem(p));
    CHECK(IsShadowMem(p + size * kShadowCnt / kShadowCell - 1));
    // FIXME: may overwrite a part outside the region
    for (uptr i = 0; i < size / kShadowCell * kShadowCnt;) {
      p[i++] = val;
      for (uptr j = 1; j < kShadowCnt; j++) p[i++] = 0;
    }
  } else {
    // The region is big, reset only beginning and end.
    const uptr kPageSize = GetPageSizeCached();
    RawShadow *begin = MemToShadow(addr);
    RawShadow *end = begin + size / kShadowCell * kShadowCnt;
    RawShadow *p = begin;
    // Set at least first kPageSize/2 to page boundary.
    while ((p < begin + kPageSize / kShadowSize / 2) || ((uptr)p % kPageSize)) {
      *p++ = val;
      for (uptr j = 1; j < kShadowCnt; j++) *p++ = 0;
    }
    // Reset middle part.
    RawShadow *p1 = p;
    p = RoundDown(end, kPageSize);
    if (!MmapFixedSuperNoReserve((uptr)p1, (uptr)p - (uptr)p1))
      Die();
    // Set the ending.
    while (p < end) {
      *p++ = val;
      for (uptr j = 1; j < kShadowCnt; j++) *p++ = 0;
    }
  }
}

void MemoryResetRange(ThreadState *thr, uptr pc, uptr addr, uptr size) {
  MemoryRangeSet(thr, pc, addr, size, 0);
}

void MemoryRangeFreed(ThreadState *thr, uptr pc, uptr addr, uptr size) {
  // Processing more than 1k (4k of shadow) is expensive,
  // can cause excessive memory consumption (user does not necessary touch
  // the whole range) and most likely unnecessary.
  if (size > 1024)
    size = 1024;
  CHECK_EQ(thr->is_freeing, false);
  thr->is_freeing = true;
  MemoryAccessRange(thr, pc, addr, size, true);
  thr->is_freeing = false;
  if (kCollectHistory) {
    thr->fast_state.IncrementEpoch();
    TraceAddEvent(thr, thr->fast_state, EventTypeMop, pc);
  }
  Shadow s(thr->fast_state);
  s.ClearIgnoreBit();
  s.MarkAsFreed();
  s.SetWrite(true);
  s.SetAddr0AndSizeLog(0, 3);
  MemoryRangeSet(thr, pc, addr, size, s.raw());
}

void MemoryRangeImitateWrite(ThreadState *thr, uptr pc, uptr addr, uptr size) {
  if (kCollectHistory) {
    thr->fast_state.IncrementEpoch();
    TraceAddEvent(thr, thr->fast_state, EventTypeMop, pc);
  }
  Shadow s(thr->fast_state);
  s.ClearIgnoreBit();
  s.SetWrite(true);
  s.SetAddr0AndSizeLog(0, 3);
  MemoryRangeSet(thr, pc, addr, size, s.raw());
}

void MemoryRangeImitateWriteOrResetRange(ThreadState *thr, uptr pc, uptr addr,
                                         uptr size) {
  if (thr->ignore_reads_and_writes == 0)
    MemoryRangeImitateWrite(thr, pc, addr, size);
  else
    MemoryResetRange(thr, pc, addr, size);
}

void MemoryAccessRange(ThreadState *thr, uptr pc, uptr addr, uptr size,
                       bool is_write) {
  if (size == 0)
    return;

  RawShadow *shadow_mem = MemToShadow(addr);
  DPrintf2("#%d: MemoryAccessRange: @%p %p size=%d is_write=%d\n", thr->tid,
           (void *)pc, (void *)addr, (int)size, is_write);

#if SANITIZER_DEBUG
  if (!IsAppMem(addr)) {
    Printf("Access to non app mem %zx\n", addr);
    DCHECK(IsAppMem(addr));
  }
  if (!IsAppMem(addr + size - 1)) {
    Printf("Access to non app mem %zx\n", addr + size - 1);
    DCHECK(IsAppMem(addr + size - 1));
  }
  if (!IsShadowMem(shadow_mem)) {
    Printf("Bad shadow addr %p (%zx)\n", shadow_mem, addr);
    DCHECK(IsShadowMem(shadow_mem));
  }
  if (!IsShadowMem(shadow_mem + size * kShadowCnt / 8 - 1)) {
    Printf("Bad shadow addr %p (%zx)\n", shadow_mem + size * kShadowCnt / 8 - 1,
           addr + size - 1);
    DCHECK(IsShadowMem(shadow_mem + size * kShadowCnt / 8 - 1));
  }
#endif

  if (*shadow_mem == kShadowRodata) {
    DCHECK(!is_write);
    // Access to .rodata section, no races here.
    // Measurements show that it can be 10-20% of all memory accesses.
    return;
  }

  FastState fast_state = thr->fast_state;
  if (fast_state.GetIgnoreBit())
    return;

  fast_state.IncrementEpoch();
  thr->fast_state = fast_state;
  TraceAddEvent(thr, fast_state, EventTypeMop, pc);

  bool unaligned = (addr % kShadowCell) != 0;

  // Handle unaligned beginning, if any.
  for (; addr % kShadowCell && size; addr++, size--) {
    int const kAccessSizeLog = 0;
    Shadow cur(fast_state);
    cur.SetWrite(is_write);
    cur.SetAddr0AndSizeLog(addr & (kShadowCell - 1), kAccessSizeLog);
    MemoryAccessImpl(thr, addr, kAccessSizeLog, is_write, false, shadow_mem,
                     cur);
  }
  if (unaligned)
    shadow_mem += kShadowCnt;
  // Handle middle part, if any.
  for (; size >= kShadowCell; addr += kShadowCell, size -= kShadowCell) {
    int const kAccessSizeLog = 3;
    Shadow cur(fast_state);
    cur.SetWrite(is_write);
    cur.SetAddr0AndSizeLog(0, kAccessSizeLog);
    MemoryAccessImpl(thr, addr, kAccessSizeLog, is_write, false, shadow_mem,
                     cur);
    shadow_mem += kShadowCnt;
  }
  // Handle ending, if any.
  for (; size; addr++, size--) {
    int const kAccessSizeLog = 0;
    Shadow cur(fast_state);
    cur.SetWrite(is_write);
    cur.SetAddr0AndSizeLog(addr & (kShadowCell - 1), kAccessSizeLog);
    MemoryAccessImpl(thr, addr, kAccessSizeLog, is_write, false, shadow_mem,
                     cur);
  }
}

}  // namespace __tsan

#if !SANITIZER_GO
// Must be included in this file to make sure everything is inlined.
#  include "tsan_interface.inc"
#endif

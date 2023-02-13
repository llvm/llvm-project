//===-- trec_rtl_mutex.cpp
//------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of TraceRecorder (TRec), a race detector.
//
//===----------------------------------------------------------------------===//

#include <sanitizer_common/sanitizer_deadlock_detector_interface.h>
#include <sanitizer_common/sanitizer_stackdepot.h>

#include "trec_flags.h"
#include "trec_platform.h"
#include "trec_rtl.h"

namespace __trec {

void MutexCreate(ThreadState *thr, uptr pc, uptr addr, u32 flagz) {
  DPrintf("#%d: MutexCreate %zx flagz=0x%x\n", thr->tid, addr, flagz);
}

void MutexDestroy(ThreadState *thr, uptr pc, uptr addr, u32 flagz) {
  DPrintf("#%d: MutexDestroy %zx\n", thr->tid, addr);
}

void MutexPreLock(ThreadState *thr, uptr pc, uptr addr, u32 flagz) {
  DPrintf("#%d: MutexPreLock %zx flagz=0x%x\n", thr->tid, addr, flagz);
}

void MutexPostLock(ThreadState *thr, uptr pc, uptr addr,
                   __trec_metadata::SourceAddressInfo SAI, u32 flagz, int rec) {
  DPrintf("#%d: MutexPostLock %zx flag=0x%x rec=%d\n", thr->tid, addr, flagz,
          rec);
  if (LIKELY(ctx->flags.output_trace) && LIKELY(ctx->flags.record_mutex)&&
      LIKELY(thr->ignore_interceptors == 0)&& LIKELY(thr->should_record)) {
    if (ctx->flags.trace_mode == 1) {
      __seqc_trace::Event e_re1, e;
      e.type = __seqc_trace::EventType::REQUEST;
      e.eid = thr->tctx->event_cnt++;
      e.iid = pc;
      e.oid = addr;
      e.tid = thr->tid;
      ctx->seqc_mtx.Lock();
      e.tot = atomic_fetch_add(&ctx->global_id, 1, memory_order_relaxed);
      ctx->put_seqc_trace(&e, sizeof(e));
      ctx->trace_summary.arNum += 1;
      ctx->seqc_mtx.Unlock();
      e.type = __seqc_trace::EventType::ACQUIRE;
      e.eid = thr->tctx->event_cnt++;
      e.iid = pc;
      e.oid = addr;
      e.tid = thr->tid;
      ctx->seqc_mtx.Lock();
      e.tot = atomic_fetch_add(&ctx->global_id, 1, memory_order_relaxed);
      ctx->put_seqc_trace(&e, sizeof(e));
      ctx->trace_summary.arNum += 1;
      ctx->seqc_mtx.Unlock();
    } else if (ctx->flags.trace_mode == 2 || ctx->flags.trace_mode == 3) {
      __trec_trace::Event e(
          __trec_trace::EventType::MutexLock,thr->tid,
          atomic_fetch_add(&ctx->global_id, 1, memory_order_relaxed),
          addr & ((((u64)1) << 48) - 1), thr->tctx->metadata_offset,pc);

      __trec_metadata::MutexMeta meta(SAI.idx, SAI.addr);
      thr->tctx->put_metadata(&meta, sizeof(meta));
      thr->tctx->put_trace(&e, sizeof(__trec_trace::Event));
      thr->tctx->header.StateInc(__trec_header::RecordType::MutexLock);
    }
  }
}

int MutexUnlock(ThreadState *thr, uptr pc, uptr addr,
                __trec_metadata::SourceAddressInfo SAI, u32 flagz) {
  DPrintf("#%d: MutexUnlock %zx flagz=0x%x\n", thr->tid, addr, flagz);
  if (LIKELY(ctx->flags.output_trace) && LIKELY(ctx->flags.record_mutex)&&
      LIKELY(thr->ignore_interceptors == 0)&& LIKELY(thr->should_record)) {
    if (ctx->flags.trace_mode == 1) {
      __seqc_trace::Event e;
      e.type = __seqc_trace::EventType::RELEASE;
      e.eid = thr->tctx->event_cnt++;
      e.iid = pc;
      e.oid = addr;
      e.tid = thr->tid;
      ctx->seqc_mtx.Lock();
      e.tot = atomic_fetch_add(&ctx->global_id, 1, memory_order_relaxed);
      ctx->put_seqc_trace(&e, sizeof(e));
      ctx->trace_summary.arNum += 1;
      ctx->seqc_mtx.Unlock();
    } else if (ctx->flags.trace_mode == 2 || ctx->flags.trace_mode == 3) {
      __trec_trace::Event e(
          __trec_trace::EventType::MutexUnlock,thr->tid,
          atomic_fetch_add(&ctx->global_id, 1, memory_order_relaxed),
          addr & ((((u64)1) << 48) - 1), thr->tctx->metadata_offset,pc);
      __trec_metadata::MutexMeta meta(SAI.idx, SAI.addr);
      thr->tctx->put_metadata(&meta, sizeof(meta));

      thr->tctx->put_trace(&e, sizeof(__trec_trace::Event));
      thr->tctx->header.StateInc(__trec_header::RecordType::MutexUnlock);
    }
  }
  return 0;
}

void CondWait(ThreadState *thr, uptr pc, uptr cond, uptr mutex,
              __trec_metadata::SourceAddressInfo cond_SAI,
              __trec_metadata::SourceAddressInfo mutex_SAI) {
  if (LIKELY(ctx->flags.output_trace)&&
      LIKELY(thr->ignore_interceptors == 0)&& LIKELY(thr->should_record)) {
    if (ctx->flags.trace_mode == 2) {
      __trec_trace::Event e(
          __trec_trace::EventType::CondWait,thr->tid,
          atomic_fetch_add(&ctx->global_id, 1, memory_order_relaxed),
          cond & ((((u64)1) << 48) - 1), thr->tctx->metadata_offset,pc);
      __trec_metadata::CondWaitMeta meta(
          mutex, cond_SAI.idx, cond_SAI.addr, mutex_SAI.idx, mutex_SAI.addr);
      thr->tctx->put_metadata(&meta, sizeof(meta));

      thr->tctx->put_trace(&e, sizeof(__trec_trace::Event));
      thr->tctx->header.StateInc(__trec_header::RecordType::CondWait);
    }
  }
}

void CondSignal(ThreadState *thr, uptr pc, uptr cond, bool is_broadcast,
                __trec_metadata::SourceAddressInfo SAI) {
  if (LIKELY(ctx->flags.output_trace)&&
      LIKELY(thr->ignore_interceptors == 0)&& LIKELY(thr->should_record)) {
    if (ctx->flags.trace_mode == 2) {
      __trec_trace::Event e(
          is_broadcast ? __trec_trace::EventType::CondBroadcast
                       : __trec_trace::EventType::CondSignal,thr->tid,
          atomic_fetch_add(&ctx->global_id, 1, memory_order_relaxed),
          cond & ((((u64)1) << 48) - 1), thr->tctx->metadata_offset,pc);
      __trec_metadata::CondSignalMeta meta(SAI.idx, SAI.addr);
      thr->tctx->put_metadata(&meta, sizeof(meta));

      thr->tctx->put_trace(&e, sizeof(__trec_trace::Event));
      thr->tctx->header.StateInc(is_broadcast
                                     ? __trec_header::RecordType::CondBroadcast
                                     : __trec_header::RecordType::CondSignal);
    }
  }
}

void MutexPreReadLock(ThreadState *thr, uptr pc, uptr addr, u32 flagz) {
  DPrintf("#%d: MutexPreReadLock %zx flagz=0x%x\n", thr->tid, addr, flagz);
}

void MutexPostReadLock(ThreadState *thr, uptr pc, uptr addr, u32 flagz,
                       __trec_metadata::SourceAddressInfo SAI) {
  DPrintf("#%d: MutexPostReadLock %zx flagz=0x%x\n", thr->tid, addr, flagz);
  if (LIKELY(ctx->flags.output_trace)&&
      LIKELY(thr->ignore_interceptors == 0)&& LIKELY(thr->should_record)) {
    if (ctx->flags.trace_mode == 2) {
      __trec_trace::Event e(
          __trec_trace::EventType::ReaderLock,thr->tid,
          atomic_fetch_add(&ctx->global_id, 1, memory_order_relaxed),
          addr & ((((u64)1) << 48) - 1), thr->tctx->metadata_offset,pc);
      __trec_metadata::MutexMeta meta(SAI.idx, SAI.addr);
      thr->tctx->put_metadata(&meta, sizeof(meta));

      thr->tctx->put_trace(&e, sizeof(__trec_trace::Event));
      thr->tctx->header.StateInc(__trec_header::RecordType::ReaderLock);
    }
  }
}

void MutexReadOrWriteUnlock(ThreadState *thr, uptr pc, uptr addr) {
  DPrintf("#%d: MutexReadOrWriteUnlock %zx\n", thr->tid, addr);
}

void MutexRepair(ThreadState *thr, uptr pc, uptr addr) {
  DPrintf("#%d: MutexRepair %zx\n", thr->tid, addr);
}

void MutexInvalidAccess(ThreadState *thr, uptr pc, uptr addr) {
  DPrintf("#%d: MutexInvalidAccess %zx\n", thr->tid, addr);
}

void ReleaseStoreAcquire(ThreadState *thr, uptr pc, uptr addr) {
  DPrintf("#%d: ReleaseStoreAcquire %zx\n", thr->tid, addr);
}

void ReleaseStore(ThreadState *thr, uptr pc, uptr addr) {
  DPrintf("#%d: ReleaseStore %zx\n", thr->tid, addr);
}

#if !SANITIZER_GO
void AfterSleep(ThreadState *thr, uptr pc) {
  DPrintf("#%d: AfterSleep %zx\n", thr->tid);
  if (thr->ignore_sync)
    return;
}
#endif

}  // namespace __trec

//===-- tsan_rtl_mutex.cpp ------------------------------------------------===//
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

#include <sanitizer_common/sanitizer_deadlock_detector_interface.h>
#include <sanitizer_common/sanitizer_stackdepot.h>

#include "tsan_rtl.h"
#include "tsan_flags.h"
#include "tsan_sync.h"
#include "tsan_report.h"
#include "tsan_symbolize.h"
#include "tsan_platform.h"

namespace __tsan {

void ReportDeadlock(ThreadState *thr, uptr pc, DDReport *r);
void ReportDestroyLocked(ThreadState* thr, uptr pc, uptr addr, FastState last_lock,
                         StackID creation_stack_id);
void ReportMutexMisuse(ThreadState* thr, uptr pc, ReportType typ, uptr addr,
                       StackID creation_stack_id);

struct Callback final : public DDCallback {
  ThreadState *thr;
  uptr pc;

  Callback(ThreadState *thr, uptr pc)
      : thr(thr)
      , pc(pc) {
    DDCallback::pt = thr->proc()->dd_pt;
    DDCallback::lt = thr->dd_lt;
  }

  StackID Unwind() override {
    return CurrentStackId(thr, pc);
  }
  int UniqueTid() override {
    return thr->tid;
  }
};

void DDMutexInit(ThreadState *thr, uptr pc, SyncVar *s) {
  Callback cb(thr, pc);
  ctx->dd->MutexInit(&cb, &s->dd);
  s->dd.ctx = s->addr;
}

void MutexCreate(ThreadState *thr, uptr pc, uptr addr, u32 flagz) {
  DPrintf("#%d: MutexCreate %zx flagz=0x%x\n", thr->tid, addr, flagz);
  if (!(flagz & MutexFlagLinkerInit) && pc && IsAppMem(addr))
    MemoryAccess(thr, pc, addr, 1, AccessWrite);
  SlotLocker locker(thr);
  auto s = ctx->metamap.GetSyncOrCreate(thr, pc, addr, true);
  s->SetFlags(flagz & MutexCreationFlagMask);
  // Save stack in the case the sync object was created before as atomic.
  if (!SANITIZER_GO && s->creation_stack_id == kInvalidStackID)
    s->creation_stack_id = CurrentStackId(thr, pc);
}

void MutexDestroy(ThreadState *thr, uptr pc, uptr addr, u32 flagz) {
  DPrintf("#%d: MutexDestroy %zx\n", thr->tid, addr);
  bool unlock_locked = false;
  StackID creation_stack_id;
  FastState last_lock;
  {
    SlotLocker locker(thr);
    auto s = ctx->metamap.GetSyncIfExists(thr, pc, addr);
    if (!s)
      return;
    Lock lock(&s->mtx);
    creation_stack_id = s->creation_stack_id;
    last_lock = s->last_lock;
    if ((flagz & MutexFlagLinkerInit) || s->IsFlagSet(MutexFlagLinkerInit) ||
        ((flagz & MutexFlagNotStatic) && !s->IsFlagSet(MutexFlagNotStatic))) {
      // Destroy is no-op for linker-initialized mutexes.
      return;
    }
    if (common_flags()->detect_deadlocks) {
      Callback cb(thr, pc);
      ctx->dd->MutexDestroy(&cb, &s->dd);
      ctx->dd->MutexInit(&cb, &s->dd);
    }
    if (flags()->report_destroy_locked && s->owner_tid != kInvalidTid &&
        !s->IsFlagSet(MutexFlagBroken)) {
      s->SetFlags(MutexFlagBroken);
      unlock_locked = true;
    }
    s->Reset();
  }
  // Imitate a memory write to catch unlock-destroy races.
  if (pc && IsAppMem(addr))
    MemoryAccess(thr, pc, addr, 1, AccessWrite | AccessFree);
  if (unlock_locked && ShouldReport(thr, ReportTypeMutexDestroyLocked))
    ReportDestroyLocked(thr, pc, addr, last_lock, creation_stack_id);
  thr->mset.Del(addr, true);
  // s will be destroyed and freed in MetaMap::FreeBlock.
}

void MutexPreLock(ThreadState *thr, uptr pc, uptr addr, u32 flagz) {
  DPrintf("#%d: MutexPreLock %zx flagz=0x%x\n", thr->tid, addr, flagz);
  if (flagz & MutexFlagTryLock)
    return;
  if (!common_flags()->detect_deadlocks)
    return;
  Callback cb(thr, pc);
  {
    SlotLocker locker(thr);
    auto s = ctx->metamap.GetSyncOrCreate(thr, pc, addr, true);
    ReadLock lock(&s->mtx);
    s->UpdateFlags(flagz);
    if (s->owner_tid != thr->tid)
      ctx->dd->MutexBeforeLock(&cb, &s->dd, true);
  }
  ReportDeadlock(thr, pc, ctx->dd->GetReport(&cb));
}

void MutexPostLock(ThreadState *thr, uptr pc, uptr addr, u32 flagz, int rec) {
  DPrintf("#%d: MutexPostLock %zx flag=0x%x rec=%d\n",
      thr->tid, addr, flagz, rec);
  if (flagz & MutexFlagRecursiveLock)
    CHECK_GT(rec, 0);
  else
    rec = 1;
  if (pc && IsAppMem(addr))
    MemoryAccess(thr, pc, addr, 1, AccessRead | AccessAtomic);
  SlotLocker locker(thr);
  auto s = ctx->metamap.GetSyncOrCreate(thr, pc, addr, true);
  StackID creation_stack_id = s->creation_stack_id;
  TraceMutexLock(thr, EventTypeLock, pc, addr, creation_stack_id);
  thr->mset.Add(addr, creation_stack_id, true);
  bool report_double_lock = false;
  bool pre_lock = false;
  bool first = false;
  {
    Lock lock(&s->mtx);
    first = s->recursion == 0;
    s->UpdateFlags(flagz);
    if (s->owner_tid == kInvalidTid) {
      CHECK_EQ(s->recursion, 0);
      s->owner_tid = thr->tid;
      s->last_lock = thr->fast_state;
    } else if (s->owner_tid == thr->tid) {
      CHECK_GT(s->recursion, 0);
    } else if (flags()->report_mutex_bugs && !s->IsFlagSet(MutexFlagBroken)) {
      s->SetFlags(MutexFlagBroken);
      report_double_lock = true;
    }
    s->recursion += rec;
    if (first) {
      if (!thr->ignore_sync) {
        thr->clock.Acquire(s->clock);
        thr->clock.Acquire(s->read_clock);
      }
    }
    if (first && common_flags()->detect_deadlocks) {
      pre_lock =
          (flagz & MutexFlagDoPreLockOnPostLock) && !(flagz & MutexFlagTryLock);
      Callback cb(thr, pc);
      if (pre_lock)
        ctx->dd->MutexBeforeLock(&cb, &s->dd, true);
      ctx->dd->MutexAfterLock(&cb, &s->dd, true, flagz & MutexFlagTryLock);
    }
  }
  if (report_double_lock)
    ReportMutexMisuse(thr, pc, ReportTypeMutexDoubleLock, addr,
                      creation_stack_id);
  if (first && pre_lock && common_flags()->detect_deadlocks) {
    Callback cb(thr, pc);
    ReportDeadlock(thr, pc, ctx->dd->GetReport(&cb));
  }
}

int MutexUnlock(ThreadState *thr, uptr pc, uptr addr, u32 flagz) {
  DPrintf("#%d: MutexUnlock %zx flagz=0x%x\n", thr->tid, addr, flagz);
  if (pc && IsAppMem(addr))
    MemoryAccess(thr, pc, addr, 1, AccessRead | AccessAtomic);
  StackID creation_stack_id;
  TraceMutexUnlock(thr, addr);
  thr->mset.Del(addr);
  bool report_bad_unlock = false;
  int rec = 0;
  {
    SlotLocker locker(thr);
    auto s = ctx->metamap.GetSyncOrCreate(thr, pc, addr, true);
    bool released = false;
    {
      Lock lock(&s->mtx);
      creation_stack_id = s->creation_stack_id;
      if (!SANITIZER_GO && (s->recursion == 0 || s->owner_tid != thr->tid)) {
        if (flags()->report_mutex_bugs && !s->IsFlagSet(MutexFlagBroken)) {
          s->SetFlags(MutexFlagBroken);
          report_bad_unlock = true;
        }
      } else {
        rec = (flagz & MutexFlagRecursiveUnlock) ? s->recursion : 1;
        s->recursion -= rec;
        if (s->recursion == 0) {
          s->owner_tid = kInvalidTid;
          if (!thr->ignore_sync) {
            thr->clock.ReleaseStore(&s->clock);
            released = true;
          }
        }
      }
      if (common_flags()->detect_deadlocks && s->recursion == 0 &&
          !report_bad_unlock) {
        Callback cb(thr, pc);
        ctx->dd->MutexBeforeUnlock(&cb, &s->dd, true);
      }
    }
    if (released)
      IncrementEpoch(thr);
  }
  if (report_bad_unlock)
    ReportMutexMisuse(thr, pc, ReportTypeMutexBadUnlock, addr,
                      creation_stack_id);
  if (common_flags()->detect_deadlocks && !report_bad_unlock) {
    Callback cb(thr, pc);
    ReportDeadlock(thr, pc, ctx->dd->GetReport(&cb));
  }
  return rec;
}

void MutexPreReadLock(ThreadState *thr, uptr pc, uptr addr, u32 flagz) {
  DPrintf("#%d: MutexPreReadLock %zx flagz=0x%x\n", thr->tid, addr, flagz);
  if ((flagz & MutexFlagTryLock) || !common_flags()->detect_deadlocks)
    return;
  Callback cb(thr, pc);
  {
    SlotLocker locker(thr);
    auto s = ctx->metamap.GetSyncOrCreate(thr, pc, addr, true);
    ReadLock lock(&s->mtx);
    s->UpdateFlags(flagz);
    ctx->dd->MutexBeforeLock(&cb, &s->dd, false);
  }
  ReportDeadlock(thr, pc, ctx->dd->GetReport(&cb));
}

void MutexPostReadLock(ThreadState *thr, uptr pc, uptr addr, u32 flagz) {
  DPrintf("#%d: MutexPostReadLock %zx flagz=0x%x\n", thr->tid, addr, flagz);
  if (pc && IsAppMem(addr))
    MemoryAccess(thr, pc, addr, 1, AccessRead | AccessAtomic);
  SlotLocker locker(thr);
  auto s = ctx->metamap.GetSyncOrCreate(thr, pc, addr, true);
  StackID creation_stack_id = s->creation_stack_id;
  //!!! Every trace can now reset state, double check that it's ok and leaves
  //! state consistent, e.g. mutex set or release epoch.
  TraceMutexLock(thr, EventTypeRLock, pc, addr, creation_stack_id);
  thr->mset.Add(addr, creation_stack_id, false);
  bool report_bad_lock = false;
  bool pre_lock = false;
  {
    ReadLock lock(&s->mtx);
    s->UpdateFlags(flagz);
    if (s->owner_tid != kInvalidTid) {
      if (flags()->report_mutex_bugs && !s->IsFlagSet(MutexFlagBroken)) {
        s->SetFlags(MutexFlagBroken);
        report_bad_lock = true;
      }
    }
    if (!thr->ignore_sync)
      thr->clock.Acquire(s->clock);
    s->last_lock = thr->fast_state;
    if (common_flags()->detect_deadlocks) {
      pre_lock =
          (flagz & MutexFlagDoPreLockOnPostLock) && !(flagz & MutexFlagTryLock);
      Callback cb(thr, pc);
      if (pre_lock)
        ctx->dd->MutexBeforeLock(&cb, &s->dd, false);
      ctx->dd->MutexAfterLock(&cb, &s->dd, false, flagz & MutexFlagTryLock);
    }
  }
  if (report_bad_lock)
    ReportMutexMisuse(thr, pc, ReportTypeMutexBadReadLock, addr,
                      creation_stack_id);
  if (pre_lock  && common_flags()->detect_deadlocks) {
    Callback cb(thr, pc);
    ReportDeadlock(thr, pc, ctx->dd->GetReport(&cb));
  }
}

void MutexReadUnlock(ThreadState *thr, uptr pc, uptr addr) {
  DPrintf("#%d: MutexReadUnlock %zx\n", thr->tid, addr);
  if (pc && IsAppMem(addr))
    MemoryAccess(thr, pc, addr, 1, AccessRead | AccessAtomic);
  TraceMutexUnlock(thr, addr);
  thr->mset.Del(addr);
  StackID creation_stack_id;
  bool report_bad_unlock = false;
  {
    SlotLocker locker(thr);
    auto s = ctx->metamap.GetSyncOrCreate(thr, pc, addr, true);
    bool released = false;
    {
      Lock lock(&s->mtx);
      creation_stack_id = s->creation_stack_id;
      if (s->owner_tid != kInvalidTid) {
        if (flags()->report_mutex_bugs && !s->IsFlagSet(MutexFlagBroken)) {
          s->SetFlags(MutexFlagBroken);
          report_bad_unlock = true;
        }
      }
      if (!thr->ignore_sync) {
        thr->clock.Release(&s->read_clock);
        released = true;
      }
      if (common_flags()->detect_deadlocks && s->recursion == 0) {
        Callback cb(thr, pc);
        ctx->dd->MutexBeforeUnlock(&cb, &s->dd, false);
      }
    }
    if (released)
      IncrementEpoch(thr);
  }
  if (report_bad_unlock)
    ReportMutexMisuse(thr, pc, ReportTypeMutexBadReadUnlock, addr,
                      creation_stack_id);
  if (common_flags()->detect_deadlocks) {
    Callback cb(thr, pc);
    ReportDeadlock(thr, pc, ctx->dd->GetReport(&cb));
  }
}

void MutexReadOrWriteUnlock(ThreadState *thr, uptr pc, uptr addr) {
  DPrintf("#%d: MutexReadOrWriteUnlock %zx\n", thr->tid, addr);
  if (pc && IsAppMem(addr))
    MemoryAccess(thr, pc, addr, 1, AccessRead | AccessAtomic);
  TraceMutexUnlock(thr, addr);
  thr->mset.Del(addr);
  StackID creation_stack_id;
  bool report_bad_unlock = false;
  bool write = true;
  {
    SlotLocker locker(thr);
    auto s = ctx->metamap.GetSyncOrCreate(thr, pc, addr, true);
    bool released = false;
    {
      Lock lock(&s->mtx);
      creation_stack_id = s->creation_stack_id;
      if (s->owner_tid == kInvalidTid) {
        // Seems to be read unlock.
        write = false;
        if (!thr->ignore_sync) {
          thr->clock.Release(&s->read_clock);
          released = true;
        }
      } else if (s->owner_tid == thr->tid) {
        // Seems to be write unlock.
        CHECK_GT(s->recursion, 0);
        s->recursion--;
        if (s->recursion == 0) {
          s->owner_tid = kInvalidTid;
          if (!thr->ignore_sync) {
            thr->clock.ReleaseStore(&s->clock);
            released = true;
          }
        }
      } else if (!s->IsFlagSet(MutexFlagBroken)) {
        s->SetFlags(MutexFlagBroken);
        report_bad_unlock = true;
      }
      if (common_flags()->detect_deadlocks && s->recursion == 0) {
        Callback cb(thr, pc);
        ctx->dd->MutexBeforeUnlock(&cb, &s->dd, write);
      }
    }
    if (released)
      IncrementEpoch(thr);
  }
  if (report_bad_unlock)
    ReportMutexMisuse(thr, pc, ReportTypeMutexBadUnlock, addr,
                      creation_stack_id);
  if (common_flags()->detect_deadlocks) {
    Callback cb(thr, pc);
    ReportDeadlock(thr, pc, ctx->dd->GetReport(&cb));
  }
}

void MutexRepair(ThreadState *thr, uptr pc, uptr addr) {
  DPrintf("#%d: MutexRepair %zx\n", thr->tid, addr);
  SlotLocker locker(thr);
  auto s = ctx->metamap.GetSyncOrCreate(thr, pc, addr, true);
  Lock lock(&s->mtx);
  s->owner_tid = kInvalidTid;
  s->recursion = 0;
}

void MutexInvalidAccess(ThreadState *thr, uptr pc, uptr addr) {
  DPrintf("#%d: MutexInvalidAccess %zx\n", thr->tid, addr);
  SlotLocker locker(thr);
  auto s = ctx->metamap.GetSyncOrCreate(thr, pc, addr, true);
  StackID creation_stack_id = s ? s->creation_stack_id : kInvalidStackID;
  ReportMutexMisuse(thr, pc, ReportTypeMutexInvalidAccess, addr,
                    creation_stack_id);
}

void Acquire(ThreadState *thr, uptr pc, uptr addr) {
  DPrintf("#%d: Acquire %zx\n", thr->tid, addr);
  if (thr->ignore_sync)
    return;
  SlotLocker locker(thr);
  auto s = ctx->metamap.GetSyncIfExists(thr, pc, addr);
  if (!s || !s->clock)
    return;
  ReadLock lock(&s->mtx);
  thr->clock.Acquire(s->clock);
}

void Release(ThreadState* thr, uptr pc, uptr addr) {
  DPrintf("#%d: Release %zx\n", thr->tid, addr);
  if (thr->ignore_sync)
    return;
  SlotLocker locker(thr);
  {
    auto s = ctx->metamap.GetSyncOrCreate(thr, pc, addr, false);
    Lock lock(&s->mtx);
    thr->clock.Release(&s->clock);
  }
  IncrementEpoch(thr);
}

void ReleaseStore(ThreadState* thr, uptr pc, uptr addr) {
  DPrintf("#%d: ReleaseStore %zx\n", thr->tid, addr);
  if (thr->ignore_sync)
    return;
  SlotLocker locker(thr);
  {
    auto s = ctx->metamap.GetSyncOrCreate(thr, pc, addr, false);
    Lock lock(&s->mtx);
    thr->clock.ReleaseStore(&s->clock);
  }
  IncrementEpoch(thr);
}

void ReleaseStoreAcquire(ThreadState* thr, uptr pc, uptr addr) {
  DPrintf("#%d: ReleaseStoreAcquire %zx\n", thr->tid, addr);
  if (thr->ignore_sync)
    return;
  SlotLocker locker(thr);
  {
    auto s = ctx->metamap.GetSyncOrCreate(thr, pc, addr, false);
    Lock lock(&s->mtx);
    thr->clock.ReleaseStoreAcquire(&s->clock);
  }
  IncrementEpoch(thr);
}

void IncrementEpoch(ThreadState* thr) {
  DCHECK(!thr->ignore_sync);
  DCHECK(thr->slot_locked);
  Epoch epoch = EpochInc(thr->fast_state.epoch());
  if (!EpochOverflow(epoch)) {
    Sid sid = thr->fast_state.sid();
    thr->clock.Set(sid, epoch);
    thr->slot->clock.Set(sid, epoch);
    thr->fast_state.SetEpoch(epoch);
    TraceTime(thr);
  }
}

void AcquireGlobal(ThreadState* thr) {
  DPrintf("#%d: AcquireGlobal\n", thr->tid);
  if (thr->ignore_sync)
    return;
  SlotLocker locker(thr);
  for (auto& slot : ctx->slots)
    thr->clock.Set(slot.sid, slot.clock.Get(slot.sid));
}

#if !SANITIZER_GO
void AfterSleep(ThreadState *thr, uptr pc) {
  DPrintf("#%d: AfterSleep\n", thr->tid);
  if (thr->ignore_sync)
    return;
  thr->last_sleep_stack_id = CurrentStackId(thr, pc);
  thr->last_sleep_clock.Reset();
  SlotLocker locker(thr);
  for (auto& slot : ctx->slots)
    thr->last_sleep_clock.Set(slot.sid, slot.clock.Get(slot.sid));
}
#endif

void ReportDeadlock(ThreadState* thr, uptr pc, DDReport* r) {
  if (r == 0 || !ShouldReport(thr, ReportTypeDeadlock))
    return;
  ReportDesc rep;
  {
    ReportScope report_scope(thr);
    rep.typ = ReportTypeDeadlock;
    for (int i = 0; i < r->n; i++) {
      rep.AddMutex(r->loop[i].mtx_ctx0, r->loop[i].stk[0]);
      rep.AddUniqueTid(r->loop[i].thr_ctx);
      if (r->loop[i].thr_ctx != kInvalidTid)
        rep.AddThread(r->loop[i].thr_ctx);
    }
    uptr dummy_pc = 0x42;
    for (int i = 0; i < r->n; i++) {
      for (int j = 0; j < (flags()->second_deadlock_stack ? 2 : 1); j++) {
        StackID stk = r->loop[i].stk[j];
        if (stk != kInvalidStackID) {
          rep.AddStack(StackDepotGet(stk), true);
        } else {
          // Sometimes we fail to extract the stack trace (FIXME: investigate),
          // but we should still produce some stack trace in the report.
          rep.AddStack(StackTrace(&dummy_pc, 1), true);
        }
      }
    }
  }
  OutputReport(thr, &rep);
}

void ReportDestroyLocked(ThreadState* thr, uptr pc, uptr addr, FastState last_lock,
                         StackID creation_stack_id) {
  MutexSet mset;
  //!!! this won't restore read lock stack because type == EventTypeLock.
  VarSizeStackTrace trace[2];
  ReportDesc rep;
  rep.typ = ReportTypeMutexDestroyLocked;
  {
    ReportScope report_scope(thr);
    Tid tid;
    if (!RestoreStack(EventTypeLock, last_lock.sid(), last_lock.epoch(), addr, 0, false,
                      false, false, &tid, &trace[1], &mset))
      return;
    rep.AddMutex(addr, creation_stack_id);
    ObtainCurrentStack(thr, pc, &trace[0]);
    rep.AddStack(trace[0], true);
    rep.AddStack(trace[1], true);
    rep.AddLocation(addr, 1);
  }
  OutputReport(thr, &rep);
}

void ReportMutexMisuse(ThreadState* thr, uptr pc, ReportType typ, uptr addr,
                       StackID creation_stack_id) {
  // In Go, these misuses are either impossible, or detected by std lib,
  // or false positives (e.g. unlock in a different thread).
  if (SANITIZER_GO)
    return;
  if (!ShouldReport(thr, typ))
    return;
  VarSizeStackTrace trace;
  ObtainCurrentStack(thr, pc, &trace);
  ReportDesc rep;
  {
    ReportScope report_scope(thr);
    rep.typ = typ;
    rep.AddMutex(addr, creation_stack_id);
    rep.AddStack(trace, true);
    rep.AddLocation(addr, 1);
  }
  OutputReport(thr, &rep);
}

}  // namespace __tsan

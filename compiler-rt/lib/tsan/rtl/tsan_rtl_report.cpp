//===-- tsan_rtl_report.cpp -----------------------------------------------===//
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

#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_placement_new.h"
#include "sanitizer_common/sanitizer_stackdepot.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_stacktrace.h"
#include "tsan_platform.h"
#include "tsan_rtl.h"
#include "tsan_suppressions.h"
#include "tsan_symbolize.h"
#include "tsan_report.h"
#include "tsan_sync.h"
#include "tsan_mman.h"
#include "tsan_flags.h"

namespace __tsan {

// Can be overriden by an application/test to intercept reports.
#ifdef TSAN_EXTERNAL_HOOKS
bool OnReport(const ReportDesc *rep, bool suppressed);
#else
SANITIZER_WEAK_CXX_DEFAULT_IMPL
bool OnReport(const ReportDesc *rep, bool suppressed) {
  (void)rep;
  return suppressed;
}
#endif

SANITIZER_WEAK_DEFAULT_IMPL
void __tsan_on_report(const ReportDesc *rep) {
  (void)rep;
}

bool InternalFrame(const char* func) {
  static const char* frames[] = {
      "ScopedInterceptor",
      "EnableIgnores",
      "__tsan::ScopedInterceptor",
      "__sanitizer::StackTrace",
  };
  for (auto frame : frames) {
    if (!internal_strncmp(func, frame, internal_strlen(frame)))
      return true;
  }
  return false;
}

static SymbolizedStack* StackStripMain(SymbolizedStack* frames) {
  for (; frames && frames->info.function; frames = frames->next) {
    // Remove top inlined frames from our interceptors.
    if (!InternalFrame(frames->info.function))
      break;
  }
  SymbolizedStack *last_frame = nullptr;
  SymbolizedStack *last_frame2 = nullptr;
  for (SymbolizedStack *cur = frames; cur; cur = cur->next) {
    last_frame2 = last_frame;
    last_frame = cur;
  }

  if (last_frame2 == 0)
    return frames;
#if !SANITIZER_GO
  const char *last = last_frame->info.function;
  const char *last2 = last_frame2->info.function;
  // Strip frame above 'main'
  if (last2 && 0 == internal_strcmp(last2, "main")) {
    last_frame->ClearAll();
    last_frame2->next = nullptr;
  // Strip our internal thread start routine.
  } else if (last && 0 == internal_strcmp(last, "__tsan_thread_start_func")) {
    last_frame->ClearAll();
    last_frame2->next = nullptr;
  // Strip global ctors init.
  } else if (last && (0 == internal_strcmp(last, "__do_global_ctors_aux") ||
                      0 == internal_strcmp(last, "__libc_csu_init"))) {
    last_frame->ClearAll();
    last_frame2->next = nullptr;
  // If both are 0, then we probably just failed to symbolize.
  } else if (last || last2) {
    // Ensure that we recovered stack completely. Trimmed stack
    // can actually happen if we do not instrument some code,
    // so it's only a debug print. However we must try hard to not miss it
    // due to our fault.
    DPrintf("Bottom stack frame is missed\n");
  }
#else
  // The last frame always point into runtime (gosched0, goexit0, runtime.main).
  last_frame->ClearAll();
  last_frame2->next = nullptr;
#endif
  return frames;
}

static SymbolizedStack* SymbolizeStack(StackTrace trace) {
  if (trace.size == 0)
    return nullptr;
  SymbolizedStack *top = nullptr;
  for (uptr si = 0; si < trace.size; si++) {
    const uptr pc = trace.trace[si];
    uptr pc1 = pc;
    // We obtain the return address, but we're interested in the previous
    // instruction.
    if ((pc & kExternalPCBit) == 0)
      pc1 = StackTrace::GetPreviousInstructionPc(pc);
    SymbolizedStack *ent = SymbolizeCode(pc1);
    CHECK_NE(ent, 0);
    SymbolizedStack *last = ent;
    while (last->next) {
      last->info.address = pc;  // restore original pc for report
      last = last->next;
    }
    last->info.address = pc;  // restore original pc for report
    last->next = top;
    top = ent;
  }
  return StackStripMain(top);
}

void PrintStack(StackTrace stack) {
  PrintStack(SymbolizeStack(stack));
}

void PrintStack(StackID id) {
  PrintStack(StackDepotGet(id));
}

bool ShouldReport(ThreadState *thr, ReportType typ) {
  if (!flags()->report_bugs || thr->suppress_reports)
    return false;
  switch (typ) {
    case ReportTypeSignalUnsafe:
      return flags()->report_signal_unsafe;
    case ReportTypeThreadLeak:
#if !SANITIZER_GO
      // It's impossible to join phantom threads
      // in the child after fork.
      if (ctx->after_multithreaded_fork)
        return false;
#endif
      return flags()->report_thread_leaks;
    case ReportTypeMutexDestroyLocked:
      return flags()->report_destroy_locked;
    default:
      return true;
  }
}

ReportScope::ReportScope(ThreadState* thr)
    : slot_locker_(thr, true)
    , registry_lock_(&ctx->thread_registry)
    , slots_lock_(&ctx->slot_mtx) {
}

#if !SANITIZER_GO
static bool IsInStackOrTls(ThreadContextBase *tctx_base, void *arg) {
  uptr addr = (uptr)arg;
  ThreadContext *tctx = static_cast<ThreadContext*>(tctx_base);
  if (tctx->status != ThreadStatusRunning)
    return false;
  ThreadState *thr = tctx->thr;
  CHECK(thr);
  return ((addr >= thr->stk_addr && addr < thr->stk_addr + thr->stk_size) ||
          (addr >= thr->tls_addr && addr < thr->tls_addr + thr->tls_size));
}

ThreadContext *IsThreadStackOrTls(uptr addr, bool *is_stack) {
  ctx->thread_registry.CheckLocked();
  ThreadContext* tctx =
      static_cast<ThreadContext*>(ctx->thread_registry.FindThreadContextLocked(
          IsInStackOrTls, (void*)addr));
  if (!tctx)
    return 0;
  ThreadState *thr = tctx->thr;
  CHECK(thr);
  *is_stack = (addr >= thr->stk_addr && addr < thr->stk_addr + thr->stk_size);
  return tctx;
}
#endif

static void SymbolizeStack(ReportStack* rep) {
  if (!rep || rep->stack.size == 0)
    return;
  CHECK(!rep->frames);
  rep->frames = SymbolizeStack(rep->stack);
}

void SymbolizeReport(ReportDesc* rep) {
  for (uptr i = 0; i < rep->stacks.Size(); i++)
    SymbolizeStack(rep->stacks[i]);
  for (uptr i = 0; i < rep->mops.Size(); i++)
    SymbolizeStack(&rep->mops[i]->stack);
  for (uptr i = 0; i < rep->locs.Size(); i++) {
    if (rep->locs[i]->type == ReportLocationInvalid)
      SymbolizeData(rep->locs[i]->global.start, rep->locs[i]);
    SymbolizeStack(&rep->locs[i]->stack);
  }
  for (uptr i = 0; i < rep->mutexes.Size(); i++)
    SymbolizeStack(&rep->mutexes[i]->stack);
  for (uptr i = 0; i < rep->threads.Size(); i++)
    SymbolizeStack(&rep->threads[i]->stack);
  SymbolizeStack(rep->sleep);
}

uptr RestoreAddr(uptr addr) {
  static_assert(kCompressedAddrBits == 44,
                "check this function if kCompressedAddrBits changes");
#if SANITIZER_GO
  return addr;
#else
  constexpr uptr kRegionIndicator = 0x0e0000000000ull;
  constexpr uptr kHighMask = 0xf00000000000ull;
  constexpr uptr ranges[] = {Mapping::kLoAppMemBeg, Mapping::kHiAppMemBeg,
                             Mapping::kHeapMemBeg, Mapping::kMidAppMemBeg,
                             Mapping::kMidAppMemEnd};
  for (auto range : ranges) {
    if ((addr & kRegionIndicator) == (range & kRegionIndicator))
      return addr | (range & kHighMask);
  }
  Printf("ThreadSanitizer: failed to restore address %p\n", addr);
  ctx->nreported++; //!!!
  return addr;
#endif
}

template <typename Func>
void TraceReplay(Trace* trace, TracePart* last, Event* last_pos, Sid sid, Epoch epoch,
                 Func f) {
  TracePart* part = trace->parts.Front();
  Sid ev_sid = kFreeSid;
  Epoch ev_epoch = kEpochOver;
  for (;;) {
    DCHECK_EQ(part->trace, trace);
    // Note: an event can't start in the last element.
    // Since an event can take up to 2 elements,
    // we ensure we have at least 2 before adding an event.
    Event* end = &part->events[TracePart::kSize - 1];
    if (part == last)
      end = last_pos;
    for (Event* evp = &part->events[0]; evp < end; evp++) {
      Event* evp0 = evp;
      if (!evp->is_access && !evp->is_func) {
        switch (evp->type) {
        case EventTypeTime: {
          auto ev = reinterpret_cast<EventTime*>(evp);
          ev_sid = static_cast<Sid>(ev->sid);
          ev_epoch = static_cast<Epoch>(ev->epoch);
          if (ev_sid == sid && ev_epoch > epoch)
            return;
          break;
        }
        case EventTypeAccessExt:
          [[fallthrough]];
        case EventTypeAccessRange:
          [[fallthrough]];
        case EventTypeLock:
          [[fallthrough]];
        case EventTypeRLock:
          evp++;
        }
      }
      CHECK_NE(ev_sid, kFreeSid);
      CHECK_NE(ev_epoch, kEpochOver);
      f(ev_sid, ev_epoch, evp0);
    }
    if (part == last)
      return;
    part = trace->parts.Next(part);
    CHECK(part);
  }
  CHECK(0);
}

bool RestoreStack(EventType type, Sid sid, Epoch epoch, uptr addr, uptr size,
                  bool isRead, bool isAtomic, bool isFreed, Tid* ptid,
                  VarSizeStackTrace* stk, MutexSet* pmset, uptr* tag) {
  // This function restores stack trace and mutex set for the thread/epoch.
  // It does so by getting stack trace and mutex set at the beginning of
  // trace part, and then replaying the trace till the given epoch.
  DPrintf2("RestoreStack: sid=%u@%u addr=0x%zx/%zu type=%u/%u/%u\n",
           static_cast<u32>(sid), static_cast<u32>(epoch), addr, size, isRead,
           isAtomic, isFreed);

  ctx->slot_mtx.CheckLocked(); // needed to prevent part recycling
  ctx->thread_registry.CheckLocked();
  TidSlot* slot = &ctx->slots[static_cast<uptr>(sid)];
  Tid tid = kInvalidTid;
  //!!! do we need to lock the slot? what protects journal?
  for (uptr i = 0; i < slot->journal.Size(); i++) {
    if (i == slot->journal.Size() - 1 || slot->journal[i + 1].epoch > epoch) {
      tid = slot->journal[i].tid;
      break;
    }
  }
  if (tid == kInvalidTid)
    return false;
  *ptid = tid;
  ThreadContext* tctx =
      static_cast<ThreadContext*>(ctx->thread_registry.GetThreadLocked(tid));
  Trace* trace = &tctx->trace;
  TracePart* first_part;
  TracePart* last_part;
  Event* last_pos;
  {
    Lock lock(&trace->mtx);
    first_part = trace->parts.Front();
    last_part = trace->parts.Back();
    last_pos = trace->final_pos;
    if (tctx->thr)
      last_pos = (Event*)atomic_load_relaxed(&tctx->thr->trace_pos);
  }
  //!!! what happens if epoch has changed from race discovery?
  //!!! can startStack and trace be inconsistent
  if (!first_part)
    return false;
  Vector<uptr> stack;
  uptr pos = first_part->start_stack.size;
  stack.Resize(pos + 64);
  for (uptr i = 0; i < pos; i++) {
    uptr pc = first_part->start_stack.trace[i];
    stack[i] = pc;
    DPrintf2("  #%02zu: pc=%zx\n", i, pc);
  }
  uptr prev_pc = first_part->prev_pc;
  MutexSet mset = first_part->start_mset;
  bool found = false;
  TraceReplay(
      trace, last_part, last_pos, sid, epoch, [&](Sid ev_sid, Epoch ev_epoch, Event* evp) {
        bool match = ev_sid == sid && ev_epoch == epoch;
        if (evp->is_access) {
          if (evp->type == 0 && evp->_ == 0) // NopEvent
            return;
          auto ev = reinterpret_cast<EventAccess*>(evp);
          //!!! also check access size and type (read/atomic).
          uptr evAddr = RestoreAddr(ev->addr);
          uptr evSize = 1 << ev->sizeLog;
          uptr evPC = prev_pc + ev->pcDelta - (1 << (EventAccess::kPCBits - 1));
          prev_pc = evPC;
          DPrintf2("  Access: pc=0x%zx addr=0x%llx/%llu type=%llu/%llu\n", evPC,
                   evAddr, evSize, ev->isRead, ev->isAtomic);
          if (match && type == EventTypeAccessExt && addr >= evAddr &&
              addr + size <= evAddr + evSize && isRead == ev->isRead &&
              isAtomic == ev->isAtomic && !isFreed) {
            DPrintf2("    MATCHED\n");
            stack[pos] = evPC;
            stk->Init(&stack[0], pos + 1);
            *pmset = mset;
            found = true;
          }
          return;
        }
        if (evp->is_func) {
          auto ev = reinterpret_cast<EventFunc*>(evp);
          if (ev->pc) {
            DPrintf2("  FuncEnter: pc=0x%zx\n", ev->pc);
            if (stack.Size() < pos + 2)
              stack.Resize(pos + 2);
            stack[pos++] = ev->pc;
          } else {
            DPrintf2("  FuncExit\n");
            // Note: we may remember a truncated stack trace in
            // the trace part header, then we can have func exit
            // events that exit from non-existent frames.
            if (pos > 0)
              pos--;
          }
          return;
        }
        switch (evp->type) {
        case EventTypeAccessExt: {
          auto ev = reinterpret_cast<EventAccessExt*>(evp);
          uptr evAddr = RestoreAddr(ev->addr);
          uptr evSize = 1 << ev->sizeLog;
          prev_pc = ev->pc;
          DPrintf2(
              "  AccessExt: pc=0x%zx addr=0x%llx/%llu type=%llu/%llu\n",
              ev->pc, evAddr, evSize, ev->isRead, ev->isAtomic);
          //!!! also check access size and type (read/atomic).
          if (match && type == EventTypeAccessExt && addr >= evAddr &&
              addr + size <= evAddr + evSize && isRead == ev->isRead &&
              isAtomic == ev->isAtomic && !isFreed) {
            DPrintf2("    MATCHED\n");
            stack[pos] = ev->pc;
            stk->Init(&stack[0], pos + 1);
            *pmset = mset;
            found = true;
          }
          break;
        }
        case EventTypeAccessRange: {
          auto ev = reinterpret_cast<EventAccessRange*>(evp);
          uptr evAddr = RestoreAddr(ev->addr);
          uptr evSize = (ev->sizeHi << EventAccessRange::kSizeLoBits) + ev->sizeLo;
          uptr ev_pc = RestoreAddr(ev->pc);
          prev_pc = ev_pc;
          DPrintf2(
              "  AccessRange: pc=0x%zx addr=0x%llx/%llu type=%llu/%llu\n",
              ev_pc, evAddr, evSize, ev->isRead, ev->isFreed);
          //!!! also check access size and type (read/atomic).
          if (match && type == EventTypeAccessExt && addr >= evAddr &&
              addr + size <= evAddr + evSize && isRead == ev->isRead &&
              !isAtomic && isFreed == ev->isFreed) {
            DPrintf2("    MATCHED\n");
            stack[pos] = ev_pc;
            stk->Init(&stack[0], pos + 1);
            *pmset = mset;
            found = true;
          }
          break;
        }
        case EventTypeLock:
          [[fallthrough]];
        case EventTypeRLock: {
          auto ev = reinterpret_cast<EventLock*>(evp);
          bool isWrite = ev->type == EventTypeLock;
          uptr evAddr = RestoreAddr(ev->addr);
          uptr evPC = RestoreAddr(ev->pc);
          StackID stackID = static_cast<StackID>(
              (ev->stackIDHi << EventLock::kStackIDLoBits) + ev->stackIDLo);
          DPrintf2("  Lock: pc=0x%zx addr=0x%llx stack=%u write=%d\n", evPC,
                   evAddr, stackID, isWrite);
          mset.Add(evAddr, stackID, isWrite);
          if (match && type == EventTypeLock && addr == evAddr) {
            DPrintf2("    MATCHED\n");
            stack[pos] = evPC;
            stk->Init(&stack[0], pos + 1);
            *pmset = mset;
            found = true;
          }
          break;
        }
        case EventTypeUnlock: {
          auto ev = reinterpret_cast<EventUnlock*>(evp);
          uptr evAddr = RestoreAddr(ev->addr);
          DPrintf2("  Unlock: addr=0x%llx\n", evAddr);
          mset.Del(evAddr);
          break;
        }
        }
      });
  ExtractTagFromStack(stk, tag);
  return found;
}

static bool FindRacyStacks(const RacyStacks &hash) {
  for (uptr i = 0; i < ctx->racy_stacks.Size(); i++) {
    if (hash == ctx->racy_stacks[i]) {
      VPrintf(2, "ThreadSanitizer: suppressing report as doubled (stack)\n");
      return true;
    }
  }
  return false;
}

static bool HandleRacyStacks(ThreadState *thr, VarSizeStackTrace traces[2]) {
  if (!flags()->suppress_equal_stacks)
    return false;
  RacyStacks hash;
  hash.hash[0] = md5_hash(traces[0].trace, traces[0].size * sizeof(uptr));
  hash.hash[1] = md5_hash(traces[1].trace, traces[1].size * sizeof(uptr));
  {
    ReadLock lock(&ctx->racy_mtx);
    if (FindRacyStacks(hash))
      return true;
  }
  Lock lock(&ctx->racy_mtx);
  if (FindRacyStacks(hash))
    return true;
  ctx->racy_stacks.PushBack(hash);
  return false;
}

bool RacyStacks::operator==(const RacyStacks& other) const {
  if (hash[0] == other.hash[0] && hash[1] == other.hash[1])
    return true;
  if (hash[0] == other.hash[1] && hash[1] == other.hash[0])
    return true;
  return false;
}

static bool FindRacyAddress(const RacyAddress &ra0) {
  for (uptr i = 0; i < ctx->racy_addresses.Size(); i++) {
    RacyAddress ra2 = ctx->racy_addresses[i];
    uptr maxbeg = max(ra0.addr_min, ra2.addr_min);
    uptr minend = min(ra0.addr_max, ra2.addr_max);
    if (maxbeg < minend) {
      VPrintf(2, "ThreadSanitizer: suppressing report as doubled (addr)\n");
      return true;
    }
  }
  return false;
}

static bool HandleRacyAddress(ThreadState *thr, uptr addr_min, uptr addr_max) {
  if (!flags()->suppress_equal_addresses)
    return false;
  RacyAddress ra0 = {addr_min, addr_max};
  {
    ReadLock lock(&ctx->racy_mtx);
    if (FindRacyAddress(ra0))
      return true;
  }
  Lock lock(&ctx->racy_mtx);
  if (FindRacyAddress(ra0))
    return true;
  ctx->racy_addresses.PushBack(ra0);
  return false;
}

struct ExternalCallbackScope : ScopedIgnoreInterceptors {
  ExternalCallbackScope(ThreadState* thr, ReportDesc* rep)
      : thr_(thr) {
    ThreadIgnoreBegin(thr_, 0);
    CHECK_EQ(thr_->current_report, nullptr);
    thr_->current_report = rep;
  }

  ~ExternalCallbackScope() {
    ThreadIgnoreEnd(thr_);
    thr_->current_report = nullptr;
  }

  ThreadState* thr_;
};

bool OutputReport(ThreadState* thr, ReportDesc* rep) {
  // These should have been checked in ShouldReport.
  // It's too late to check them here, we have already taken locks.
  CHECK(flags()->report_bugs);
  CHECK(!thr->suppress_reports);
  SlotUnlocker unlocker(thr);
  CheckNoLocks();
  ScopedErrorReportLock error_lock;
  ExternalCallbackScope scope(thr, rep);
  SymbolizeReport(rep);
  atomic_store_relaxed(&ctx->last_symbolize_time_ns, NanoTime());
  Suppression *supp = 0;
  uptr pc_or_addr = 0;
  for (uptr i = 0; pc_or_addr == 0 && i < rep->mops.Size(); i++)
    pc_or_addr = IsSuppressed(rep->typ, &rep->mops[i]->stack, &supp);
  for (uptr i = 0; pc_or_addr == 0 && i < rep->stacks.Size(); i++)
    pc_or_addr = IsSuppressed(rep->typ, rep->stacks[i], &supp);
  for (uptr i = 0; pc_or_addr == 0 && i < rep->threads.Size(); i++)
    pc_or_addr = IsSuppressed(rep->typ, &rep->threads[i]->stack, &supp);
  for (uptr i = 0; pc_or_addr == 0 && i < rep->locs.Size(); i++)
    pc_or_addr = IsSuppressed(rep->typ, rep->locs[i], &supp);
  if (pc_or_addr != 0) {
    Lock lock(&ctx->fired_suppressions_mtx);
    FiredSuppression s = {rep->typ, pc_or_addr, supp};
    ctx->fired_suppressions.push_back(s);
  }
  if (OnReport(rep, pc_or_addr != 0))
    return false;
  PrintReport(rep);
  __tsan_on_report(rep);
  ctx->nreported++;
  if (flags()->halt_on_error)
    Die();
  return true;
}

bool IsFiredSuppression(Context *ctx, ReportType type, StackTrace trace) {
  ReadLock lock(&ctx->fired_suppressions_mtx);
  for (uptr k = 0; k < ctx->fired_suppressions.size(); k++) {
    if (ctx->fired_suppressions[k].type != type)
      continue;
    for (uptr j = 0; j < trace.size; j++) {
      FiredSuppression *s = &ctx->fired_suppressions[k];
      if (trace.trace[j] == s->pc_or_addr) {
        if (s->supp)
          atomic_fetch_add(&s->supp->hit_count, 1, memory_order_relaxed);
        return true;
      }
    }
  }
  return false;
}

static bool IsFiredSuppression(Context *ctx, ReportType type, uptr addr) {
  ReadLock lock(&ctx->fired_suppressions_mtx);
  for (uptr k = 0; k < ctx->fired_suppressions.size(); k++) {
    if (ctx->fired_suppressions[k].type != type)
      continue;
    FiredSuppression *s = &ctx->fired_suppressions[k];
    if (addr == s->pc_or_addr) {
      if (s->supp)
        atomic_fetch_add(&s->supp->hit_count, 1, memory_order_relaxed);
      return true;
    }
  }
  return false;
}

void ReportRace(ThreadState* thr, RawShadow* shadow_mem, Shadow cur,
                Shadow old, AccessType typ) {
  VPrintf(1, "#%d: ReportRace\n", thr->tid);
  if (!ShouldReport(thr, ReportTypeRace))
    return;
  if (!flags()->report_atomic_races &&
      (cur.IsAtomic() || old.IsAtomic()) &&
      !old.IsFree() && !(typ & AccessFree))
    return;

  const uptr kMop = 2;
  Shadow s[kMop] = {cur, old};
  uptr addr = ShadowToMem((uptr)shadow_mem);
  uptr addr0 = addr + s[0].addr0();
  uptr addr1 = addr + s[1].addr0();
  uptr end0 = addr0 + s[0].size();
  uptr end1 = addr1 + s[1].size();
  uptr addr_min = min(addr0, addr1);
  uptr addr_max = max(end0, end1);
  if (IsExpectedReport(addr_min, addr_max - addr_min))
    return;
  if (HandleRacyAddress(thr, addr_min, addr_max))
    return;

  VarSizeStackTrace traces[kMop];
  ReportDesc rep;
  rep.typ = ReportTypeRace;
  if ((typ & AccessVptr) && s[1].IsFree())
    rep.typ = ReportTypeVptrUseAfterFree;
  else if (typ & AccessVptr)
    rep.typ = ReportTypeVptrRace;
  else if (s[1].IsFree())
    rep.typ = ReportTypeUseAfterFree;

  if (IsFiredSuppression(ctx, rep.typ, addr))
    return;

  Tid tids[kMop] = {thr->tid, kInvalidTid};
  uptr tags[kMop] = {kExternalTagNone, kExternalTagNone};

  ObtainCurrentStack(thr, thr->trace_prev_pc, &traces[0], &tags[0]);
  if (IsFiredSuppression(ctx, rep.typ, traces[0]))
    return;

  // MutexSet is too large to live on stack.
  Vector<u64> mset_buffer;
  mset_buffer.Resize(sizeof(MutexSet) / sizeof(u64) + 1);
  MutexSet* mset1 = new (&mset_buffer[0]) MutexSet();
  MutexSet* mset[kMop] = {&thr->mset, mset1};

  //!!! re slots_mtx: how atomic is this? if we detect a bug, then reset
  //! happens, traces reset,
  // then we try to report and fail to restore traces
  {
    ReportScope report_scope(thr);
    if (!RestoreStack(EventTypeAccessExt, s[1].sid(), s[1].epoch(), addr1,
                      s[1].size(), s[1].IsRead(), s[1].IsAtomic(),
                      s[1].IsFree(), &tids[1], &traces[1], mset[1], &tags[1]))
      return;
    if (IsFiredSuppression(ctx, rep.typ, traces[1]))
      return;

    if (HandleRacyStacks(thr, traces))
      return;

    // If any of the accesses has a tag, treat this as an "external" race.
    for (uptr i = 0; i < kMop; i++) {
      if (tags[i] != kExternalTagNone) {
        rep.typ = ReportTypeExternalRace;
        rep.tag = tags[i];
        break;
      }
    }

    for (uptr i = 0; i < kMop; i++)
      rep.AddMemoryAccess(addr, tags[i], s[i], tids[i], traces[i], mset[i]);

    for (uptr i = 0; i < kMop; i++) {
      if (tids[i] == kInvalidTid) //!!! should not happen
        continue;
      ThreadContext* tctx = static_cast<ThreadContext*>(
          ctx->thread_registry.GetThreadLocked(tids[i]));
      rep.AddThread(tctx);
    }

    rep.AddLocation(addr_min, addr_max - addr_min);

#if !SANITIZER_GO
    if (!s[1].IsFree() &&
        s[1].epoch() <= thr->last_sleep_clock.Get(s[1].sid()))
      rep.AddSleep(thr->last_sleep_stack_id);
#endif
  }
  OutputReport(thr, &rep);
}

void PrintCurrentStack(ThreadState *thr, uptr pc) {
  VarSizeStackTrace trace;
  ObtainCurrentStack(thr, pc, &trace);
  PrintStack(SymbolizeStack(trace));
}

// Always inlining PrintCurrentStackSlow, because LocatePcInTrace assumes
// __sanitizer_print_stack_trace exists in the actual unwinded stack, but
// tail-call to PrintCurrentStackSlow breaks this assumption because
// __sanitizer_print_stack_trace disappears after tail-call.
// However, this solution is not reliable enough, please see dvyukov's comment
// http://reviews.llvm.org/D19148#406208
// Also see PR27280 comment 2 and 3 for breaking examples and analysis.
ALWAYS_INLINE USED void PrintCurrentStackSlow(uptr pc) {
#if !SANITIZER_GO
  uptr bp = GET_CURRENT_FRAME();
  auto ptrace = New<BufferedStackTrace>();
  ptrace->Unwind(pc, bp, nullptr, false);

  for (uptr i = 0; i < ptrace->size / 2; i++) {
    uptr tmp = ptrace->trace_buffer[i];
    ptrace->trace_buffer[i] = ptrace->trace_buffer[ptrace->size - i - 1];
    ptrace->trace_buffer[ptrace->size - i - 1] = tmp;
  }
  PrintStack(SymbolizeStack(*ptrace));
#endif
}

}  // namespace __tsan

using namespace __tsan;

extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE
void __sanitizer_print_stack_trace() {
  PrintCurrentStackSlow(StackTrace::GetCurrentPc());
}
}  // extern "C"

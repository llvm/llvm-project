//===-- tsan_rtl.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
// Main file (entry points) for the TSan run-time.
//===----------------------------------------------------------------------===//

#include "tsan_rtl.h"

#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_file.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_placement_new.h"
#include "sanitizer_common/sanitizer_stackdepot.h"
#include "sanitizer_common/sanitizer_symbolizer.h"
#include "tsan_defs.h"
#include "tsan_interface.h"
#include "tsan_mman.h"
#include "tsan_platform.h"
#include "tsan_suppressions.h"
#include "tsan_symbolize.h"
#include "ubsan/ubsan_init.h"

volatile int __tsan_resumed = 0;

extern "C" void __tsan_resume() {
  __tsan_resumed = 1;
}

namespace __tsan {

#if !SANITIZER_GO && !SANITIZER_MAC
__attribute__((tls_model("initial-exec")))
THREADLOCAL char cur_thread_placeholder[sizeof(ThreadState)] ALIGNED(64);
#endif
static char ctx_placeholder[sizeof(Context)] ALIGNED(64);
Context* ctx;

bool is_initialized;

// Can be overriden by a front-end.
#ifdef TSAN_EXTERNAL_HOOKS
bool OnFinalize(bool failed);
void OnInitialize();
#else
SANITIZER_WEAK_CXX_DEFAULT_IMPL
bool OnFinalize(bool failed) {
#if !SANITIZER_GO
  if (__tsan_on_finalize)
    return reinterpret_cast<int (*)(int)>(__tsan_on_finalize)(failed);
#endif
  return failed;
}

SANITIZER_WEAK_CXX_DEFAULT_IMPL
void OnInitialize() {
#if !SANITIZER_GO
  if (__tsan_on_initialize)
    return reinterpret_cast<void (*)(void)>(__tsan_on_initialize)();
#endif
}
#endif

TracePart* TracePartAlloc(ThreadState* thr) {
  TracePart* part = nullptr;
  {
    Lock lock(&ctx->slot_mtx);
    // We need at least 3 parts per thread, because we want to keep at last
    // 2 parts per thread that are not queued into ctx->trace_part_recycle.
    uptr max_parts = Max(3, flags()->history_size);
    Trace* trace = &thr->tctx->trace;
    if (trace->parts_allocated == max_parts || ctx->trace_part_slack != 0) {
      part = ctx->trace_part_recycle.PopFront();
      DPrintf("#%d: TracePartAlloc: part=%p\n", thr->tid, part);
      if (part && part->trace) {
        Trace* trace1 = part->trace;
        Lock trace_lock(&trace1->mtx);
        part->trace = nullptr;
        TracePart* part1 = trace1->parts.PopFront();
        CHECK_EQ(part, part1);
        if (trace1->parts_allocated > trace1->parts.Size()) {
          ctx->trace_part_slack += trace1->parts_allocated - trace1->parts.Size();
          trace1->parts_allocated = trace1->parts.Size();
        }
      }
    }
    if (trace->parts_allocated < max_parts) {
      trace->parts_allocated++;
      if (ctx->trace_part_slack)
        ctx->trace_part_slack--;
    }
    if (!part)
      ctx->trace_part_count++;
  }
  if (!part)
    part = new (MmapOrDie(sizeof(TracePart), "TracePart")) TracePart();
  return part;
}

void TracePartFree(TracePart* part) REQUIRES(ctx->slot_mtx) {
  DCHECK(part->trace);
  part->trace = nullptr;
  ctx->trace_part_recycle.PushFront(part);
}

void DoResetImpl(ThreadState* thr, uptr epoch) {
  ThreadRegistryLock lock0(&ctx->thread_registry);
  Lock lock1(&ctx->slot_mtx);
  DPrintf("#%d: DoReset epoch=%d\n", thr->tid, epoch);
  if (epoch)
    CHECK_EQ(ctx->global_epoch, epoch);
  ctx->global_epoch++;
  {
    for (u32 i = ctx->thread_registry.NumThreadsLocked(); i--;) {
      ThreadContext* tctx =
          (ThreadContext*)ctx->thread_registry.GetThreadLocked(
              static_cast<Tid>(i));
      // Potentially we could purge all ThreadStatusDead threads from the
      // registry. Since we reset all shadow, they can't race with anything
      // anymore. However, their tid's can still be stored in some aux places
      // (e.g. tid of thread that created something).
      auto trace = &tctx->trace;
      Lock lock(&trace->mtx);
      bool attached = tctx->thr && tctx->thr->slot;
      auto parts = &trace->parts;
      bool local = false;
      while (!parts->Empty()) {
        auto part = parts->Front();
        local = local || part == trace->local_head;
        if (local)
          CHECK(!ctx->trace_part_recycle.Queued(part));
        else
          ctx->trace_part_recycle.Remove(part);
        if (attached && parts->Size() == 1) {
          //!!! reset thr->trace_pos to the end of the part to force it to switch
          break;
        }
        parts->Remove(part);
        TracePartFree(part);
      }
      CHECK_LE(parts->Size(), 1);
      trace->local_head = parts->Front();
      if (tctx->thr && !tctx->thr->slot) {
        atomic_store_relaxed(&tctx->thr->trace_pos, 0);
        tctx->thr->trace_prev_pc = 0;
      }
      if (trace->parts_allocated > trace->parts.Size()) {
        ctx->trace_part_slack += trace->parts_allocated - trace->parts.Size();
        trace->parts_allocated = trace->parts.Size();
      }
    }
  }
  while (ctx->slot_queue.PopFront()) {
  }
  for (auto slot = &ctx->slots[0]; slot < &ctx->slots[kSlotCount]; slot++) {
    slot->clock.Reset();
    slot->journal.Reset();
    slot->thr = nullptr;
    ctx->slot_queue.PushBack(slot);
  }

  DPrintf("Resetting shadow...\n");
  if (!MmapFixedNoReserve(ShadowBeg(), ShadowEnd() - ShadowBeg(), "shadow")) {
    Printf("failed to reset shadow memory\n");
    Die();
  }
  DPrintf("Resetting meta shadow...\n");
  ctx->metamap.ResetClocks();
}

// Clang does not understand locking all slots in the loop:
// error: expecting mutex 'slot.mtx' to be held at start of each loop
void DoReset(ThreadState* thr, uptr epoch) NO_THREAD_SAFETY_ANALYSIS {
  for (auto& slot : ctx->slots) {
    slot.mtx.Lock();
    if (UNLIKELY(epoch != 0 && epoch != ctx->global_epoch)) {
      // Epoch can't change once we've locked the first slot.
      CHECK_EQ(slot.sid, 0);
      slot.mtx.Unlock();
      return;
    }
  }
  DoResetImpl(thr, epoch);
  for (auto& slot : ctx->slots)
    slot.mtx.Unlock();
}

TidSlot* FindSlotAndLock(ThreadState* thr) ACQUIRE(thr->slot->mtx) NO_THREAD_SAFETY_ANALYSIS {
  CHECK(!thr->slot);
  for (;;) {
    TidSlot* slot;
    uptr epoch;
    {
      Lock lock(&ctx->slot_mtx);
      //!!! if there are too few of them, return null
      // otherwise threads can be constantly preempting each other when few slots left
      slot = ctx->slot_queue.PopFront();
      if (slot)
        ctx->slot_queue.PushBack(slot);
      epoch = ctx->global_epoch;
    }
    if (!slot) {
      DoReset(thr, epoch);
      continue;
    }
    slot->mtx.Lock();
    CHECK(!thr->slot_locked);
    thr->slot_locked = true;
    if (slot->thr) {
      DPrintf("#%d: preempting sid=%d tid=%d\n",
          thr->tid, (u32)slot->sid, slot->thr->tid);
      slot->clock = slot->thr->clock;
      slot->thr = nullptr;
    }
    if (slot->clock.Get(slot->sid) == kEpochLast) {
      {
        //!!! do this on the next iteration.
        Lock lock(&ctx->slot_mtx);
        if (ctx->slot_queue.Queued(slot))
          ctx->slot_queue.Remove(slot);
      }
      thr->slot_locked = false;
      slot->mtx.Unlock();
      continue;
    }
    return slot;
  }
}

//!!! needs to return a locked slot
void SlotAttachAndLock(ThreadState* thr) {
  //Lock lock(&ctx->slots_mtx);
  TidSlot* slot = FindSlotAndLock(thr);
  DPrintf("#%d: SlotAttach: slot=%u\n", thr->tid, slot->sid);
  CHECK(!slot->thr);
  CHECK(!thr->slot);
  slot->thr = thr;
  thr->slot = slot;
  Epoch epoch = EpochInc(slot->clock.Get(slot->sid));
  CHECK(!EpochOverflow(epoch));
  slot->clock.Set(slot->sid, epoch);
  if (thr->slot_epoch == ctx->global_epoch) {
    thr->clock.Acquire(&slot->clock);
  } else {
    thr->slot_epoch = ctx->global_epoch;
    thr->clock = slot->clock;
#if !SANITIZER_GO
    thr->last_sleep_stack_id = kInvalidStackID;
    thr->last_sleep_clock.Reset();
#endif
  }
  thr->fast_state.SetSid(slot->sid);
  thr->fast_state.SetEpoch(thr->clock.Get(slot->sid));
  slot->journal.PushBack({thr->tid, epoch});
}

void SlotDetachImpl(ThreadState* thr) {
  TidSlot* slot = thr->slot;
  thr->slot = nullptr;
  if (thr != slot->thr) {
    slot = nullptr; // we don't own the slot anymore
    if (thr->slot_epoch != ctx->global_epoch) {
      //!!! Do we need to hold trace->mtx here?
      auto parts = &thr->tctx->trace.parts;
      // The trace can be completely empty in an unlikely event
      // the thread is preempted right after it acquired the slot
      // in ThreadStart and did not trace any events yet.
      CHECK_LE(parts->Size(), 1);
      auto part = parts->PopFront();
      if (part) {
        Lock l(&ctx->slot_mtx);
        TracePartFree(part);
      }
      thr->tctx->trace.local_head = nullptr;
      atomic_store_relaxed(&thr->trace_pos, 0);
      thr->trace_prev_pc = 0;
    }
  } else {
    //!!! consider sending thr->slot as hint to FindSlotAndLock
    // so that it can remove it from the ctx->slot_queue right away.
    slot->thr = nullptr;
    //!!! do we need to update whole clock or only own element?
    //!!! maybe at least for exiting threads we can update only own elem?
    slot->clock = thr->clock;
  }
}

//!!! Do we now need SlotDetach at all?
// We could just leave it for future preemption?
void SlotDetach(ThreadState* thr) {
  TidSlot* slot = thr->slot;
  Lock lock(&slot->mtx);
  SlotDetachImpl(thr);
}

void SlotLock(ThreadState* thr) NO_THREAD_SAFETY_ANALYSIS {
  DCHECK(!thr->slot_locked);
  TidSlot* slot = thr->slot;
  slot->mtx.Lock();
  thr->slot_locked = true;
  if (LIKELY(thr == slot->thr && thr->fast_state.epoch() != kEpochLast))
    return;
  SlotDetachImpl(thr);
  thr->slot_locked = false;
  slot->mtx.Unlock();
  SlotAttachAndLock(thr);
}

void SlotUnlock(ThreadState* thr) {
  DCHECK(thr->slot_locked);
  thr->slot_locked = false;
  thr->slot->mtx.Unlock();
}

Context::Context()
    : initialized(), nreported(),
      thread_registry([](Tid tid) -> ThreadContextBase* {
        return new (Alloc(sizeof(ThreadContext))) ThreadContext(tid);
      }),
      racy_mtx(MutexTypeRacy), racy_stacks(), racy_addresses(),
      fired_suppressions_mtx(MutexTypeFired), slot_mtx(MutexTypeSlots) {
  fired_suppressions.reserve(8);
  for (uptr i = 0; i < ARRAY_SIZE(slots); i++) {
    TidSlot* slot = &slots[i];
    slot->sid = static_cast<Sid>(i);
    slot_queue.PushBack(slot);
  }
  global_epoch = 1;
}

TidSlot::TidSlot() : mtx(MutexTypeSlot) {
}

// The objects are allocated in TLS, so one may rely on zero-initialization.
ThreadState::ThreadState(Tid tid)
    // Do not touch these, rely on zero initialization,
    // they may be accessed before the ctor.
    // ignore_accesses()
    // ignore_interceptors()
    : tid(tid) {
#if !SANITIZER_GO
  shadow_stack_pos = shadow_stack;
  shadow_stack_end = shadow_stack + kShadowStackSize;
#else
  // Setup dynamic shadow stack.
  const int kInitStackSize = 8;
  shadow_stack = (uptr*)Alloc(kInitStackSize * sizeof(uptr));
  shadow_stack_pos = shadow_stack;
  shadow_stack_end = shadow_stack + kInitStackSize;
#endif
}

#if !SANITIZER_GO
void MemoryProfiler(u64 uptime) {
  if (ctx->memprof_fd == kInvalidFd)
    return;
  InternalMmapVector<char> buf(4096);
  WriteMemoryProfile(buf.data(), buf.size(), uptime);
  WriteToFile(ctx->memprof_fd, buf.data(), internal_strlen(buf.data()));
}

void InitializeMemoryProfiler() {
  ctx->memprof_fd = kInvalidFd;
  const char* fname = flags()->profile_memory;
  if (!fname || !fname[0])
    return;
  if (internal_strcmp(fname, "stdout") == 0) {
    ctx->memprof_fd = 1;
  } else if (internal_strcmp(fname, "stderr") == 0) {
    ctx->memprof_fd = 2;
  } else {
    InternalScopedString filename;
    filename.append("%s.%d", fname, (int)internal_getpid());
    ctx->memprof_fd = OpenFile(filename.data(), WrOnly);
    if (ctx->memprof_fd == kInvalidFd) {
      Printf("ThreadSanitizer: failed to open memory profile file '%s'\n",
           filename.data());
      return;
    }
  }
  MemoryProfiler(0);
  MaybeSpawnBackgroundThread();
}

static void* BackgroundThread(void* arg) {
  // This is a non-initialized non-user thread, nothing to see here.
  // We don't use ScopedIgnoreInterceptors, because we want ignores to be
  // enabled even when the thread function exits (e.g. during pthread thread
  // shutdown code).
  cur_thread_init()->ignore_interceptors++;
  const u64 flush_symbolizer = flags()->flush_symbolizer_ms * 1000 * 1000;
  const u64 start = NanoTime();

  //u64 last_flush = NanoTime();
  //uptr last_rss = 0;
  while (!atomic_load_relaxed(&ctx->stop_background_thread)) {
    internal_usleep(1000 * 1000);
    u64 now = NanoTime();

    MemoryProfiler(now - start);

    //!!! we may still want a periodic shadow flush,
    // otherwise we may consume more memory than v2

    // Flush symbolizer cache if requested.
    if (flush_symbolizer > 0) {
      u64 last = atomic_load_relaxed(&ctx->last_symbolize_time_ns);
      if (last && last + flush_symbolizer < now) {
        ScopedErrorReportLock lock;
        SymbolizerFlush();
        atomic_store_relaxed(&ctx->last_symbolize_time_ns, 0);
      }
    }
  }
  return nullptr;
}

static void StartBackgroundThread() {
  ctx->background_thread = internal_start_thread(&BackgroundThread, 0);
}

#ifndef __mips__
static void StopBackgroundThread() {
  atomic_store(&ctx->stop_background_thread, 1, memory_order_relaxed);
  internal_join_thread(ctx->background_thread);
  ctx->background_thread = 0;
}
#endif
#endif

void DontNeedShadowFor(uptr addr, uptr size) {
  ReleaseMemoryPagesToOS(MemToShadow(addr), MemToShadow(addr + size));
}

#if !SANITIZER_GO
void UnmapShadow(ThreadState* thr, uptr addr, uptr size) {
  if (size == 0)
    return;
  DontNeedShadowFor(addr, size);
  ScopedGlobalProcessor sgp;
  SlotLocker locker(thr, true);
  ctx->metamap.ResetRange(thr->proc(), addr, size);
}
#endif

void MapShadow(uptr addr, uptr size) {
  // Global data is not 64K aligned, but there are no adjacent mappings,
  // so we can get away with unaligned mapping.
  // CHECK_EQ(addr, addr & ~((64 << 10) - 1));  // windows wants 64K alignment
  const uptr kPageSize = GetPageSizeCached();
  uptr shadow_begin = RoundDownTo((uptr)MemToShadow(addr), kPageSize);
  uptr shadow_end = RoundUpTo((uptr)MemToShadow(addr + size), kPageSize);
  if (!MmapFixedSuperNoReserve(shadow_begin, shadow_end - shadow_begin,
                               "shadow"))
    Die();

  // Meta shadow is 2:1, so tread carefully.
  static bool data_mapped = false;
  static uptr mapped_meta_end = 0;
  uptr meta_begin = (uptr)MemToMeta(addr);
  uptr meta_end = (uptr)MemToMeta(addr + size);
  meta_begin = RoundDownTo(meta_begin, 64 << 10);
  meta_end = RoundUpTo(meta_end, 64 << 10);
  if (!data_mapped) {
    // First call maps data+bss.
    data_mapped = true;
    if (!MmapFixedSuperNoReserve(meta_begin, meta_end - meta_begin,
                                 "meta shadow"))
      Die();
  } else {
    // Mapping continous heap.
    // Windows wants 64K alignment.
    meta_begin = RoundDownTo(meta_begin, 64 << 10);
    meta_end = RoundUpTo(meta_end, 64 << 10);
    if (meta_end <= mapped_meta_end)
      return;
    if (meta_begin < mapped_meta_end)
      meta_begin = mapped_meta_end;
    if (!MmapFixedSuperNoReserve(meta_begin, meta_end - meta_begin,
                                 "meta shadow"))
      Die();
    mapped_meta_end = meta_end;
  }
  VPrintf(2, "mapped meta shadow for (%p-%p) at (%p-%p)\n", addr, addr + size,
          meta_begin, meta_end);
}

static void CheckShadowMapping() {
  uptr beg, end;
  for (int i = 0; GetUserRegion(i, &beg, &end); i++) {
    // Skip cases for empty regions (heap definition for architectures that
    // do not use 64-bit allocator).
    if (beg == end)
      continue;
    VPrintf(3, "checking shadow region %p-%p\n", beg, end);
    uptr prev = 0;
    for (uptr p0 = beg; p0 <= end; p0 += (end - beg) / 4) {
      for (int x = -(int)kShadowCell; x <= (int)kShadowCell; x += kShadowCell) {
        const uptr p = RoundDown(p0 + x, kShadowCell);
        if (p < beg || p >= end)
          continue;
        EventAccess ev;
        ev.addr = p;
        const uptr restored = RestoreAddr(ev.addr);
        CHECK_EQ(p, restored);
        const uptr s = MemToShadow(p);
        const uptr m = (uptr)MemToMeta(p);
        VPrintf(3, "  checking pointer %p: shadow=%p meta=%p\n", p, s, m);
        CHECK(IsAppMem(p));
        CHECK(IsShadowMem(s));
        CHECK_EQ(p, ShadowToMem(s));
        CHECK(IsMetaMem(m));
        if (prev) {
          // Ensure that shadow and meta mappings are linear within a single
          // user range. Lots of code that processes memory ranges assumes it.
          const uptr prev_s = MemToShadow(prev);
          const uptr prev_m = (uptr)MemToMeta(prev);
          CHECK_EQ(s - prev_s, (p - prev) * kShadowMultiplier);
          CHECK_EQ((m - prev_m) / kMetaShadowSize,
                   (p - prev) / kMetaShadowCell);
        }
        prev = p;
      }
    }
  }
}

#if !SANITIZER_GO
static void OnStackUnwind(const SignalContext& sig, const void*,
                          BufferedStackTrace* stack) {
  stack->Unwind(StackTrace::GetNextInstructionPc(sig.pc), sig.bp, sig.context,
                common_flags()->fast_unwind_on_fatal);
}

static void TsanOnDeadlySignal(int signo, void* siginfo, void* context) {
  HandleDeadlySignal(siginfo, context, static_cast<Tid>(GetTid()),
                     &OnStackUnwind, nullptr);
}
#endif

void CheckUnwind() {
  // There is high probability that interceptors will check-fail as well,
  // on the other hand there is no sense in processing interceptors
  // since we are going to die soon.
  ScopedIgnoreInterceptors ignore;
#if !SANITIZER_GO
  ThreadState* thr = cur_thread();
  thr->nomalloc = false;
  thr->ignore_sync++;
  thr->ignore_accesses++;
  atomic_store_relaxed(&thr->in_signal_handler, 0);
#endif
  PrintCurrentStackSlow(StackTrace::GetCurrentPc());
}

void Initialize(ThreadState *thr) {
  // Thread safe because done before all threads exist.
  if (is_initialized)
    return;
  is_initialized = true;
  // We are not ready to handle interceptors yet.
  ScopedIgnoreInterceptors ignore;
  SanitizerToolName = "ThreadSanitizer";
  // Install tool-specific callbacks in sanitizer_common.
  SetCheckUnwindCallback(CheckUnwind);

  ctx = new (ctx_placeholder) Context;
  const char* env_name = SANITIZER_GO ? "GORACE" : "TSAN_OPTIONS";
  const char* options = GetEnv(env_name);
  CacheBinaryName();
  CheckASLR();
  InitializeFlags(&ctx->flags, options, env_name);
  AvoidCVE_2016_2143();
  __sanitizer::InitializePlatformEarly();
  __tsan::InitializePlatformEarly();

#if !SANITIZER_GO
  // Re-exec ourselves if we need to set additional env or command line args.
  MaybeReexec();

  InitializeAllocator();
  ReplaceSystemMalloc();
#endif
  if (common_flags()->detect_deadlocks)
    ctx->dd = DDetector::Create(flags());
  Processor* proc = ProcCreate();
  ProcWire(proc, thr);
  InitializeInterceptors();
  CheckShadowMapping();
  InitializePlatform();
  InitializeMutex();
  InitializeDynamicAnnotations();
#if !SANITIZER_GO
  InitializeShadowMemory();
  InitializeAllocatorLate();
  InstallDeadlySignalHandlers(TsanOnDeadlySignal);
#endif
  // Setup correct file descriptor for error reports.
  __sanitizer_set_report_path(common_flags()->log_path);
  InitializeSuppressions();
#if !SANITIZER_GO
  InitializeLibIgnore();
#endif

  VPrintf(1, "***** Running under ThreadSanitizer v3 (pid %d) *****\n",
          (int)internal_getpid());

  // Initialize thread 0.
  Tid tid = ThreadCreate(nullptr, 0, 0, true);
  CHECK_EQ(tid, 0);
  ThreadStart(thr, tid, GetTid(), ThreadType::Regular);
#if TSAN_CONTAINS_UBSAN
  __ubsan::InitAsPlugin();
#endif

#if !SANITIZER_GO
  Symbolizer::LateInitialize();
  InitializeMemoryProfiler();
#endif
  ctx->initialized = true;

  if (flags()->stop_on_start) {
    Printf("ThreadSanitizer is suspended at startup (pid %d)."
           " Call __tsan_resume().\n",
           (int)internal_getpid());
    while (__tsan_resumed == 0) {
    }
  }

  OnInitialize();
}

void MaybeSpawnBackgroundThread() {
  // On MIPS, TSan initialization is run before
  // __pthread_initialize_minimal_internal() is finished, so we can not spawn
  // new threads.
#if !SANITIZER_GO && !defined(__mips__)
  static atomic_uint32_t bg_thread = {};
  if (atomic_load(&bg_thread, memory_order_relaxed) == 0 &&
      atomic_exchange(&bg_thread, 1, memory_order_relaxed) == 0) {
    StartBackgroundThread();
    SetSandboxingCallback(StopBackgroundThread);
  }
#endif
}

int Finalize(ThreadState* thr) {
  if (common_flags()->print_module_map == 1)
    DumpProcessMap();

  if (flags()->atexit_sleep_ms > 0 && ctx->thread_registry.RunningThreads() > 1)
    internal_usleep(u64(flags()->atexit_sleep_ms) * 1000);

  {
    // Wait for pending reports.
    ScopedErrorReportLock lock;
  }

#if !SANITIZER_GO
  if (Verbosity())
    AllocatorPrintStats();
#endif

  ThreadFinalize(thr);

  bool failed = false;
  if (ctx->nreported) {
    failed = true;
#if !SANITIZER_GO
    Printf("ThreadSanitizer: reported %d warnings\n", ctx->nreported);
#else
    Printf("Found %d data race(s)\n", ctx->nreported);
#endif
  }

  if (common_flags()->print_suppressions)
    PrintMatchedSuppressions();

  failed = OnFinalize(failed);

  return failed ? common_flags()->exitcode : 0;
}

#if !SANITIZER_GO
void ForkBefore(ThreadState* thr, uptr pc) {
  //!!! we probably should lock the current slot as well?
  ctx->thread_registry.Lock();
  ctx->slot_mtx.Lock();
  ScopedErrorReportLock::Lock();
  // Suppress all reports in the pthread_atfork callbacks.
  // Reports will deadlock on the report_mtx.
  // We could ignore interceptors and sync operations as well,
  // but so far it's unclear if it will do more good or harm.
  // Unnecessarily ignoring things can lead to false positives later.
  thr->suppress_reports++;
}

void ForkParentAfter(ThreadState *thr, uptr pc) {
  thr->suppress_reports--;  // Enabled in ForkBefore.
  ScopedErrorReportLock::Unlock();
  ctx->slot_mtx.Unlock();
  ctx->thread_registry.Unlock();
}

void ForkChildAfter(ThreadState *thr, uptr pc) {
  thr->suppress_reports--;  // Enabled in ForkBefore.
  ScopedErrorReportLock::Unlock();
  ctx->slot_mtx.Unlock();
  ctx->thread_registry.Unlock();

  u32 nthread = ctx->thread_registry.RunningThreads();
  VPrintf(1,
          "ThreadSanitizer: forked new process with pid %d,"
          " parent had %d threads\n",
          (int)internal_getpid(), (int)nthread);
  if (nthread == 1) {
    StartBackgroundThread();
  } else {
    // We've just forked a multi-threaded process. We cannot reasonably function
    // after that (some mutexes may be locked before fork). So just enable
    // ignores for everything in the hope that we will exec soon.
    ctx->after_multithreaded_fork = true;
    thr->ignore_interceptors++;
    thr->suppress_reports++;
    ThreadIgnoreBegin(thr, pc);
    ThreadIgnoreSyncBegin(thr, pc);
  }
}
#endif

#if SANITIZER_GO
NOINLINE
void GrowShadowStack(ThreadState* thr) {
  const int sz = thr->shadow_stack_end - thr->shadow_stack;
  const int newsz = 2 * sz;
  uptr* newstack = (uptr*)Alloc(newsz * sizeof(uptr));
  internal_memcpy(newstack, thr->shadow_stack, sz * sizeof(uptr));
  Free(thr->shadow_stack);
  thr->shadow_stack = newstack;
  thr->shadow_stack_pos = newstack + sz;
  thr->shadow_stack_end = newstack + newsz;
}
#endif

StackID CurrentStackId(ThreadState* thr, uptr pc) {
#if !SANITIZER_GO
  if (!thr->is_inited) // May happen during bootstrap.
    return kInvalidStackID;
#endif
  if (pc != 0) {
#if !SANITIZER_GO
    DCHECK_LT(thr->shadow_stack_pos, thr->shadow_stack_end);
#else
    if (thr->shadow_stack_pos == thr->shadow_stack_end)
      GrowShadowStack(thr);
#endif
    thr->shadow_stack_pos[0] = pc;
    thr->shadow_stack_pos++;
  }
  StackID id = StackDepotPut(
      StackTrace(thr->shadow_stack, thr->shadow_stack_pos - thr->shadow_stack));
  if (pc != 0)
    thr->shadow_stack_pos--;
  return id;
}

NOINLINE
void TraceSwitch(ThreadState* thr) {
  Trace* trace = &thr->tctx->trace;
  Event* pos = (Event*)atomic_load_relaxed(&thr->trace_pos);
  DCHECK_EQ((uptr)(pos + 1) & 0xff0, 0);
  auto part = trace->parts.Back();
  if (part) {
    Event* end = &part->events[TracePart::kSize];
    DCHECK_GE(pos, &part->events[0]);
    DCHECK_LE(pos, end);
    if (pos + 1 < end) {
      if (((uptr)pos & 0xfff) == 0xff8)
        *pos++ = NopEvent;
      *pos++ = NopEvent;
      DCHECK_LE(pos + 2, end);
      atomic_store_relaxed(&thr->trace_pos, (uptr)pos);
      Event* ev;
      CHECK(TraceAcquire(thr, &ev));
      return;
    }
    for (; pos < end; pos++)
      *pos = NopEvent;
  }
#if !SANITIZER_GO
  if (ctx->after_multithreaded_fork) {
    // We just need to survive till exec.
    CHECK(part);
    atomic_store_relaxed(&thr->trace_pos, (uptr)&part->events[0]);
    return;
  }
#endif
  SlotLocker locker(thr, true);
  part = TracePartAlloc(thr);
  part->trace = trace;
  //!!! instead of storing start_stack/mset, trace corresponding events (func entry, mutex lock)?
  part->start_stack.Init(thr->shadow_stack,
                         thr->shadow_stack_pos - thr->shadow_stack);
  part->start_mset = thr->mset;
  part->prev_pc = thr->trace_prev_pc;
  TracePart* recycle = nullptr;
  // Keep roughly half of parts local to the thread (not queued into the recycle queue).
  uptr local_parts = (Max(3, flags()->history_size) + 1) / 2;
  {
    Lock lock(&trace->mtx);
    if (trace->parts.Empty())
      trace->local_head = part;
    if (trace->parts.Size() >= local_parts) {
      recycle = trace->local_head;
      trace->local_head = trace->parts.Next(recycle);
    }
    trace->parts.PushBack(part);
    atomic_store_relaxed(&thr->trace_pos, (uptr)&part->events[0]);
    TraceTime(thr);
  }
  {
    //!!! TracePartAlloc also locks ctx->slot_mtx.
    // Is it possible to combine this critical section and the one in TracePartAlloc?
    Lock lock(&ctx->slot_mtx);
    ctx->slot_queue.Remove(thr->slot);
    ctx->slot_queue.PushBack(thr->slot);
    if (recycle)
      ctx->trace_part_recycle.PushBack(recycle);
  }
}

ALWAYS_INLINE Shadow LoadShadow(RawShadow* p) {
  return Shadow(atomic_load((atomic_uint32_t*)p, memory_order_relaxed));
}

ALWAYS_INLINE void StoreShadow(RawShadow* sp, RawShadow s) {
  atomic_store((atomic_uint32_t*)sp, s, memory_order_relaxed);
}

NOINLINE void DoReportRace(ThreadState* thr, RawShadow* shadow_mem, Shadow cur,
                           Shadow old, AccessType typ) {
  // This prevents trapping of this address in future.
  for (uptr i = 0; i < kShadowCnt; i++)
    StoreShadow(&shadow_mem[i], i == 0 ? Shadow::kShadowRodata : 0);
  ReportRace(thr, shadow_mem, cur, Shadow(old), typ);
}

ALWAYS_INLINE
bool HappensBefore(Shadow old, ThreadState* thr) {
  return thr->clock.Get(old.sid()) >= old.epoch();
}

ALWAYS_INLINE
void MemoryAccessImpl1(ThreadState* thr, uptr addr, u32 kAccessSize,
                       bool kAccessIsWrite, bool kIsAtomic,
                       RawShadow* shadow_mem, Shadow cur) {
  // Scan all the shadow values and dispatch to 4 categories:
  // same, replace, candidate and race (see comments below).
  // we consider only 3 cases regarding access sizes:
  // equal, intersect and not intersect. initially I considered
  // larger and smaller as well, it allowed to replace some
  // 'candidates' with 'same' or 'replace', but I think
  // it's just not worth it (performance- and complexity-wise).

  bool stored = false;
  Shadow old(0);

  for (int idx = 0; idx < 4; idx++) {
    RawShadow* sp = &shadow_mem[idx];
    old = LoadShadow(sp);
    if (LIKELY(old.IsZero())) {
      if (!stored)
        StoreShadow(sp, cur.raw());
      return;
    }
    if (LIKELY(!Shadow::TwoRangesIntersect(cur, old)))
      continue;
    if (LIKELY(Shadow::SidsAreEqual(old, cur))) {
      if (LIKELY(Shadow::AddrSizeEqual(cur, old) &&
                 old.IsRWWeakerOrEqual(cur, kAccessIsWrite, kIsAtomic))) {
        StoreShadow(sp, cur.raw());
        stored = true;
      }
      continue;
    }
    if (LIKELY(old.IsBothReadsOrAtomic(kAccessIsWrite, kIsAtomic)))
      continue;
    if (LIKELY(!HappensBefore(old, thr)))
      goto RACE;
  }

  // We did not find any races and had already stored
  // the current access info, so we are done.
  if (LIKELY(stored))
    return;
  {
    // Choose a random candidate slot and replace it.
    uptr index = static_cast<uptr>(cur.epoch()) %
                 kShadowCnt; //!!! very low entropy, epoch does not change often
    StoreShadow(&shadow_mem[index], cur.raw());
  }
  return;
RACE:
  DoReportRace(thr, shadow_mem, cur, old, 0);
}

ALWAYS_INLINE
bool ContainsSameAccessSlow(RawShadow* s, RawShadow a, bool isRead) {
  Shadow cur(a);
  for (uptr i = 0; i < kShadowCnt; i++) {
    Shadow old(LoadShadow(&s[i]));
    if (isRead && old.raw() == Shadow::kShadowRodata)
      return true;
    //!!! speed up, this is used at least for Go.
    if (Shadow::AddrSizeEqual(cur, old) && old.sid() == cur.sid() &&
        old.epoch() == cur.epoch() && old.IsAtomic() == cur.IsAtomic() &&
        old.IsRead() <= cur.IsRead() /* && !old.IsFreed()*/)
      return true;
  }
  return false;
}

#if defined(__SSE3__)
ALWAYS_INLINE
bool ContainsSameAccessFast(RawShadow* s, RawShadow a, bool isRead) {
  // This is an optimized version of ContainsSameAccessSlow.
  const m128 access = _mm_set1_epi32(a);
  const m128 shadow = _mm_load_si128((m128*)s);
  if (isRead) {
    // For reads we need to reset read bit in the shadow,
    // because we need to match read with both reads and writes.
    // kShadowRodata has only read bit set, so it does what we want.
    // We also abuse it for rodata check to save few cycles
    // since we already loaded kShadowRodata into a register.
    // Reads from rodata can't race.
    // Measurements show that they can be 10-20% of all memory accesses.
    // kShadowRodata has epoch 0 which cannot appear in shadow normally
    // (thread epochs start from 1). So the same read bit mask
    // serves as rodata indicator.
    // Access to .rodata section, no races here.
    const m128 read_mask = _mm_set1_epi32(Shadow::kShadowRodata);
    //!!! we can also skip it for range memory access, they already checked
    //! rodata.
#  if !SANITIZER_GO
    const m128 ro = _mm_cmpeq_epi32(shadow, read_mask);
#  endif
    const m128 masked_shadow = _mm_or_si128(shadow, read_mask);
    const m128 same = _mm_cmpeq_epi32(masked_shadow, access);
#  if !SANITIZER_GO
    const m128 res = _mm_or_si128(ro, same);
    return _mm_movemask_epi8(res);
#  else
    return _mm_movemask_epi8(same);
#  endif
  }
  const m128 same = _mm_cmpeq_epi32(shadow, access);
  return _mm_movemask_epi8(same);
}
#endif

ALWAYS_INLINE
bool ContainsSameAccess(RawShadow* s, RawShadow a, bool isRead) {
#if defined(__SSE3__)
  bool res = ContainsSameAccessFast(s, a, isRead);
  // NOTE: this check can fail if the shadow is concurrently mutated
  // by other threads. But it still can be useful if you modify
  // ContainsSameAccessFast and want to ensure that it's not completely broken.
  // DCHECK_EQ(res, ContainsSameAccessSlow(s, a, isRead));
  return res;
#else
  return ContainsSameAccessSlow(s, a, isRead);
#endif
}

char* DumpShadow(char* buf, RawShadow raw) {
  if (raw == 0) {
    internal_snprintf(buf, 64, "0");
    return buf;
  }
  Shadow s(raw);
  internal_snprintf(buf, 64, "{tid=%u@%u access=0x%x type=%u/%u}",
                    static_cast<u32>(s.sid()), static_cast<u32>(s.epoch()),
                    s.access(), s.IsRead(), s.IsAtomic());
  return buf;
}

ALWAYS_INLINE WARN_UNUSED_RESULT
bool TraceMemoryAccess(ThreadState* thr, uptr pc, uptr addr, uptr size, AccessType typ) {
  DCHECK(size == 1 || size == 2 || size == 4 || size == 8);
  if (!kCollectHistory)
    return true;
  EventAccess* ev;
  if (!TraceAcquire(thr, &ev))
    return false;
  u64 size_log = size == 1 ? 0 : size == 2 ? 1 : size == 4 ? 2 : 3;
  uptr pcDelta = pc - thr->trace_prev_pc + (1 << (EventAccess::kPCBits - 1));
  thr->trace_prev_pc = pc;
  if (LIKELY(pcDelta < (1 << EventAccess::kPCBits))) {
    ev->is_access = 1;
    ev->isRead = !!(typ & AccessRead);
    ev->isAtomic = !!(typ & AccessAtomic);
    ev->sizeLog = size_log;
    ev->pcDelta = pcDelta;
    DCHECK_EQ(ev->pcDelta, pcDelta);
    ev->addr = addr;
    TraceRelease(thr, ev);
    return true;
  }
  auto evex = reinterpret_cast<EventAccessExt*>(ev);
  evex->is_access = 0;
  evex->is_func = 0;
  evex->type = EventTypeAccessExt;
  evex->isRead = !!(typ & AccessRead);
  evex->isAtomic = !!(typ & AccessAtomic);
  evex->sizeLog = size_log;
  evex->addr = addr;
  evex->pc = pc;
  TraceRelease(thr, evex);
  return true;
}

NOINLINE void TraceRestartMemoryAccess(ThreadState* thr, uptr pc, uptr addr, uptr size, AccessType typ) {
  TraceSwitch(thr);
  MemoryAccess(thr, pc, addr, size, typ);
}

NOINLINE void DoReportRaceV(ThreadState* thr, RawShadow* shadow_mem, Shadow cur,
                            u32 race_mask, m128 shadow, AccessType typ) {
  CHECK_NE(race_mask, 0);
  u32 old;
  switch (__builtin_ffs(race_mask) / 4) {
  case 0:
    old = _mm_extract_epi32(shadow, 0);
    break;
  case 1:
    old = _mm_extract_epi32(shadow, 1);
    break;
  case 2:
    old = _mm_extract_epi32(shadow, 2);
    break;
  case 3:
    old = _mm_extract_epi32(shadow, 3);
    break;
  }
  Shadow prev(old);
  //!!! this part is not supported in the non-vector code.
  if (prev.sid() == kFreeSid)
    prev = Shadow(_mm_extract_epi32(shadow, 1));
  DoReportRace(thr, shadow_mem, cur, prev, typ);
}

ALWAYS_INLINE
bool ContainsSameAccessV(m128 shadow, m128 access, AccessType typ) {
  //!!! we could check there is already a larger access of the same type,
  // e.g. we just allocated a block (so it has an 8 byte write) and doing
  // smaller writes to it, these don't need to be handled/stored separately.
  // However, the check will be more expensive then.
  if (!(typ & AccessRead)) {
    const m128 same = _mm_cmpeq_epi32(shadow, access);
    return _mm_movemask_epi8(same);
  }
  // For reads we need to reset read bit in the shadow,
  // because we need to match read with both reads and writes.
  // kShadowRodata has only read bit set, so it does what we want.
  // We also abuse it for rodata check to save few cycles
  // since we already loaded kShadowRodata into a register.
  // Reads from rodata can't race.
  // Measurements show that they can be 10-20% of all memory accesses.
  // kShadowRodata has epoch 0 which cannot appear in shadow normally
  // (thread epochs start from 1). So the same read bit mask
  // serves as rodata indicator.

  //!!! we can skip kShadowRodata check for range memory access,
  // they already checked rodata.
  const m128 read_mask = _mm_set1_epi32(Shadow::kShadowRodata);
  const m128 masked_shadow = _mm_or_si128(shadow, read_mask);
  m128 same = _mm_cmpeq_epi32(masked_shadow, access);
#if !SANITIZER_GO
  const m128 ro = _mm_cmpeq_epi32(shadow, read_mask);
  same = _mm_or_si128(ro, same);
#else
#endif
  return _mm_movemask_epi8(same);
}

ALWAYS_INLINE
bool CheckRaces(ThreadState* thr, RawShadow* shadow_mem, Shadow cur,
                m128 shadow, m128 access, AccessType typ) {
  const m128 zero = _mm_setzero_si128();
  //!!! These constants are compiled into loads of globals.
  // Would it be faster to obtain some consts from others with e.g. shifts (mask_sid = mask_access << 8)?
  //!!! Is it possible/make sense to compute them in some way?
  // E.g. mask_access == take zero, shift left by 8 shifting in 1s?
  const m128 mask_access = _mm_set1_epi32(0x000000ff);
  const m128 mask_sid = _mm_set1_epi32(0x0000ff00);
  const m128 mask_read_atomic = _mm_set1_epi32(0xc0000000);
  const m128 access_and = _mm_and_si128(access, shadow);
  const m128 access_xor = _mm_xor_si128(access, shadow);
  //!!! can we and intersect+not_same_sid and then negate once?
  const m128 intersect = _mm_and_si128(access_and, mask_access);
  const m128 not_intersect = _mm_cmpeq_epi32(intersect, zero);
  const m128 not_same_sid = _mm_and_si128(access_xor, mask_sid);
  const m128 same_sid = _mm_cmpeq_epi32(not_same_sid, zero);
  const m128 both_read_or_atomic = _mm_and_si128(access_and, mask_read_atomic);
  const m128 no_race =
      _mm_or_si128(_mm_or_si128(not_intersect, same_sid), both_read_or_atomic);
  const int race_mask = _mm_movemask_epi8(_mm_cmpeq_epi32(no_race, zero));
  DPrintf2("#%d:  MOP: not_intersect=%V same_sid=%V both_read_or_atomic=%V "
           "race_mask=%04x\n", (int)cur.sid(),
           not_intersect, same_sid, both_read_or_atomic, race_mask);
  if (UNLIKELY(race_mask))
    goto SHARED;

STORE : {
  if (typ & AccessTemp)
    return false;
  //!!! we could also replace different sid's if access is the same,
  // rw weaker and happens before. However, just checking access below
  // is not enough because we also need to check that !both_read_or_atomic
  // (reads from different sids can be concurrent).
  //!!! theoretically we could also replace smaller accesses with larger accesses,
  // but it's unclear if it's worth doing.
  const m128 mask_access_sid = _mm_set1_epi32(0x0000ffff);
  const m128 not_same_sid_access = _mm_and_si128(access_xor, mask_access_sid);
  const m128 same_sid_access = _mm_cmpeq_epi32(not_same_sid_access, zero);
  const m128 access_read_atomic =
      _mm_set1_epi32((typ & (AccessRead | AccessAtomic)) << 30);
  const m128 rw_weaker =
      _mm_cmpeq_epi32(_mm_max_epu32(shadow, access_read_atomic), shadow);
  const m128 rewrite = _mm_and_si128(same_sid_access, rw_weaker);
  const int rewrite_mask = _mm_movemask_epi8(rewrite);
  int index = __builtin_ffs(rewrite_mask);
  if (UNLIKELY(index == 0)) {
    const m128 empty = _mm_cmpeq_epi32(shadow, zero);
    const int empty_mask = _mm_movemask_epi8(empty);
    index = __builtin_ffs(empty_mask);
    if (UNLIKELY(index == 0))
      index = (atomic_load_relaxed(&thr->trace_pos) / 2) % 16;
  }
  DPrintf2("#%d:  MOP: replacing index=%d\n", (int)cur.sid(), index / 4);
  StoreShadow(&shadow_mem[index / 4], cur.raw());
  //!!! consider zeroing other slots determined by rewrite_mask
  return false;
}

SHARED:
  m128 thread_epochs = _mm_set1_epi32(0x7fffffff);
  // Need to unwind this because _mm_extract_epi8/_mm_insert_epi32
  // indexes must be constants.
#define LOAD_EPOCH(idx)                                                        \
  if (LIKELY(race_mask & (1 << (idx * 4)))) {                                  \
    u8 sid = _mm_extract_epi8(shadow, idx * 4 + 1);                            \
    u16 epoch = static_cast<u16>(thr->clock.Get(static_cast<Sid>(sid)));       \
    thread_epochs = _mm_insert_epi32(thread_epochs, u32(epoch) << 16, idx);    \
  }
  LOAD_EPOCH(0);
  LOAD_EPOCH(1);
  LOAD_EPOCH(2);
  LOAD_EPOCH(3);
#undef LOAD_EPOCH
  const m128 mask_epoch = _mm_set1_epi32(0x3fff0000);
  const m128 shadow_epochs = _mm_and_si128(shadow, mask_epoch);
  const m128 concurrent = _mm_cmplt_epi32(thread_epochs, shadow_epochs);
  const int concurrent_mask = _mm_movemask_epi8(concurrent);
  DPrintf2("#%d:  MOP: shadow_epochs=%V thread_epochs=%V concurrent_mask=%04x\n",
            (int)cur.sid(), shadow_epochs, thread_epochs, concurrent_mask);
  if (LIKELY(concurrent_mask == 0))
    goto STORE;

  DoReportRaceV(thr, shadow_mem, cur, concurrent_mask, shadow, typ);
  return true;
}

ALWAYS_INLINE USED void
MemoryAccess(ThreadState* thr, uptr pc, uptr addr, uptr size, AccessType typ) {
  RawShadow* shadow_mem = (RawShadow*)MemToShadow(addr);
  char memBuf[4][64];
  (void)memBuf;
  DPrintf2("#%d: Access: @%p %p size=%d"
           " typ=%x shadow=%p {%s, %s, %s, %s}\n",
           (int)thr->tid, (void*)pc, (void*)addr, size, typ,
           shadow_mem, DumpShadow(memBuf[0], shadow_mem[0]),
           DumpShadow(memBuf[1], shadow_mem[1]),
           DumpShadow(memBuf[2], shadow_mem[2]),
           DumpShadow(memBuf[3], shadow_mem[3]));
#if SANITIZER_DEBUG
  if (!IsAppMem(addr)) {
    Printf("Access to non app mem %zx\n", addr);
    DCHECK(IsAppMem(addr));
  }
  if (!IsShadowMem((uptr)shadow_mem)) {
    Printf("Bad shadow addr %p (%zx)\n", shadow_mem, addr);
    DCHECK(IsShadowMem((uptr)shadow_mem));
  }
#endif

  FastState fast_state = thr->fast_state;
  Shadow cur(fast_state, addr, size, typ);

  // This is an optimized version of ContainsSameAccessSlow.
  const m128 access = _mm_set1_epi32(cur.raw());
  const m128 shadow = _mm_load_si128((m128*)shadow_mem);
  DPrintf2("#%d:  MOP: shadow=%V access=%V\n", (int)cur.sid(), shadow, access);
  if (LIKELY(ContainsSameAccessV(shadow, access, typ)))
    return;
  if (UNLIKELY(fast_state.IgnoreAccesses()))
    return;
  if (!TraceMemoryAccess(thr, pc, addr, size, typ))
    return TraceRestartMemoryAccess(thr, pc, addr, size, typ);
  CheckRaces(thr, shadow_mem, cur, shadow, access, typ);
}

ALWAYS_INLINE WARN_UNUSED_RESULT bool
TryTraceMemoryAccessRange(ThreadState* thr, uptr pc, uptr addr, uptr size, AccessType typ) {
  if (!kCollectHistory)
    return true;
  EventAccessRange* ev;
  if (!TraceAcquire(thr, &ev))
    return false;
  thr->trace_prev_pc = pc;
  ev->is_access = 0;
  ev->is_func = 0;
  ev->type = EventTypeAccessRange;
  ev->isRead = !!(typ & AccessRead);
  ev->isFreed = !!(typ & AccessFree);
  ev->sizeLo = size;
  ev->pc = pc;
  ev->addr = addr;
  ev->sizeHi = size >> EventAccessRange::kSizeLoBits;
  TraceRelease(thr, ev);
  return true;
}

void TraceMemoryAccessRange(ThreadState* thr, uptr pc, uptr addr, uptr size, AccessType typ) {
  if (TryTraceMemoryAccessRange(thr, pc, addr, size, typ))
    return;
  TraceSwitch(thr);
  bool res = TryTraceMemoryAccessRange(thr, pc, addr, size, typ);
  DCHECK(res);
  (void)res;
}

NOINLINE
void RestartUnalignedMemoryAccess(ThreadState* thr, uptr pc, uptr addr,
                                  uptr size, AccessType typ) {
  TraceSwitch(thr);
  UnalignedMemoryAccess(thr, pc, addr, size, typ);
}

ALWAYS_INLINE USED void UnalignedMemoryAccess(ThreadState* thr, uptr pc,
                                              uptr addr, uptr size,
                                              AccessType typ) {
  DCHECK_LE(size, 8);
  FastState fast_state = thr->fast_state;
  if (UNLIKELY(fast_state.IgnoreAccesses()))
    return;
  RawShadow* shadow_mem = (RawShadow*)MemToShadow(addr);
  bool traced = false;
  uptr size1 = Min<uptr>(size, RoundUp(addr + 1, kShadowCell) - addr);
  {
    Shadow cur(fast_state, addr, size1, typ);

    const m128 access = _mm_set1_epi32(cur.raw());
    const m128 shadow = _mm_load_si128((m128*)shadow_mem);
    if (LIKELY(ContainsSameAccessV(shadow, access, typ)))
      goto SECOND;
    if (!TryTraceMemoryAccessRange(thr, pc, addr, size, typ))
      return RestartUnalignedMemoryAccess(thr, pc, addr, size, typ);
    traced = true;
    if (UNLIKELY(CheckRaces(thr, shadow_mem, cur, shadow, access, typ)))
      return;
  }
SECOND:
  uptr size2 = size - size1;
  if (LIKELY(size2 == 0))
    return;
  {
    shadow_mem += kShadowCnt;
    Shadow cur(fast_state, 0, size2, typ);
    const m128 access = _mm_set1_epi32(cur.raw());
    const m128 shadow = _mm_load_si128((m128*)shadow_mem);
    if (LIKELY(ContainsSameAccessV(shadow, access, typ)))
      return;
    if (!traced) {
      if (!TryTraceMemoryAccessRange(thr, pc, addr, size, typ))
        return RestartUnalignedMemoryAccess(thr, pc, addr, size, typ);
    }
    CheckRaces(thr, shadow_mem, cur, shadow, access, typ);
  }
}

static void MemoryRangeSet(ThreadState* thr, uptr pc, uptr addr, uptr size,
                           RawShadow val) {
  (void)thr;
  (void)pc;
  if (size == 0)
    return;
  DCHECK_EQ(addr % kShadowCell, 0);
  DCHECK_EQ(size % kShadowCell, 0);
  // If a user passes some insane arguments (memset(0)),
  // let it just crash as usual.
  if (!IsAppMem(addr) || !IsAppMem(addr + size - 1))
    return;
  // Don't want to touch lots of shadow memory.
  // If a program maps 10MB stack, there is no need reset the whole range.
  // UnmapOrDie/MmapFixedNoReserve does not work on Windows.
  if (SANITIZER_WINDOWS || size < common_flags()->clear_shadow_mmap_threshold) {
    RawShadow* p = (RawShadow*)MemToShadow(addr);
    CHECK(IsShadowMem((uptr)p));
    CHECK(IsShadowMem((uptr)(p + size * kShadowCnt / kShadowCell - 1)));
    //!!! optimize using vector instructions
    for (uptr i = 0; i < size / kShadowCell * kShadowCnt;) {
      p[i++] = val;
      for (uptr j = 1; j < kShadowCnt; j++)
        p[i++] = 0;
    }
  } else {
    // The region is big, reset only beginning and end.
    const uptr kPageSize = GetPageSizeCached();
    RawShadow* begin = (RawShadow*)MemToShadow(addr);
    RawShadow* end = begin + size / kShadowCell * kShadowCnt;
    RawShadow* p = begin;
    // Set at least first kPageSize/2 to page boundary.
    //!!! optimize using vector instructions
    while ((p < begin + kPageSize / kShadowSize / 2) || ((uptr)p % kPageSize)) {
      *p++ = val;
      for (uptr j = 1; j < kShadowCnt; j++)
        *p++ = 0;
    }
    // Reset middle part.
    RawShadow* p1 = p;
    p = RoundDown(end, kPageSize);
    if (!MmapFixedSuperNoReserve((uptr)p1, (uptr)p - (uptr)p1))
      Die();
    // Set the ending.
    //!!! optimize using vector instructions
    while (p < end) {
      *p++ = val;
      for (uptr j = 1; j < kShadowCnt; j++)
        *p++ = 0;
    }
  }
}

void MemoryResetRange(ThreadState* thr, uptr pc, uptr addr, uptr size) {
  uptr addr1 = RoundDown(addr, kShadowCell);
  uptr size1 = RoundUp(size + addr - addr1, kShadowCell);
  MemoryRangeSet(thr, pc, addr1, size1, 0);
}

void MemoryRangeFreed(ThreadState* thr, uptr pc, uptr addr, uptr size) {
  DCHECK_EQ(addr % kShadowCell, 0);
  size = RoundUp(size, kShadowCell);
  // Processing more than 1k (2k of shadow) is expensive,
  // can cause excessive memory consumption (user does not necessary touch
  // the whole range) and most likely unnecessary.
  size = Min<uptr>(size, 1024);
  //!!! This may need to lock the slot to ensure synchronization
  // with the reset. The problem with "freed" memory is that it's
  // not "monotonic" -- freed memory is bad to access, but then
  // if the heap block is reallocated, it's good to access again.
  // It's not the case with bad accesses due to races.
  // As the result a garbage "freed" shadow can lead to a false
  // positive if it happen to match a real free in the trace,
  // but the heap block was reallocated, so it's still good to access.
  const AccessType typ = AccessWrite | AccessFree | AccessTemp;
  TraceMemoryAccessRange(thr, pc, addr, size, typ);
  RawShadow* shadow_mem = (RawShadow*)MemToShadow(addr);
  Shadow cur(thr->fast_state, 0, kShadowCell, typ);
  const m128 access = _mm_set1_epi32(cur.raw());
  const m128 freed = _mm_setr_epi32(Shadow::FreedMarker(), Shadow::Freed(cur.sid(), cur.epoch()), 0, 0);
  for (; size >= kShadowCell; size -= kShadowCell, shadow_mem += kShadowCnt) {
    const m128 shadow = _mm_load_si128((m128*)shadow_mem);
    _mm_store_si128((m128*)shadow_mem, freed);
    if (LIKELY(ContainsSameAccessV(shadow, access, typ)))
      continue;
    if (UNLIKELY(CheckRaces(thr, shadow_mem, cur, shadow, access, typ)))
      return;
  }
}

void MemoryRangeImitateWrite(ThreadState* thr, uptr pc, uptr addr, uptr size) {
  DCHECK_EQ(addr % kShadowCell, 0);
  size = RoundUp(size, kShadowCell);
  TraceMemoryAccessRange(thr, pc, addr, size, AccessWrite);
  Shadow cur(thr->fast_state, 0, 8, AccessWrite);
  MemoryRangeSet(thr, pc, addr, size, cur.raw());
}

void MemoryRangeImitateWriteOrReset(ThreadState* thr, uptr pc, uptr addr,
                                    uptr size) {
  if (thr->ignore_accesses == 0)
    MemoryRangeImitateWrite(thr, pc, addr, size);
  else
    MemoryResetRange(thr, pc, addr, size);
}

ALWAYS_INLINE
bool MemoryAccessRangeOne(ThreadState* thr, RawShadow* shadow_mem, Shadow cur, m128 access,
                          uptr addr, uptr size, AccessType typ) {
  const m128 shadow = _mm_load_si128((m128*)shadow_mem);
  if (LIKELY(ContainsSameAccessV(shadow, access, typ)))
    return false;
  return CheckRaces(thr, shadow_mem, cur, shadow, access, typ);
}

template <bool is_write>
NOINLINE void RestartMemoryAccessRange(ThreadState* thr, uptr pc, uptr addr,
                                       uptr size) {
  TraceSwitch(thr);
  MemoryAccessRangeT<is_write>(thr, pc, addr, size);
}

template <bool is_write>
void MemoryAccessRangeT(ThreadState* thr, uptr pc, uptr addr,
                        uptr size) { //!!! change all is_write to isRead
  const AccessType typ = is_write ? AccessWrite : AccessRead;
  RawShadow* shadow_mem = (RawShadow*)MemToShadow(addr);
  DPrintf2("#%d: MemoryAccessRange: @%p %p size=%d is_write=%d\n", thr->tid,
           (void*)pc, (void*)addr, (int)size, is_write);

#if SANITIZER_DEBUG
  if (!IsAppMem(addr)) {
    Printf("Access to non app mem %zx\n", addr);
    DCHECK(IsAppMem(addr));
  }
  if (!IsAppMem(addr + size - 1)) {
    Printf("Access to non app mem %zx\n", addr + size - 1);
    DCHECK(IsAppMem(addr + size - 1));
  }
  if (!IsShadowMem((uptr)shadow_mem)) {
    Printf("Bad shadow addr %p (%zx)\n", shadow_mem, addr);
    DCHECK(IsShadowMem((uptr)shadow_mem));
  }
  if (!IsShadowMem((uptr)(shadow_mem + size * kShadowCnt / 8 - 1))) {
    Printf("Bad shadow addr %p (%zx)\n", shadow_mem + size * kShadowCnt / 8 - 1,
           addr + size - 1);
    DCHECK(IsShadowMem((uptr)(shadow_mem + size * kShadowCnt / 8 - 1)));
  }
#endif

  // Access to .rodata section, no races here.
  // Measurements show that it can be 10-20% of all memory accesses.
  if (*shadow_mem == Shadow::kShadowRodata)
    return;

  FastState fast_state = thr->fast_state;
  if (UNLIKELY(fast_state.IgnoreAccesses()))
    return;

  if (!TryTraceMemoryAccessRange(thr, pc, addr, size, typ))
    return RestartMemoryAccessRange<is_write>(thr, pc, addr, size);

  //!!! update comment
  // Don't report more than one race in the same range access.
  // First, it's just unnecessary and can produce lots of reports.
  // Second, thr->trace_prev_pc that we use below may become invalid.
  // The scenario where this happens is rather elaborate and requires
  // an instrumented __sanitizer_report_error_summary callback and
  // a __tsan_symbolize_external callback and a race during a range memory
  // access larger than 8 bytes. MemoryAccessRange adds the current PC to
  // the trace and starts processing memory accesses. A first memory access
  // triggers a race, we report it and call the instrumented
  // __sanitizer_report_error_summary, which adds more stuff to the trace
  // since it is intrumented. Then a second memory access in MemoryAccessRange
  // also triggers a race and we get here and use thr->trace_prev_pc
  // which is incorrect now.
  // test/tsan/double_race.cpp contains a test case for this.

  if (UNLIKELY(addr % kShadowCell)) {
    // Handle unaligned beginning, if any.
    uptr size1 = Min(size, RoundUp(addr, kShadowCell) - addr);
    size -= size1;
    Shadow cur(fast_state, addr, size1, typ);
    const m128 access = _mm_set1_epi32(cur.raw());
    if (UNLIKELY(
            MemoryAccessRangeOne(thr, shadow_mem, cur, access, addr, size1, typ)))
      return;
    shadow_mem += kShadowCnt;
  }
  // Handle middle part, if any.
  Shadow cur(fast_state, 0, kShadowCell, typ);
  const m128 access = _mm_set1_epi32(cur.raw());
  for (; size >= kShadowCell; size -= kShadowCell, shadow_mem += kShadowCnt) {
    if (UNLIKELY(MemoryAccessRangeOne(thr, shadow_mem, cur, access, 0, kShadowCell, typ)))
      return;
  }
  // Handle ending, if any.
  if (UNLIKELY(size)) {
    Shadow cur(fast_state, 0, size, typ);
    const m128 access = _mm_set1_epi32(cur.raw());
    if (UNLIKELY(MemoryAccessRangeOne(thr, shadow_mem, cur, access, 0, size, typ)))
      return;
  }
}

template void MemoryAccessRangeT<true>(ThreadState* thr, uptr pc, uptr addr,
                                       uptr size);
template void MemoryAccessRangeT<false>(ThreadState* thr, uptr pc, uptr addr,
                                        uptr size);

void TraceMutexLock(ThreadState* thr, EventType type, uptr pc, uptr addr,
                    StackID stk) {
  DCHECK(type == EventTypeLock || type == EventTypeRLock);
  if (!kCollectHistory)
    return;
  //!!! should these events set trace_prev_pc as well?
  EventLock ev;
  ev.is_access = 0;
  ev.is_func = 0;
  ev.type = type;
  ev.pc = pc;
  ev.stackIDLo = static_cast<u64>(stk);
  ev.stackIDHi = static_cast<u64>(stk) >> EventLock::kStackIDLoBits;
  ev._ = 0;
  ev.addr = addr;
  TraceEvent(thr, ev);
}

void TraceMutexUnlock(ThreadState* thr, uptr addr) {
  if (!kCollectHistory)
    return;
  EventUnlock ev;
  ev.is_access = 0;
  ev.is_func = 0;
  ev.type = EventTypeUnlock;
  ev._ = 0;
  ev.addr = addr;
  TraceEvent(thr, ev);
}

void TraceTime(ThreadState* thr) {
  if (!kCollectHistory)
    return;
  FastState fast_state = thr->fast_state;
  EventTime ev;
  ev.is_access = 0;
  ev.is_func = 0;
  ev.type = EventTypeTime;
  ev.sid = static_cast<u64>(fast_state.sid());
  ev.epoch = static_cast<u64>(fast_state.epoch());
  ev._ = 0;
  TraceEvent(thr, ev);
}

NOINLINE void TraceRestartFuncEntry(ThreadState* thr, uptr pc) {
  TraceSwitch(thr);
  FuncEntry(thr, pc);
}

NOINLINE void TraceRestartFuncExit(ThreadState* thr) {
  TraceSwitch(thr);
  FuncExit(thr);
}

void ThreadIgnoreBegin(ThreadState* thr, uptr pc) {
  DPrintf("#%d: ThreadIgnoreBegin\n", thr->tid);
  thr->ignore_accesses++;
  CHECK_GT(thr->ignore_accesses, 0);
  thr->fast_state.SetIgnoreAccesses(true);
#if !SANITIZER_GO
  if (pc && !ctx->after_multithreaded_fork)
    thr->mop_ignore_set.Add(CurrentStackId(thr, pc));
#endif
}

void ThreadIgnoreEnd(ThreadState* thr) {
  DPrintf("#%d: ThreadIgnoreEnd\n", thr->tid);
  CHECK_GT(thr->ignore_accesses, 0);
  thr->ignore_accesses--;
  thr->fast_state.SetIgnoreAccesses(thr->ignore_accesses);
#if !SANITIZER_GO
  if (!thr->ignore_accesses)
    thr->mop_ignore_set.Reset();
#endif
}

#if !SANITIZER_GO
extern "C" SANITIZER_INTERFACE_ATTRIBUTE uptr
__tsan_testonly_shadow_stack_current_size() {
  ThreadState* thr = cur_thread();
  return thr->shadow_stack_pos - thr->shadow_stack;
}
#endif

void ThreadIgnoreSyncBegin(ThreadState* thr, uptr pc) {
  DPrintf("#%d: ThreadIgnoreSyncBegin\n", thr->tid);
  thr->ignore_sync++;
  CHECK_GT(thr->ignore_sync, 0);
#if !SANITIZER_GO
  if (pc && !ctx->after_multithreaded_fork)
    thr->sync_ignore_set.Add(CurrentStackId(thr, pc));
#endif
}

void ThreadIgnoreSyncEnd(ThreadState* thr) {
  DPrintf("#%d: ThreadIgnoreSyncEnd\n", thr->tid);
  CHECK_GT(thr->ignore_sync, 0);
  thr->ignore_sync--;
#if !SANITIZER_GO
  if (thr->ignore_sync == 0)
    thr->sync_ignore_set.Reset();
#endif
}

#if SANITIZER_DEBUG
void build_consistency_debug() {
}
#else
void build_consistency_release() {
}
#endif
} // namespace __tsan

namespace __sanitizer {

void PrintfBefore() {
#if !SANITIZER_GO
  // using namespace __tsan;
  // atomic_fetch_add(&cur_thread()->in_runtime, 2, memory_order_relaxed);
#endif
}

void PrintfAfter() {
#if !SANITIZER_GO
  // using namespace __tsan;
  // atomic_fetch_add(&cur_thread()->in_runtime, -2, memory_order_relaxed);
#endif
}
} // namespace __sanitizer

#if !SANITIZER_GO
// Must be included in this file to make sure everything is inlined.
#  include "tsan_interface_inl.h"
#endif

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
#include "sanitizer_common/sanitizer_interface_internal.h"
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

SANITIZER_WEAK_DEFAULT_IMPL
void __tsan_test_only_on_fork() {}

namespace __tsan {

#if !SANITIZER_GO
void (*on_initialize)(void);
int (*on_finalize)(int);
#endif

#if !SANITIZER_GO && !SANITIZER_APPLE
__attribute__((tls_model("initial-exec")))
THREADLOCAL char cur_thread_placeholder[sizeof(ThreadState)] ALIGNED(
    SANITIZER_CACHE_LINE_SIZE);
#endif
static char ctx_placeholder[sizeof(Context)] ALIGNED(SANITIZER_CACHE_LINE_SIZE);
Context *ctx;

// Can be overriden by a front-end.
#ifdef TSAN_EXTERNAL_HOOKS
bool OnFinalize(bool failed);
void OnInitialize();
#else
#include <dlfcn.h>
SANITIZER_WEAK_CXX_DEFAULT_IMPL
bool OnFinalize(bool failed) {
#if !SANITIZER_GO
  if (on_finalize)
    return on_finalize(failed);
#endif
  return failed;
}
SANITIZER_WEAK_CXX_DEFAULT_IMPL
void OnInitialize() {
#if !SANITIZER_GO
  if (on_initialize)
    on_initialize();
#endif
}
#endif

static ThreadContextBase *CreateThreadContext(Tid tid) {
  // Map thread trace when context is created.
  char name[50];
  internal_snprintf(name, sizeof(name), "trace %u", tid);
  MapThreadTrace(GetThreadTrace(tid), TraceSize() * sizeof(Event), name);
  const uptr hdr = GetThreadTraceHeader(tid);
  internal_snprintf(name, sizeof(name), "trace header %u", tid);
  MapThreadTrace(hdr, sizeof(Trace), name);
  new((void*)hdr) Trace();
  // We are going to use only a small part of the trace with the default
  // value of history_size. However, the constructor writes to the whole trace.
  // Release the unused part.
  uptr hdr_end = hdr + sizeof(Trace);
  hdr_end -= sizeof(TraceHeader) * (kTraceParts - TraceParts());
  hdr_end = RoundUp(hdr_end, GetPageSizeCached());
  if (hdr_end < hdr + sizeof(Trace)) {
    ReleaseMemoryPagesToOS(hdr_end, hdr + sizeof(Trace));
    uptr unused = hdr + sizeof(Trace) - hdr_end;
    if (hdr_end != (uptr)MmapFixedNoAccess(hdr_end, unused)) {
      Report("ThreadSanitizer: failed to mprotect [0x%zx-0x%zx) \n", hdr_end,
             unused);
      CHECK("unable to mprotect" && 0);
    }
  }
  return New<ThreadContext>(tid);
}

#if !SANITIZER_GO
static const u32 kThreadQuarantineSize = 16;
#else
static const u32 kThreadQuarantineSize = 64;
#endif

Context::Context()
    : initialized(),
      report_mtx(MutexTypeReport),
      nreported(),
      thread_registry(CreateThreadContext, kMaxTid, kThreadQuarantineSize,
                      kMaxTidReuse),
      racy_mtx(MutexTypeRacy),
      racy_stacks(),
      racy_addresses(),
      fired_suppressions_mtx(MutexTypeFired),
      clock_alloc(LINKER_INITIALIZED, "clock allocator") {
  fired_suppressions.reserve(8);
}

// The objects are allocated in TLS, so one may rely on zero-initialization.
ThreadState::ThreadState(Context *ctx, Tid tid, int unique_id, u64 epoch,
                         unsigned reuse_count, uptr stk_addr, uptr stk_size,
                         uptr tls_addr, uptr tls_size)
    : fast_state(tid, epoch)
      // Do not touch these, rely on zero initialization,
      // they may be accessed before the ctor.
      // , ignore_reads_and_writes()
      // , ignore_interceptors()
      ,
      clock(tid, reuse_count)
#if !SANITIZER_GO
      ,
      jmp_bufs()
#endif
      ,
      tid(tid),
      unique_id(unique_id),
      stk_addr(stk_addr),
      stk_size(stk_size),
      tls_addr(tls_addr),
      tls_size(tls_size)
#if !SANITIZER_GO
      ,
      last_sleep_clock(tid)
#endif
{
  CHECK_EQ(reinterpret_cast<uptr>(this) % SANITIZER_CACHE_LINE_SIZE, 0);
#if !SANITIZER_GO
  // C/C++ uses fixed size shadow stack.
  const int kInitStackSize = kShadowStackSize;
  shadow_stack = static_cast<uptr *>(
      MmapNoReserveOrDie(kInitStackSize * sizeof(uptr), "shadow stack"));
  SetShadowRegionHugePageMode(reinterpret_cast<uptr>(shadow_stack),
                              kInitStackSize * sizeof(uptr));
#else
  // Go uses malloc-allocated shadow stack with dynamic size.
  const int kInitStackSize = 8;
  shadow_stack = static_cast<uptr *>(Alloc(kInitStackSize * sizeof(uptr)));
#endif
  shadow_stack_pos = shadow_stack;
  shadow_stack_end = shadow_stack + kInitStackSize;
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
  const char *fname = flags()->profile_memory;
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

static void *BackgroundThread(void *arg) {
  // This is a non-initialized non-user thread, nothing to see here.
  // We don't use ScopedIgnoreInterceptors, because we want ignores to be
  // enabled even when the thread function exits (e.g. during pthread thread
  // shutdown code).
  cur_thread_init()->ignore_interceptors++;
  const u64 kMs2Ns = 1000 * 1000;
  const u64 start = NanoTime();

  u64 last_flush = NanoTime();
  uptr last_rss = 0;
  for (int i = 0;
      atomic_load(&ctx->stop_background_thread, memory_order_relaxed) == 0;
      i++) {
    SleepForMillis(100);
    u64 now = NanoTime();

    // Flush memory if requested.
    if (flags()->flush_memory_ms > 0) {
      if (last_flush + flags()->flush_memory_ms * kMs2Ns < now) {
        VPrintf(1, "ThreadSanitizer: periodic memory flush\n");
        FlushShadowMemory();
        last_flush = NanoTime();
      }
    }
    if (flags()->memory_limit_mb > 0) {
      uptr rss = GetRSS();
      uptr limit = uptr(flags()->memory_limit_mb) << 20;
      VPrintf(1, "ThreadSanitizer: memory flush check"
                 " RSS=%llu LAST=%llu LIMIT=%llu\n",
              (u64)rss >> 20, (u64)last_rss >> 20, (u64)limit >> 20);
      if (2 * rss > limit + last_rss) {
        VPrintf(1, "ThreadSanitizer: flushing memory due to RSS\n");
        FlushShadowMemory();
        rss = GetRSS();
        VPrintf(1, "ThreadSanitizer: memory flushed RSS=%llu\n", (u64)rss>>20);
      }
      last_rss = rss;
    }

    MemoryProfiler(now - start);

    // Flush symbolizer cache if requested.
    if (flags()->flush_symbolizer_ms > 0) {
      u64 last = atomic_load(&ctx->last_symbolize_time_ns,
                             memory_order_relaxed);
      if (last != 0 && last + flags()->flush_symbolizer_ms * kMs2Ns < now) {
        Lock l(&ctx->report_mtx);
        ScopedErrorReportLock l2;
        SymbolizeFlush();
        atomic_store(&ctx->last_symbolize_time_ns, 0, memory_order_relaxed);
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
  ReleaseMemoryPagesToOS(reinterpret_cast<uptr>(MemToShadow(addr)),
                         reinterpret_cast<uptr>(MemToShadow(addr + size)));
}

#if !SANITIZER_GO
// We call UnmapShadow before the actual munmap, at that point we don't yet
// know if the provided address/size are sane. We can't call UnmapShadow
// after the actual munmap becuase at that point the memory range can
// already be reused for something else, so we can't rely on the munmap
// return value to understand is the values are sane.
// While calling munmap with insane values (non-canonical address, negative
// size, etc) is an error, the kernel won't crash. We must also try to not
// crash as the failure mode is very confusing (paging fault inside of the
// runtime on some derived shadow address).
static bool IsValidMmapRange(uptr addr, uptr size) {
  if (size == 0)
    return true;
  if (static_cast<sptr>(size) < 0)
    return false;
  if (!IsAppMem(addr) || !IsAppMem(addr + size - 1))
    return false;
  // Check that if the start of the region belongs to one of app ranges,
  // end of the region belongs to the same region.
  const uptr ranges[][2] = {
      {LoAppMemBeg(), LoAppMemEnd()},
      {MidAppMemBeg(), MidAppMemEnd()},
      {HiAppMemBeg(), HiAppMemEnd()},
  };
  for (auto range : ranges) {
    if (addr >= range[0] && addr < range[1])
      return addr + size <= range[1];
  }
  return false;
}

void UnmapShadow(ThreadState *thr, uptr addr, uptr size) {
  if (size == 0 || !IsValidMmapRange(addr, size))
    return;
  DontNeedShadowFor(addr, size);
  ScopedGlobalProcessor sgp;
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
    // Mapping continuous heap.
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
  VPrintf(2, "mapped meta shadow for (0x%zx-0x%zx) at (0x%zx-0x%zx)\n", addr,
          addr + size, meta_begin, meta_end);
}

void MapThreadTrace(uptr addr, uptr size, const char *name) {
  DPrintf("#0: Mapping trace at 0x%zx-0x%zx(0x%zx)\n", addr, addr + size, size);
  CHECK_GE(addr, TraceMemBeg());
  CHECK_LE(addr + size, TraceMemEnd());
  CHECK_EQ(addr, addr & ~((64 << 10) - 1));  // windows wants 64K alignment
  if (!MmapFixedSuperNoReserve(addr, size, name)) {
    Printf("FATAL: ThreadSanitizer can not mmap thread trace (0x%zx/0x%zx)\n",
           addr, size);
    Die();
  }
}

#if !SANITIZER_GO
static void OnStackUnwind(const SignalContext &sig, const void *,
                          BufferedStackTrace *stack) {
  stack->Unwind(StackTrace::GetNextInstructionPc(sig.pc), sig.bp, sig.context,
                common_flags()->fast_unwind_on_fatal);
}

static void TsanOnDeadlySignal(int signo, void *siginfo, void *context) {
  HandleDeadlySignal(siginfo, context, GetTid(), &OnStackUnwind, nullptr);
}
#endif

void CheckUnwind() {
  // There is high probability that interceptors will check-fail as well,
  // on the other hand there is no sense in processing interceptors
  // since we are going to die soon.
  ScopedIgnoreInterceptors ignore;
#if !SANITIZER_GO
  cur_thread()->ignore_sync++;
  cur_thread()->ignore_reads_and_writes++;
#endif
  PrintCurrentStackSlow(StackTrace::GetCurrentPc());
}

bool is_initialized;

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

  ctx = new(ctx_placeholder) Context;
  const char *env_name = SANITIZER_GO ? "GORACE" : "TSAN_OPTIONS";
  const char *options = GetEnv(env_name);
  CacheBinaryName();
  CheckASLR();
  InitializeFlags(&ctx->flags, options, env_name);
  AvoidCVE_2016_2143();
  __sanitizer::InitializePlatformEarly();
  __tsan::InitializePlatformEarly();

#if !SANITIZER_GO
  InitializeAllocator();
  ReplaceSystemMalloc();
#endif
  if (common_flags()->detect_deadlocks)
    ctx->dd = DDetector::Create(flags());
  Processor *proc = ProcCreate();
  ProcWire(proc, thr);
  InitializeInterceptors();
  InitializePlatform();
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
  Symbolizer::GetOrInit()->AddHooks(EnterSymbolizer, ExitSymbolizer);
#endif

  VPrintf(1, "***** Running under ThreadSanitizer v2 (pid %d) *****\n",
          (int)internal_getpid());

  // Initialize thread 0.
  Tid tid = ThreadCreate(thr, 0, 0, true);
  CHECK_EQ(tid, kMainTid);
  ThreadStart(thr, tid, GetTid(), ThreadType::Regular);
#if TSAN_CONTAINS_UBSAN
  __ubsan::InitAsPlugin();
#endif
  ctx->initialized = true;

#if !SANITIZER_GO
  Symbolizer::LateInitialize();
  InitializeMemoryProfiler();
#endif

  if (flags()->stop_on_start) {
    Printf("ThreadSanitizer is suspended at startup (pid %d)."
           " Call __tsan_resume().\n",
           (int)internal_getpid());
    while (__tsan_resumed == 0) {}
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


int Finalize(ThreadState *thr) {
  bool failed = false;

  if (common_flags()->print_module_map == 1)
    DumpProcessMap();

  if (flags()->atexit_sleep_ms > 0 && ThreadCount(thr) > 1)
    SleepForMillis(flags()->atexit_sleep_ms);

  // Wait for pending reports.
  ctx->report_mtx.Lock();
  { ScopedErrorReportLock l; }
  ctx->report_mtx.Unlock();

#if !SANITIZER_GO
  if (Verbosity()) AllocatorPrintStats();
#endif

  ThreadFinalize(thr);

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
void ForkBefore(ThreadState *thr, uptr pc) SANITIZER_NO_THREAD_SAFETY_ANALYSIS {
  ctx->thread_registry.Lock();
  ctx->report_mtx.Lock();
  ScopedErrorReportLock::Lock();
  AllocatorLock();
  // Suppress all reports in the pthread_atfork callbacks.
  // Reports will deadlock on the report_mtx.
  // We could ignore sync operations as well,
  // but so far it's unclear if it will do more good or harm.
  // Unnecessarily ignoring things can lead to false positives later.
  thr->suppress_reports++;
  // On OS X, REAL(fork) can call intercepted functions (OSSpinLockLock), and
  // we'll assert in CheckNoLocks() unless we ignore interceptors.
  // On OS X libSystem_atfork_prepare/parent/child callbacks are called
  // after/before our callbacks and they call free.
  thr->ignore_interceptors++;
  // Disables memory write in OnUserAlloc/Free.
  thr->ignore_reads_and_writes++;

  __tsan_test_only_on_fork();
}

void ForkParentAfter(ThreadState *thr,
                     uptr pc) SANITIZER_NO_THREAD_SAFETY_ANALYSIS {
  thr->suppress_reports--;  // Enabled in ForkBefore.
  thr->ignore_interceptors--;
  thr->ignore_reads_and_writes--;
  AllocatorUnlock();
  ScopedErrorReportLock::Unlock();
  ctx->report_mtx.Unlock();
  ctx->thread_registry.Unlock();
}

void ForkChildAfter(ThreadState *thr, uptr pc,
                    bool start_thread) SANITIZER_NO_THREAD_SAFETY_ANALYSIS {
  thr->suppress_reports--;  // Enabled in ForkBefore.
  thr->ignore_interceptors--;
  thr->ignore_reads_and_writes--;
  AllocatorUnlock();
  ScopedErrorReportLock::Unlock();
  ctx->report_mtx.Unlock();
  ctx->thread_registry.Unlock();

  uptr nthread = 0;
  ctx->thread_registry.GetNumberOfThreads(0, 0, &nthread /* alive threads */);
  VPrintf(1, "ThreadSanitizer: forked new process with pid %d,"
      " parent had %d threads\n", (int)internal_getpid(), (int)nthread);
  if (nthread == 1) {
    if (start_thread)
      StartBackgroundThread();
  } else {
    // We've just forked a multi-threaded process. We cannot reasonably function
    // after that (some mutexes may be locked before fork). So just enable
    // ignores for everything in the hope that we will exec soon.
    ctx->after_multithreaded_fork = true;
    thr->ignore_interceptors++;
    ThreadIgnoreBegin(thr, pc);
    ThreadIgnoreSyncBegin(thr, pc);
  }
}
#endif

#if SANITIZER_GO
NOINLINE
void GrowShadowStack(ThreadState *thr) {
  const int sz = thr->shadow_stack_end - thr->shadow_stack;
  const int newsz = 2 * sz;
  auto *newstack = (uptr *)Alloc(newsz * sizeof(uptr));
  internal_memcpy(newstack, thr->shadow_stack, sz * sizeof(uptr));
  Free(thr->shadow_stack);
  thr->shadow_stack = newstack;
  thr->shadow_stack_pos = newstack + sz;
  thr->shadow_stack_end = newstack + newsz;
}
#endif

StackID CurrentStackId(ThreadState *thr, uptr pc) {
  if (!thr->is_inited)  // May happen during bootstrap.
    return kInvalidStackID;
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

namespace v3 {

NOINLINE
void TraceSwitchPart(ThreadState *thr) {
  Trace *trace = &thr->tctx->trace;
  Event *pos = reinterpret_cast<Event *>(atomic_load_relaxed(&thr->trace_pos));
  DCHECK_EQ(reinterpret_cast<uptr>(pos + 1) & TracePart::kAlignment, 0);
  auto *part = trace->parts.Back();
  DPrintf("TraceSwitchPart part=%p pos=%p\n", part, pos);
  if (part) {
    // We can get here when we still have space in the current trace part.
    // The fast-path check in TraceAcquire has false positives in the middle of
    // the part. Check if we are indeed at the end of the current part or not,
    // and fill any gaps with NopEvent's.
    Event *end = &part->events[TracePart::kSize];
    DCHECK_GE(pos, &part->events[0]);
    DCHECK_LE(pos, end);
    if (pos + 1 < end) {
      if ((reinterpret_cast<uptr>(pos) & TracePart::kAlignment) ==
          TracePart::kAlignment)
        *pos++ = NopEvent;
      *pos++ = NopEvent;
      DCHECK_LE(pos + 2, end);
      atomic_store_relaxed(&thr->trace_pos, reinterpret_cast<uptr>(pos));
      // Ensure we setup trace so that the next TraceAcquire
      // won't detect trace part end.
      Event *ev;
      CHECK(TraceAcquire(thr, &ev));
      return;
    }
    // We are indeed at the end.
    for (; pos < end; pos++) *pos = NopEvent;
  }
#if !SANITIZER_GO
  if (ctx->after_multithreaded_fork) {
    // We just need to survive till exec.
    CHECK(part);
    atomic_store_relaxed(&thr->trace_pos,
                         reinterpret_cast<uptr>(&part->events[0]));
    return;
  }
#endif
  part = new (MmapOrDie(sizeof(TracePart), "TracePart")) TracePart();
  part->trace = trace;
  thr->trace_prev_pc = 0;
  {
    Lock lock(&trace->mtx);
    trace->parts.PushBack(part);
    atomic_store_relaxed(&thr->trace_pos,
                         reinterpret_cast<uptr>(&part->events[0]));
  }
  // Make this part self-sufficient by restoring the current stack
  // and mutex set in the beginning of the trace.
  TraceTime(thr);
  for (uptr *pos = &thr->shadow_stack[0]; pos < thr->shadow_stack_pos; pos++)
    CHECK(TryTraceFunc(thr, *pos));
  for (uptr i = 0; i < thr->mset.Size(); i++) {
    MutexSet::Desc d = thr->mset.Get(i);
    TraceMutexLock(thr, d.write ? EventType::kLock : EventType::kRLock, 0,
                   d.addr, d.stack_id);
  }
}

}  // namespace v3

void TraceSwitch(ThreadState *thr) {
#if !SANITIZER_GO
  if (ctx->after_multithreaded_fork)
    return;
#endif
  thr->nomalloc++;
  Trace *thr_trace = ThreadTrace(thr->tid);
  Lock l(&thr_trace->mtx);
  unsigned trace = (thr->fast_state.epoch() / kTracePartSize) % TraceParts();
  TraceHeader *hdr = &thr_trace->headers[trace];
  hdr->epoch0 = thr->fast_state.epoch();
  ObtainCurrentStack(thr, 0, &hdr->stack0);
  hdr->mset0 = thr->mset;
  thr->nomalloc--;
}

Trace *ThreadTrace(Tid tid) { return (Trace *)GetThreadTraceHeader(tid); }

uptr TraceTopPC(ThreadState *thr) {
  Event *events = (Event*)GetThreadTrace(thr->tid);
  uptr pc = events[thr->fast_state.GetTracePos()];
  return pc;
}

uptr TraceSize() {
  return (uptr)(1ull << (kTracePartSizeBits + flags()->history_size + 1));
}

uptr TraceParts() {
  return TraceSize() / kTracePartSize;
}

#if !SANITIZER_GO
extern "C" void __tsan_trace_switch() {
  TraceSwitch(cur_thread());
}

extern "C" void __tsan_report_race() {
  ReportRace(cur_thread());
}
#endif

void ThreadIgnoreBegin(ThreadState *thr, uptr pc) {
  DPrintf("#%d: ThreadIgnoreBegin\n", thr->tid);
  thr->ignore_reads_and_writes++;
  CHECK_GT(thr->ignore_reads_and_writes, 0);
  thr->fast_state.SetIgnoreBit();
#if !SANITIZER_GO
  if (pc && !ctx->after_multithreaded_fork)
    thr->mop_ignore_set.Add(CurrentStackId(thr, pc));
#endif
}

void ThreadIgnoreEnd(ThreadState *thr) {
  DPrintf("#%d: ThreadIgnoreEnd\n", thr->tid);
  CHECK_GT(thr->ignore_reads_and_writes, 0);
  thr->ignore_reads_and_writes--;
  if (thr->ignore_reads_and_writes == 0) {
    thr->fast_state.ClearIgnoreBit();
#if !SANITIZER_GO
    thr->mop_ignore_set.Reset();
#endif
  }
}

#if !SANITIZER_GO
extern "C" SANITIZER_INTERFACE_ATTRIBUTE
uptr __tsan_testonly_shadow_stack_current_size() {
  ThreadState *thr = cur_thread();
  return thr->shadow_stack_pos - thr->shadow_stack;
}
#endif

void ThreadIgnoreSyncBegin(ThreadState *thr, uptr pc) {
  DPrintf("#%d: ThreadIgnoreSyncBegin\n", thr->tid);
  thr->ignore_sync++;
  CHECK_GT(thr->ignore_sync, 0);
#if !SANITIZER_GO
  if (pc && !ctx->after_multithreaded_fork)
    thr->sync_ignore_set.Add(CurrentStackId(thr, pc));
#endif
}

void ThreadIgnoreSyncEnd(ThreadState *thr) {
  DPrintf("#%d: ThreadIgnoreSyncEnd\n", thr->tid);
  CHECK_GT(thr->ignore_sync, 0);
  thr->ignore_sync--;
#if !SANITIZER_GO
  if (thr->ignore_sync == 0)
    thr->sync_ignore_set.Reset();
#endif
}

bool MD5Hash::operator==(const MD5Hash &other) const {
  return hash[0] == other.hash[0] && hash[1] == other.hash[1];
}

#if SANITIZER_DEBUG
void build_consistency_debug() {}
#else
void build_consistency_release() {}
#endif

}  // namespace __tsan

#if SANITIZER_CHECK_DEADLOCKS
namespace __sanitizer {
using namespace __tsan;
MutexMeta mutex_meta[] = {
    {MutexInvalid, "Invalid", {}},
    {MutexThreadRegistry, "ThreadRegistry", {}},
    {MutexTypeTrace, "Trace", {}},
    {MutexTypeReport,
     "Report",
     {MutexTypeSyncVar, MutexTypeGlobalProc, MutexTypeTrace}},
    {MutexTypeSyncVar, "SyncVar", {MutexTypeTrace}},
    {MutexTypeAnnotations, "Annotations", {}},
    {MutexTypeAtExit, "AtExit", {MutexTypeSyncVar}},
    {MutexTypeFired, "Fired", {MutexLeaf}},
    {MutexTypeRacy, "Racy", {MutexLeaf}},
    {MutexTypeGlobalProc, "GlobalProc", {}},
    {MutexTypeInternalAlloc, "InternalAlloc", {MutexLeaf}},
    {},
};

void PrintMutexPC(uptr pc) { StackTrace(&pc, 1).Print(); }
}  // namespace __sanitizer
#endif

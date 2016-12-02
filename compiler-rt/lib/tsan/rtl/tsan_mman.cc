//===-- tsan_mman.cc ------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
//===----------------------------------------------------------------------===//
#include "sanitizer_common/sanitizer_allocator_interface.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_placement_new.h"
#include "tsan_mman.h"
#include "tsan_rtl.h"
#include "tsan_report.h"
#include "tsan_flags.h"

// May be overriden by front-end.
SANITIZER_WEAK_DEFAULT_IMPL
void __sanitizer_malloc_hook(void *ptr, uptr size) {
  (void)ptr;
  (void)size;
}

SANITIZER_WEAK_DEFAULT_IMPL
void __sanitizer_free_hook(void *ptr) {
  (void)ptr;
}

namespace __tsan {

struct MapUnmapCallback {
  void OnMap(uptr p, uptr size) const { }
  void OnUnmap(uptr p, uptr size) const {
    // We are about to unmap a chunk of user memory.
    // Mark the corresponding shadow memory as not needed.
    DontNeedShadowFor(p, size);
    // Mark the corresponding meta shadow memory as not needed.
    // Note the block does not contain any meta info at this point
    // (this happens after free).
    const uptr kMetaRatio = kMetaShadowCell / kMetaShadowSize;
    const uptr kPageSize = GetPageSizeCached() * kMetaRatio;
    // Block came from LargeMmapAllocator, so must be large.
    // We rely on this in the calculations below.
    CHECK_GE(size, 2 * kPageSize);
    uptr diff = RoundUp(p, kPageSize) - p;
    if (diff != 0) {
      p += diff;
      size -= diff;
    }
    diff = p + size - RoundDown(p + size, kPageSize);
    if (diff != 0)
      size -= diff;
    uptr p_meta = (uptr)MemToMeta(p);
    ReleaseMemoryPagesToOS(p_meta, p_meta + size / kMetaRatio);
  }
};

static char allocator_placeholder[sizeof(Allocator)] ALIGNED(64);
Allocator *allocator() {
  return reinterpret_cast<Allocator*>(&allocator_placeholder);
}

struct GlobalProc {
  Mutex mtx;
  Processor *proc;

  GlobalProc()
      : mtx(MutexTypeGlobalProc, StatMtxGlobalProc)
      , proc(ProcCreate()) {
  }
};

static char global_proc_placeholder[sizeof(GlobalProc)] ALIGNED(64);
GlobalProc *global_proc() {
  return reinterpret_cast<GlobalProc*>(&global_proc_placeholder);
}

ScopedGlobalProcessor::ScopedGlobalProcessor() {
  GlobalProc *gp = global_proc();
  ThreadState *thr = cur_thread();
  if (thr->proc())
    return;
  // If we don't have a proc, use the global one.
  // There are currently only two known case where this path is triggered:
  //   __interceptor_free
  //   __nptl_deallocate_tsd
  //   start_thread
  //   clone
  // and:
  //   ResetRange
  //   __interceptor_munmap
  //   __deallocate_stack
  //   start_thread
  //   clone
  // Ideally, we destroy thread state (and unwire proc) when a thread actually
  // exits (i.e. when we join/wait it). Then we would not need the global proc
  gp->mtx.Lock();
  ProcWire(gp->proc, thr);
}

ScopedGlobalProcessor::~ScopedGlobalProcessor() {
  GlobalProc *gp = global_proc();
  ThreadState *thr = cur_thread();
  if (thr->proc() != gp->proc)
    return;
  ProcUnwire(gp->proc, thr);
  gp->mtx.Unlock();
}

void InitializeAllocator() {
  allocator()->Init(
      common_flags()->allocator_may_return_null,
      common_flags()->allocator_release_to_os_interval_ms);
}

void InitializeAllocatorLate() {
  new(global_proc()) GlobalProc();
}

void AllocatorProcStart(Processor *proc) {
  allocator()->InitCache(&proc->alloc_cache);
  internal_allocator()->InitCache(&proc->internal_alloc_cache);
}

void AllocatorProcFinish(Processor *proc) {
  allocator()->DestroyCache(&proc->alloc_cache);
  internal_allocator()->DestroyCache(&proc->internal_alloc_cache);
}

void AllocatorPrintStats() {
  allocator()->PrintStats();
}

static void SignalUnsafeCall(ThreadState *thr, uptr pc) {
  if (atomic_load_relaxed(&thr->in_signal_handler) == 0 ||
      !flags()->report_signal_unsafe)
    return;
  VarSizeStackTrace stack;
  ObtainCurrentStack(thr, pc, &stack);
  if (IsFiredSuppression(ctx, ReportTypeSignalUnsafe, stack))
    return;
  ThreadRegistryLock l(ctx->thread_registry);
  ScopedReport rep(ReportTypeSignalUnsafe);
  rep.AddStack(stack, true);
  OutputReport(thr, rep);
}

void *user_alloc(ThreadState *thr, uptr pc, uptr sz, uptr align, bool signal) {
  if ((sz >= (1ull << 40)) || (align >= (1ull << 40)))
    return allocator()->ReturnNullOrDieOnBadRequest();
  void *p = allocator()->Allocate(&thr->proc()->alloc_cache, sz, align);
  if (p == 0)
    return 0;
  if (ctx && ctx->initialized)
    OnUserAlloc(thr, pc, (uptr)p, sz, true);
  if (signal)
    SignalUnsafeCall(thr, pc);
  return p;
}

void *user_calloc(ThreadState *thr, uptr pc, uptr size, uptr n) {
  if (CallocShouldReturnNullDueToOverflow(size, n))
    return allocator()->ReturnNullOrDieOnBadRequest();
  void *p = user_alloc(thr, pc, n * size);
  if (p)
    internal_memset(p, 0, n * size);
  return p;
}

void user_free(ThreadState *thr, uptr pc, void *p, bool signal) {
  ScopedGlobalProcessor sgp;
  if (ctx && ctx->initialized)
    OnUserFree(thr, pc, (uptr)p, true);
  allocator()->Deallocate(&thr->proc()->alloc_cache, p);
  if (signal)
    SignalUnsafeCall(thr, pc);
}

void OnUserAlloc(ThreadState *thr, uptr pc, uptr p, uptr sz, bool write) {
  DPrintf("#%d: alloc(%zu) = %p\n", thr->tid, sz, p);
  ctx->metamap.AllocBlock(thr, pc, p, sz);
  if (write && thr->ignore_reads_and_writes == 0)
    MemoryRangeImitateWrite(thr, pc, (uptr)p, sz);
  else
    MemoryResetRange(thr, pc, (uptr)p, sz);
}

void OnUserFree(ThreadState *thr, uptr pc, uptr p, bool write) {
  CHECK_NE(p, (void*)0);
  uptr sz = ctx->metamap.FreeBlock(thr->proc(), p);
  DPrintf("#%d: free(%p, %zu)\n", thr->tid, p, sz);
  if (write && thr->ignore_reads_and_writes == 0)
    MemoryRangeFreed(thr, pc, (uptr)p, sz);
}

void *user_realloc(ThreadState *thr, uptr pc, void *p, uptr sz) {
  // FIXME: Handle "shrinking" more efficiently,
  // it seems that some software actually does this.
  void *p2 = user_alloc(thr, pc, sz);
  if (p2 == 0)
    return 0;
  if (p) {
    uptr oldsz = user_alloc_usable_size(p);
    internal_memcpy(p2, p, min(oldsz, sz));
    user_free(thr, pc, p);
  }
  return p2;
}

uptr user_alloc_usable_size(const void *p) {
  if (p == 0)
    return 0;
  MBlock *b = ctx->metamap.GetBlock((uptr)p);
  if (!b)
    return 0;  // Not a valid pointer.
  if (b->siz == 0)
    return 1;  // Zero-sized allocations are actually 1 byte.
  return b->siz;
}

void invoke_malloc_hook(void *ptr, uptr size) {
  ThreadState *thr = cur_thread();
  if (ctx == 0 || !ctx->initialized || thr->ignore_interceptors)
    return;
  __sanitizer_malloc_hook(ptr, size);
  RunMallocHooks(ptr, size);
}

void invoke_free_hook(void *ptr) {
  ThreadState *thr = cur_thread();
  if (ctx == 0 || !ctx->initialized || thr->ignore_interceptors)
    return;
  __sanitizer_free_hook(ptr);
  RunFreeHooks(ptr);
}

void *internal_alloc(MBlockType typ, uptr sz) {
  ThreadState *thr = cur_thread();
  if (thr->nomalloc) {
    thr->nomalloc = 0;  // CHECK calls internal_malloc().
    CHECK(0);
  }
  return InternalAlloc(sz, &thr->proc()->internal_alloc_cache);
}

void internal_free(void *p) {
  ThreadState *thr = cur_thread();
  if (thr->nomalloc) {
    thr->nomalloc = 0;  // CHECK calls internal_malloc().
    CHECK(0);
  }
  InternalFree(p, &thr->proc()->internal_alloc_cache);
}

}  // namespace __tsan

using namespace __tsan;

extern "C" {
uptr __sanitizer_get_current_allocated_bytes() {
  uptr stats[AllocatorStatCount];
  allocator()->GetStats(stats);
  return stats[AllocatorStatAllocated];
}

uptr __sanitizer_get_heap_size() {
  uptr stats[AllocatorStatCount];
  allocator()->GetStats(stats);
  return stats[AllocatorStatMapped];
}

uptr __sanitizer_get_free_bytes() {
  return 1;
}

uptr __sanitizer_get_unmapped_bytes() {
  return 1;
}

uptr __sanitizer_get_estimated_allocated_size(uptr size) {
  return size;
}

int __sanitizer_get_ownership(const void *p) {
  return allocator()->GetBlockBegin(p) != 0;
}

uptr __sanitizer_get_allocated_size(const void *p) {
  return user_alloc_usable_size(p);
}

void __tsan_on_thread_idle() {
  ThreadState *thr = cur_thread();
  allocator()->SwallowCache(&thr->proc()->alloc_cache);
  internal_allocator()->SwallowCache(&thr->proc()->internal_alloc_cache);
  ctx->metamap.OnProcIdle(thr->proc());
}
}  // extern "C"

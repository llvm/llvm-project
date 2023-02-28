//===-- trec_mman.cpp
//-----------------------------------------------------===//
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
#include "trec_mman.h"

#include "interception/interception.h"
#include "sanitizer_common/sanitizer_allocator_checks.h"
#include "sanitizer_common/sanitizer_allocator_interface.h"
#include "sanitizer_common/sanitizer_allocator_report.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_errno.h"
#include "sanitizer_common/sanitizer_placement_new.h"
#include "trec_flags.h"
#include "trec_mutex.h"
#include "trec_rtl.h"

// May be overriden by front-end.
SANITIZER_WEAK_DEFAULT_IMPL
void __sanitizer_malloc_hook(void *ptr, uptr size) {
  (void)ptr;
  (void)size;
}

SANITIZER_WEAK_DEFAULT_IMPL
void __sanitizer_free_hook(void *ptr) { (void)ptr; }

namespace __trec {

struct MapUnmapCallback {
  void OnMap(uptr p, uptr size) const {}
  void OnUnmap(uptr p, uptr size) const {}
};

static char allocator_placeholder[sizeof(Allocator)] ALIGNED(64);
Allocator *allocator() {
  return reinterpret_cast<Allocator *>(&allocator_placeholder);
}

struct GlobalProc {
  Mutex mtx;
  Processor *proc;

  GlobalProc() : mtx(MutexTypeGlobalProc), proc(ProcCreate()) {}
};

static char global_proc_placeholder[sizeof(GlobalProc)] ALIGNED(64);
GlobalProc *global_proc() {
  return reinterpret_cast<GlobalProc *>(&global_proc_placeholder);
}

ScopedGlobalProcessor::ScopedGlobalProcessor() {
  GlobalProc *gp = global_proc();
  ThreadState *thr = cur_thread();
  if (thr->proc())
    return;
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

static constexpr uptr kMaxAllowedMallocSize = 1ull << 40;
static uptr max_user_defined_malloc_size;

void InitializeAllocator() {
  SetAllocatorMayReturnNull(common_flags()->allocator_may_return_null);
  allocator()->Init(common_flags()->allocator_release_to_os_interval_ms);
  max_user_defined_malloc_size = common_flags()->max_allocation_size_mb
                                     ? common_flags()->max_allocation_size_mb
                                           << 20
                                     : kMaxAllowedMallocSize;
}

void InitializeAllocatorLate() { new (global_proc()) GlobalProc(); }

void AllocatorProcStart(Processor *proc) {
  allocator()->InitCache(&proc->alloc_cache);
  internal_allocator()->InitCache(&proc->internal_alloc_cache);
}

void AllocatorProcFinish(Processor *proc) {
  allocator()->DestroyCache(&proc->alloc_cache);
  internal_allocator()->DestroyCache(&proc->internal_alloc_cache);
}

void AllocatorPrintStats() { allocator()->PrintStats(); }

static void SignalUnsafeCall(ThreadState *thr, uptr pc) { return; }

void *user_alloc_internal(ThreadState *thr, uptr pc, uptr sz, uptr align,
                          bool signal, bool trace_record, const char *func) {
  if (sz >= kMaxAllowedMallocSize || align >= kMaxAllowedMallocSize ||
      sz > max_user_defined_malloc_size) {
    if (AllocatorMayReturnNull())
      return nullptr;
    uptr malloc_limit =
        Min(kMaxAllowedMallocSize, max_user_defined_malloc_size);
  }
  void *p = allocator()->Allocate(&thr->proc()->alloc_cache, sz, align);
  if (UNLIKELY(!p)) {
    SetAllocatorOutOfMemory();
    if (AllocatorMayReturnNull())
      return nullptr;
  }
  if (ctx && ctx->initialized) {
    // OnUserAlloc(thr, pc, (uptr)p, sz, true, trace_record);
  }
  if (signal)
    SignalUnsafeCall(thr, pc);
  return p;
}

void user_free(ThreadState *thr, uptr pc, void *p, bool signal,
               bool record_trace) {
  ScopedGlobalProcessor sgp;
  if (ctx && ctx->initialized && thr && thr->tctx) {
    OnUserFree(thr, pc, (uptr)p, true,
               record_trace && !thr->ignore_interceptors);
  }
  allocator()->Deallocate(&thr->proc()->alloc_cache, p);
  if (signal)
    SignalUnsafeCall(thr, pc);
}

void *user_alloc(ThreadState *thr, uptr pc, uptr sz) {
  return SetErrnoOnNull(user_alloc_internal(thr, pc, sz, kDefaultAlignment,
                                            true, true, __func__));
}

void *user_calloc(ThreadState *thr, uptr pc, uptr size, uptr n) {
  if (UNLIKELY(CheckForCallocOverflow(size, n))) {
    if (AllocatorMayReturnNull())
      return SetErrnoOnNull(nullptr);
  }
  void *p = user_alloc_internal(thr, pc, n * size, kDefaultAlignment, true,
                                true, __func__);
  if (p)
    internal_memset(p, 0, n * size);
  return SetErrnoOnNull(p);
}

void *user_reallocarray(ThreadState *thr, uptr pc, void *p, uptr size, uptr n) {
  if (UNLIKELY(CheckForCallocOverflow(size, n))) {
    if (AllocatorMayReturnNull())
      return SetErrnoOnNull(nullptr);
  }
  return user_realloc(thr, pc, p, size * n);
}

bool IsInternalMem(uptr p, uptr sz) {
  return (p == 0x7b0c00000000) || (p == 0x7b1c00000000) ||
         (p == 0x7b0c00000030) || (p == 0x7bc400000000) ||
         // printf cache sz=1024?
         (p == 0x7b6000000000) ||
         // thread state ?
         (sz == 304 && p >= 0x7b4400000000 &&
          (p - 0x7b4400000000) % 0x140 == 0);
}

void OnUserAlloc(ThreadState *thr, uptr pc, uptr p, uptr sz, bool write,
                 bool record_trace) {
  DPrintf("#%d: alloc(%zu) = %p\n", thr->tid, sz, p);
  if (record_trace && LIKELY(ctx->flags.record_alloc_free) && thr &&
      thr->tctx && LIKELY(ctx->flags.output_trace) &&
      LIKELY(thr->ignore_interceptors == 0) && LIKELY(thr->should_record)) {
    if (ctx->flags.trace_mode == 2 || ctx->flags.trace_mode == 3) {
      __trec_trace::Event e(
          __trec_trace::EventType::MemAlloc, thr->tid,
          atomic_fetch_add(&ctx->global_id, 1, memory_order_relaxed),
          ((sz & 0xffff) << 48) | (p & ((((u64)1) << 48) - 1)), 0, pc);

      thr->tctx->put_trace(&e, sizeof(__trec_trace::Event));
      thr->tctx->header.StateInc(__trec_header::RecordType::MemAlloc);
    }
  }
}

void OnUserFree(ThreadState *thr, uptr pc, uptr p, bool write,
                bool record_trace) {
  CHECK_NE(p, (void *)0);
  uptr sz = user_alloc_usable_size((void *)p);
  DPrintf("#%d: free(%p, %zu)\n", thr->tid, p, sz);
  if (record_trace && LIKELY(thr->ignore_interceptors == 0) &&
      LIKELY(thr->should_record)) {
    __trec_trace::Event e(
        __trec_trace::EventType::MemFree, thr->tid,
        atomic_fetch_add(&ctx->global_id, 1, memory_order_relaxed),
        ((sz & 0xffff) << 48) | (p & ((((u64)1) << 48) - 1)),
        thr->tctx->metadata_offset, pc);

    __trec_metadata::MemFreeMeta meta(0x8000, 1);
    thr->tctx->put_metadata(&meta, sizeof(meta));
    thr->tctx->put_trace(&e, sizeof(__trec_trace::Event));
    thr->tctx->header.StateInc(__trec_header::RecordType::MemFree);
  }
}

void *user_realloc(ThreadState *thr, uptr pc, void *p, uptr sz) {
  // FIXME: Handle "shrinking" more efficiently,
  // it seems that some software actually does this.
  if (!p)
    return SetErrnoOnNull(user_alloc_internal(thr, pc, sz, kDefaultAlignment,
                                              true, true, __func__));
  if (!sz) {
    user_free(thr, pc, p, true, true);
    return nullptr;
  }
  void *new_p =
      user_alloc_internal(thr, pc, sz, kDefaultAlignment, true, true, __func__);
  if (new_p) {
    uptr old_sz = user_alloc_usable_size(p);
    internal_memcpy(new_p, p, min(old_sz, sz));
    user_free(thr, pc, p, true, true);
  }
  return SetErrnoOnNull(new_p);
}

void *user_memalign(ThreadState *thr, uptr pc, uptr align, uptr sz) {
  if (UNLIKELY(!IsPowerOfTwo(align))) {
    errno = errno_EINVAL;
    if (AllocatorMayReturnNull())
      return nullptr;
  }
  return SetErrnoOnNull(
      user_alloc_internal(thr, pc, sz, align, true, true, __func__));
}

int user_posix_memalign(ThreadState *thr, uptr pc, void **memptr, uptr align,
                        uptr sz) {
  if (UNLIKELY(!CheckPosixMemalignAlignment(align))) {
    if (AllocatorMayReturnNull())
      return errno_EINVAL;
  }
  void *ptr = user_alloc_internal(thr, pc, sz, align, true, true, __func__);
  if (UNLIKELY(!ptr))
    // OOM error is already taken care of by user_alloc_internal.
    return errno_ENOMEM;
  CHECK(IsAligned((uptr)ptr, align));
  *memptr = ptr;
  return 0;
}

void *user_aligned_alloc(ThreadState *thr, uptr pc, uptr align, uptr sz) {
  if (UNLIKELY(!CheckAlignedAllocAlignmentAndSize(align, sz))) {
    errno = errno_EINVAL;
    if (AllocatorMayReturnNull())
      return nullptr;
  }
  return SetErrnoOnNull(
      user_alloc_internal(thr, pc, sz, align, true, true, __func__));
}

void *user_valloc(ThreadState *thr, uptr pc, uptr sz) {
  return SetErrnoOnNull(user_alloc_internal(thr, pc, sz, GetPageSizeCached(),
                                            true, true, __func__));
}

void *user_pvalloc(ThreadState *thr, uptr pc, uptr sz) {
  uptr PageSize = GetPageSizeCached();
  if (UNLIKELY(CheckForPvallocOverflow(sz, PageSize))) {
    errno = errno_ENOMEM;
    if (AllocatorMayReturnNull())
      return nullptr;
  }
  // pvalloc(0) should allocate one page.
  sz = sz ? RoundUpTo(sz, PageSize) : PageSize;
  return SetErrnoOnNull(
      user_alloc_internal(thr, pc, sz, PageSize, true, true, __func__));
}

uptr user_alloc_usable_size(void *p) {
  return allocator()->GetActuallyAllocatedSize(p);
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

}  // namespace __trec

using namespace __trec;

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

uptr __sanitizer_get_free_bytes() { return 1; }

uptr __sanitizer_get_unmapped_bytes() { return 1; }

uptr __sanitizer_get_estimated_allocated_size(uptr size) { return size; }

int __sanitizer_get_ownership(const void *p) {
  return allocator()->GetBlockBegin(p) != 0;
}

uptr __sanitizer_get_allocated_size(const void *p) {
  return user_alloc_usable_size((void *)p);
}

void __trec_on_thread_idle() {
  ThreadState *thr = cur_thread();
  allocator()->SwallowCache(&thr->proc()->alloc_cache);
  internal_allocator()->SwallowCache(&thr->proc()->internal_alloc_cache);
}
}  // extern "C"

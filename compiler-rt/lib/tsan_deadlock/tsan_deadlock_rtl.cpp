//===-- tsan_deadlock_rtl.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of DeadlockSanitizer.
//
//===----------------------------------------------------------------------===//

#include "tsan_deadlock_rtl.h"

#include "sanitizer_common/sanitizer_allocator_internal.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_flag_parser.h"
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_placement_new.h"
#include "sanitizer_common/sanitizer_report_decorator.h"
#include "sanitizer_common/sanitizer_stackdepot.h"
#include "sanitizer_common/sanitizer_stacktrace.h"

#if SANITIZER_APPLE
#include <pthread.h>
#endif

namespace __tsan_deadlock {

Flags tsan_deadlock_flags;

static Context *ctx;
static atomic_uint32_t error_count;

static const uptr kShadowStackSize = 64 * 1024;

#if SANITIZER_APPLE
static char main_thread_state[sizeof(Thread)];
static pthread_key_t thread_key;

static void ThreadKeyDestructor(void *thr) {
  pthread_setspecific(thread_key, thr);
}

static void InitializeThreadStorage() {
  CHECK_EQ(thread_key, 0);
  int res = pthread_key_create(&thread_key, ThreadKeyDestructor);
  CHECK_EQ(res, 0);
  res = pthread_setspecific(thread_key, main_thread_state);
  CHECK_EQ(res, 0);
}

Thread *cur_thread() {
  if (UNLIKELY(!thread_key))
    return (Thread *)main_thread_state;
  Thread *thr = (Thread *)pthread_getspecific(thread_key);
  if (UNLIKELY(!thr)) {
    thr = (Thread *)InternalAlloc(sizeof(Thread));
    internal_memset(thr, 0, sizeof(Thread));
    int res = pthread_setspecific(thread_key, thr);
    CHECK_EQ(res, 0);
  }
  return thr;
}

void set_cur_thread(Thread *thr) { pthread_setspecific(thread_key, thr); }
#else
__attribute__((tls_model("initial-exec"))) THREADLOCAL Thread thr_tls;
#endif

class Decorator : public __sanitizer::SanitizerCommonDecorator {
public:
  Decorator() : SanitizerCommonDecorator() {}
  const char *Mutex() { return Magenta(); }
  const char *ThreadDescription() { return Cyan(); }
};

static u32 CurrentStackId(Thread *thr, uptr pc) {
  if (pc) {
    DCHECK_LT(thr->shadow_stack_pos, thr->shadow_stack_end);
    *thr->shadow_stack_pos = pc;
    thr->shadow_stack_pos++;
  }
  u32 id = StackDepotPut(
      StackTrace(thr->shadow_stack, thr->shadow_stack_pos - thr->shadow_stack));
  if (pc)
    thr->shadow_stack_pos--;
  return id;
}

static void PrintStack(Thread *thr, u32 stk) {
  if (!stk)
    return;
  StackTrace trace = StackDepotGet(stk);
  thr->ignore_interceptors++;
  trace.Print();
  thr->ignore_interceptors--;
}

static void ReportDeadlock(Thread *thr, DDReport *rep) {
  if (!rep)
    return;
  Decorator d;
  Lock lock(&ctx->report_mutex);
  Printf("==================\n");
  Printf(
      "%sWARNING: %s: lock-order-inversion (potential deadlock) (pid=%d)%s\n",
      d.Warning(), SanitizerToolName, (int)internal_getpid(), d.Default());
  Printf("  Cycle in lock order graph: ");
  for (int i = 0; i < rep->n; i++)
    Printf("%sM%d%s (0x%zx) => ", d.Mutex(), i, d.Default(),
           (uptr)rep->loop[i].mtx_ctx0);
  Printf("%sM0%s\n\n", d.Mutex(), d.Default());
  for (int i = 0; i < rep->n; i++) {
    int next = (i + 1) % rep->n;
    Printf("  %sMutex M%d%s acquired here while holding %sMutex M%d%s"
           " in %sthread T%lld%s:\n",
           d.Mutex(), next, d.Default(), d.Mutex(), i, d.Default(),
           d.ThreadDescription(), (s64)rep->loop[i].thr_ctx, d.Default());
    PrintStack(thr, rep->loop[i].stk[0]);
    if (flags()->second_deadlock_stack && rep->loop[i].stk[1]) {
      Printf("  %sMutex M%d%s previously acquired by the same thread here:\n",
             d.Mutex(), i, d.Default());
      PrintStack(thr, rep->loop[i].stk[1]);
    }
    if (i == 0 && !flags()->second_deadlock_stack) {
      Printf(
          "    HINT: use TSAN_DEADLOCK_OPTIONS=second_deadlock_stack=1 to get "
          "more informative warning message\n\n");
    }
  }
  Printf("==================\n");
  StackTrace empty;
  ReportErrorSummary("lock-order-inversion", &empty);
  atomic_fetch_add(&error_count, 1, memory_order_relaxed);
  if (flags()->halt_on_error)
    Die();
}

static void ReportMutexMisuse(Thread *thr, const char *what, uptr addr, uptr pc,
                              u32 creation_stk, u32 last_lock_stk = 0) {
  Decorator d;
  Lock lock(&ctx->report_mutex);
  Printf("==================\n");
  Printf("%sWARNING: %s: %s on %smutex 0x%zx%s (pid=%d)%s\n", d.Warning(),
         SanitizerToolName, what, d.Mutex(), addr, d.Default(),
         (int)internal_getpid(), d.Default());
  PrintStack(thr, CurrentStackId(thr, pc));
  if (last_lock_stk) {
    Printf("  and:\n");
    PrintStack(thr, last_lock_stk);
  }
  if (creation_stk) {
    Printf("%s  Mutex M0 (%p) created at:%s\n", d.Mutex(),
           reinterpret_cast<void *>(addr), d.Default());
    PrintStack(thr, creation_stk);
  }
  Printf("==================\n");
  StackTrace empty;
  ReportErrorSummary(what, &empty);
  atomic_fetch_add(&error_count, 1, memory_order_relaxed);
  if (flags()->halt_on_error)
    Die();
}

Callback::Callback(Thread *thr, uptr pc) : thr(thr), pc(pc) {
  DDCallback::pt = thr->dd_pt;
  DDCallback::lt = thr->dd_lt;
}

u32 Callback::Unwind() { return CurrentStackId(thr, pc); }

int Callback::UniqueTid() { return thr->unique_id; }

static void InitializeFlags() {
  Flags *f = flags();
  f->SetDefaults();
  SetCommonFlagsDefaults();
  {
    CommonFlags cf;
    cf.CopyFrom(*common_flags());
    cf.allow_addr2line = true;
    cf.exitcode = 66;
    OverrideCommonFlags(cf);
  }
  FlagParser parser;
  RegisterFlag(&parser, "second_deadlock_stack",
               "Report where each mutex is locked in deadlock reports",
               &f->second_deadlock_stack);
  RegisterFlag(&parser, "report_destroy_locked",
               "Report destruction of a locked mutex?",
               &f->report_destroy_locked);
  RegisterFlag(&parser, "report_mutex_bugs",
               "Report incorrect usages of mutexes and mutex annotations?",
               &f->report_mutex_bugs);
  RegisterFlag(&parser, "report_bugs",
               "Turns off bug reporting entirely (useful for benchmarking).",
               &f->report_bugs);
  RegisterFlag(&parser, "halt_on_error", "Exit after first reported error.",
               &f->halt_on_error);
  RegisterCommonFlags(&parser);
  parser.ParseStringFromEnv("TSAN_DEADLOCK_OPTIONS");
  if (!f->report_bugs) {
    f->report_destroy_locked = false;
    f->report_mutex_bugs = false;
  }
  SetVerbosity(common_flags()->verbosity);
}

static void Finalize() {
  if (atomic_load(&error_count, memory_order_relaxed) > 0)
    internal__exit(common_flags()->exitcode);
}

void Initialize() {
  static atomic_uint32_t initialized;
  if (atomic_fetch_add(&initialized, 1, memory_order_relaxed) != 0)
    return;
  static u64 ctx_mem[sizeof(Context) / sizeof(u64) + 1];
  ctx = new (ctx_mem) Context();
  SanitizerToolName = "DeadlockSanitizer";
  InitializeFlags();
  ctx->dd = DDetector::Create(flags());
#if SANITIZER_APPLE
  InitializeThreadStorage();
#endif
  InitializeInterceptors();
  Atexit(Finalize);
}

void ThreadInit(Thread *thr) {
  static atomic_uintptr_t id_gen;
  uptr id = atomic_fetch_add(&id_gen, 1, memory_order_relaxed);
  thr->unique_id = (int)id;
  thr->dd_pt = ctx->dd->CreatePhysicalThread();
  thr->dd_lt = ctx->dd->CreateLogicalThread(id);
  thr->shadow_stack = static_cast<uptr *>(
      MmapNoReserveOrDie(kShadowStackSize * sizeof(uptr), "shadow stack"));
  thr->shadow_stack_pos = thr->shadow_stack;
  thr->shadow_stack_end = thr->shadow_stack + kShadowStackSize;
  thr->is_inited = true;
}

void ThreadDestroy(Thread *thr) {
  thr->is_inited = false;
  ctx->dd->DestroyLogicalThread(thr->dd_lt);
  thr->dd_lt = nullptr;
  UnmapOrDie(thr->shadow_stack, kShadowStackSize * sizeof(uptr));
  thr->shadow_stack = nullptr;
  thr->shadow_stack_pos = nullptr;
  thr->shadow_stack_end = nullptr;
  ctx->dd->DestroyPhysicalThread(thr->dd_pt);
  thr->dd_pt = nullptr;
#if SANITIZER_APPLE
  if (thr != (Thread *)main_thread_state) {
    set_cur_thread(nullptr);
    InternalFree(thr);
  }
#endif
}

static void EnsureMutexInit(Callback *cb, MutexHashMap::Handle *h, uptr m,
                            uptr pc) {
  if (h->created()) {
    ctx->dd->MutexInit(cb, &(*h)->dd);
    (*h)->dd.ctx = m;
    (*h)->creation_stk = CurrentStackId(cb->thr, pc);
  }
}

void MutexInit(Thread *thr, uptr m, bool reentrant, uptr pc) {
  if (!thr || thr->ignore_interceptors)
    return;
  Callback cb(thr, pc);
  MutexHashMap::Handle h(&ctx->mutex_map, m);
  EnsureMutexInit(&cb, &h, m, pc);
  h->reentrant = reentrant;
}

void MutexBeforeLock(Thread *thr, uptr m, bool writelock, uptr pc) {
  if (!thr || thr->ignore_interceptors)
    return;
  Callback cb(thr, pc);
  bool report_double_lock = false;
  {
    MutexHashMap::Handle h(&ctx->mutex_map, m);
    EnsureMutexInit(&cb, &h, m, pc);
    if (writelock) {
      uptr owner = atomic_load(&h->owner_tid, memory_order_relaxed);
      if (owner == (uptr)thr->dd_lt && !h->reentrant)
        report_double_lock = true;
      else if (owner != (uptr)thr->dd_lt)
        ctx->dd->MutexBeforeLock(&cb, &h->dd, writelock);
    } else {
      ctx->dd->MutexBeforeLock(&cb, &h->dd, writelock);
    }
  }
  if (report_double_lock && flags()->report_mutex_bugs) {
    MutexHashMap::Handle h(&ctx->mutex_map, m, /*remove=*/false,
                           /*create=*/false);
    u32 cstk = h.exists() ? h->creation_stk : 0;
    ReportMutexMisuse(thr, "double lock of a mutex", m, pc, cstk);
  } else if (flags()->report_bugs) {
    ReportDeadlock(thr, ctx->dd->GetReport(&cb));
  }
}

void MutexAfterLock(Thread *thr, uptr m, bool writelock, bool trylock,
                    uptr pc) {
  if (!thr || thr->ignore_interceptors)
    return;
  Callback cb(thr, pc);
  {
    MutexHashMap::Handle h(&ctx->mutex_map, m);
    EnsureMutexInit(&cb, &h, m, pc);
    if (writelock) {
      uptr owner = atomic_load(&h->owner_tid, memory_order_relaxed);
      if (owner == (uptr)thr->dd_lt) {
        DCHECK(h->reentrant);
        h->recursion++;
      } else {
        DCHECK_EQ(owner, 0u);
        CHECK_EQ(h->recursion, 0);
        h->recursion = 1;
        h->last_lock_stk = CurrentStackId(thr, pc);
        atomic_store(&h->owner_tid, (uptr)thr->dd_lt, memory_order_relaxed);
        ctx->dd->MutexAfterLock(&cb, &h->dd, writelock, trylock);
      }
    } else {
      ctx->dd->MutexAfterLock(&cb, &h->dd, writelock, trylock);
    }
  }
  if (flags()->report_bugs)
    ReportDeadlock(thr, ctx->dd->GetReport(&cb));
}

void MutexBeforeUnlock(Thread *thr, uptr m, bool writelock, uptr pc) {
  if (!thr || thr->ignore_interceptors)
    return;
  Callback cb(thr, pc);
  bool report_bad_unlock = false;
  {
    MutexHashMap::Handle h(&ctx->mutex_map, m, /*remove=*/false,
                           /*create=*/false);
    if (!h.exists())
      return;
    if (writelock) {
      uptr owner = atomic_load(&h->owner_tid, memory_order_relaxed);
      if (owner != (uptr)thr->dd_lt) {
        report_bad_unlock = true;
      } else {
        h->recursion--;
        if (h->recursion == 0) {
          atomic_store(&h->owner_tid, 0, memory_order_relaxed);
          ctx->dd->MutexBeforeUnlock(&cb, &h->dd, writelock);
        }
      }
    } else {
      ctx->dd->MutexBeforeUnlock(&cb, &h->dd, writelock);
    }
  }
  if (report_bad_unlock && flags()->report_mutex_bugs) {
    MutexHashMap::Handle h(&ctx->mutex_map, m, /*remove=*/false,
                           /*create=*/false);
    u32 cstk = h.exists() ? h->creation_stk : 0;
    ReportMutexMisuse(thr, "unlock of an unlocked mutex (or by a wrong thread)",
                      m, pc, cstk);
  } else if (flags()->report_bugs) {
    ReportDeadlock(thr, ctx->dd->GetReport(&cb));
  }
}

void MutexDestroy(Thread *thr, uptr m, uptr pc) {
  if (!thr || thr->ignore_interceptors)
    return;
  Callback cb(thr, pc);
  bool report_destroy_locked = false;
  u32 creation_stk = 0;
  u32 last_lock_stk = 0;
  {
    MutexHashMap::Handle h(&ctx->mutex_map, m, /*remove=*/true);
    if (!h.exists())
      return;
    creation_stk = h->creation_stk;
    last_lock_stk = h->last_lock_stk;
    if (atomic_load(&h->owner_tid, memory_order_relaxed) != 0)
      report_destroy_locked = true;
    ctx->dd->MutexDestroy(&cb, &h->dd);
  }
  if (report_destroy_locked && flags()->report_destroy_locked)
    ReportMutexMisuse(thr, "destroy of a locked mutex", m, pc, creation_stk,
                      last_lock_stk);
}

} // namespace __tsan_deadlock

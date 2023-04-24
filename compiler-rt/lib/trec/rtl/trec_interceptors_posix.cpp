//===-- trec_interceptors_posix.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of TraceRecorder (TRec), a race detector.
//
// FIXME: move as many interceptors as possible into
// sanitizer_common/sanitizer_common_interceptors.inc
//===----------------------------------------------------------------------===//

#include <dlfcn.h>
#include <stdarg.h>

#include "interception/interception.h"
#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_errno.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_linux.h"
#include "sanitizer_common/sanitizer_placement_new.h"
#include "sanitizer_common/sanitizer_platform_limits_netbsd.h"
#include "sanitizer_common/sanitizer_platform_limits_posix.h"
#include "sanitizer_common/sanitizer_posix.h"
#include "sanitizer_common/sanitizer_stacktrace.h"
#include "sanitizer_common/sanitizer_tls_get_addr.h"
#include "trec_interceptors.h"
#include "trec_interface.h"
#include "trec_platform.h"
#include "trec_rtl.h"

using namespace __trec;

#if SANITIZER_FREEBSD || SANITIZER_MAC
#define stdout __stdoutp
#define stderr __stderrp
#endif

#if SANITIZER_NETBSD
#define dirfd(dirp) (*(int *)(dirp))
#define fileno_unlocked(fp)              \
  (((__sanitizer_FILE *)fp)->_file == -1 \
       ? -1                              \
       : (int)(unsigned short)(((__sanitizer_FILE *)fp)->_file))

#define stdout ((__sanitizer_FILE *)&__sF[1])
#define stderr ((__sanitizer_FILE *)&__sF[2])

#define vfork __vfork14
#endif

#ifdef __mips__
const int kSigCount = 129;
#else
const int kSigCount = 65;
#endif

#ifdef __mips__
struct ucontext_t {
  u64 opaque[768 / sizeof(u64) + 1];
};
#else
struct ucontext_t {
  // The size is determined by looking at sizeof of real ucontext_t on linux.
  u64 opaque[936 / sizeof(u64) + 1];
};
#endif

#if defined(__i386__) || defined(__riscv) || defined(__x86_64__) || \
    defined(__mips__) || SANITIZER_PPC64V1
#define PTHREAD_ABI_BASE "GLIBC_2.3.2"
#elif defined(__aarch64__) || SANITIZER_PPC64V2
#define PTHREAD_ABI_BASE "GLIBC_2.17"
#endif

extern "C" int pthread_attr_init(void *attr);
extern "C" int pthread_attr_destroy(void *attr);
extern "C" int pthread_attr_getdetachstate(void *attr, int *v);
extern "C" int pthread_attr_setstacksize(void *attr, uptr stacksize);
extern "C" int pthread_key_create(unsigned *key, void (*destructor)(void *v));
extern "C" int pthread_setspecific(unsigned key, const void *v);
DECLARE_REAL(int, pthread_mutexattr_gettype, void *, void *)
extern "C" int fflush(__sanitizer_FILE *fp);
DECLARE_REAL_AND_INTERCEPTOR(void *, malloc, uptr size)
DECLARE_REAL_AND_INTERCEPTOR(void, free, void *ptr)
extern "C" void *pthread_self();
#if !SANITIZER_NETBSD
extern "C" int fileno_unlocked(void *stream);
extern "C" int dirfd(void *dirp);
#endif
#if SANITIZER_GLIBC
extern "C" int mallopt(int param, int value);
#endif
#if SANITIZER_NETBSD
extern __sanitizer_FILE __sF[];
#else
extern __sanitizer_FILE *stdout, *stderr;
#endif
#if !SANITIZER_FREEBSD && !SANITIZER_MAC && !SANITIZER_NETBSD
const int PTHREAD_MUTEX_RECURSIVE = 1;
const int PTHREAD_MUTEX_RECURSIVE_NP = 1;
#else
const int PTHREAD_MUTEX_RECURSIVE = 2;
const int PTHREAD_MUTEX_RECURSIVE_NP = 2;
#endif
#if !SANITIZER_FREEBSD && !SANITIZER_MAC && !SANITIZER_NETBSD
const int EPOLL_CTL_ADD = 1;
#endif
void *const MAP_FAILED = (void *)-1;
#if SANITIZER_NETBSD
const int PTHREAD_BARRIER_SERIAL_THREAD = 1234567;
#elif !SANITIZER_MAC
const int PTHREAD_BARRIER_SERIAL_THREAD = -1;
#endif
const int MAP_FIXED = 0x10;
typedef long long_t;
typedef __sanitizer::u16 mode_t;

// From /usr/include/unistd.h
#define F_ULOCK 0 /* Unlock a previously locked region.  */
#define F_LOCK 1  /* Lock a region for exclusive use.  */
#define F_TLOCK 2 /* Test and lock a region for exclusive use.  */
#define F_TEST 3  /* Test a region for other processes locks.  */

#if SANITIZER_FREEBSD || SANITIZER_MAC || SANITIZER_NETBSD
const int SA_SIGINFO = 0x40;
const int SIG_SETMASK = 3;
#elif defined(__mips__)
const int SA_SIGINFO = 8;
const int SIG_SETMASK = 3;
#else
const int SA_SIGINFO = 4;
const int SIG_SETMASK = 2;
#endif

#define COMMON_INTERCEPTOR_NOTHING_IS_INITIALIZED \
  (cur_thread_init(), !cur_thread()->is_inited)

namespace __trec {
struct SignalDesc {
  bool armed;
  bool sigaction;
  __sanitizer_siginfo siginfo;
  ucontext_t ctx;
};

struct ThreadSignalContext {
  int int_signal_send;
  atomic_uintptr_t in_blocking_func;
  atomic_uintptr_t have_pending_signals;
  SignalDesc pending_signals[kSigCount];
  // emptyset and oldset are too big for stack.
  __sanitizer_sigset_t emptyset;
  __sanitizer_sigset_t oldset;
};

// The sole reason trec wraps atexit callbacks is to establish synchronization
// between callback setup and callback execution.
struct AtExitCtx {
  void (*f)();
  void *arg;
};

// InterceptorContext holds all global data required for interceptors.
// It's explicitly constructed in InitializeInterceptors with placement new
// and is never destroyed. This allows usage of members with non-trivial
// constructors and destructors.
struct InterceptorContext {
  // The object is 64-byte aligned, because we want hot data to be located
  // in a single cache line if possible (it's accessed in every interceptor).
  ALIGNED(64) LibIgnore libignore;
  __sanitizer_sigaction sigactions[kSigCount];
#if !SANITIZER_MAC && !SANITIZER_NETBSD
  unsigned finalize_key;
#endif

  Mutex atexit_mu;
  Vector<struct AtExitCtx *> AtExitStack;

  InterceptorContext() : libignore(LINKER_INITIALIZED), AtExitStack() {}
};

static ALIGNED(64) char interceptor_placeholder[sizeof(InterceptorContext)];
InterceptorContext *interceptor_ctx() {
  return reinterpret_cast<InterceptorContext *>(&interceptor_placeholder[0]);
}

}  // namespace __trec

static ThreadSignalContext *SigCtx(ThreadState *thr) {
  ThreadSignalContext *ctx = (ThreadSignalContext *)thr->signal_ctx;
  if (ctx == 0 && !thr->is_dead) {
    ctx = (ThreadSignalContext *)MmapOrDie(sizeof(*ctx), "ThreadSignalContext");
    thr->signal_ctx = ctx;
  }
  return ctx;
}

ScopedInterceptor::ScopedInterceptor(ThreadState *thr, const char *fname,
                                     uptr pc)
    : thr_(thr), pc_(pc), should_record(false) {
  Initialize(thr);
  if (!thr_->is_inited)
    return;
  should_record = thr->should_record;

  if (!thr_->ignore_interceptors) {
    bool should_record_ret = false;
    RecordFuncEntry(thr, should_record_ret, fname, pc);
    thr->should_record &= should_record_ret;
  }
   if (!thr_->ignore_interceptors) {
    bool should_record_ret = false;
    RecordBBLEntry(thr, should_record_ret);
    thr->should_record &= should_record_ret;
  }
  if (internal_strcmp(fname, "pthread_create")) {
    thr->tctx->isFuncEnterMetaVaild = false;
    thr->tctx->isFuncExitMetaVaild = false;
    thr->tctx->parammetas.Resize(0);
    thr->tctx->dbg_temp_buffer_size = 0;
  }

  internal_strlcpy(func_name, fname, 255);
}

ScopedInterceptor::~ScopedInterceptor() {
  if (!thr_->is_inited)
    return;
  if (!thr_->ignore_interceptors) {
    ProcessPendingSignals(thr_);
    RecordFuncExit(thr_, thr_->should_record, func_name);
  }
  thr_->should_record = should_record;
  thr_->tctx->isFuncEnterMetaVaild = false;
  thr_->tctx->isFuncExitMetaVaild = false;
  thr_->tctx->parammetas.Resize(0);
  thr_->tctx->dbg_temp_buffer_size = 0;
}

#define TREC_INTERCEPT(func) INTERCEPT_FUNCTION(func);
#if SANITIZER_FREEBSD
#define TREC_INTERCEPT_VER(func, ver) INTERCEPT_FUNCTION(func)
#define TREC_MAYBE_INTERCEPT_NETBSD_ALIAS(func)
#define TREC_MAYBE_INTERCEPT_NETBSD_ALIAS_THR(func)
#elif SANITIZER_NETBSD
#define TREC_INTERCEPT_VER(func, ver) INTERCEPT_FUNCTION(func)
#define TREC_MAYBE_INTERCEPT_NETBSD_ALIAS(func) \
  INTERCEPT_FUNCTION(__libc_##func)
#define TREC_MAYBE_INTERCEPT_NETBSD_ALIAS_THR(func) \
  INTERCEPT_FUNCTION(__libc_thr_##func)
#else
#define TREC_INTERCEPT_VER(func, ver) INTERCEPT_FUNCTION_VER(func, ver)
#define TREC_MAYBE_INTERCEPT_NETBSD_ALIAS(func)
#define TREC_MAYBE_INTERCEPT_NETBSD_ALIAS_THR(func)
#endif

#define BLOCK_REAL(name) (BlockingCall(thr), REAL(name))

struct BlockingCall {
  explicit BlockingCall(ThreadState *thr) : thr(thr), ctx(SigCtx(thr)) {
    for (;;) {
      atomic_store(&ctx->in_blocking_func, 1, memory_order_relaxed);
      if (atomic_load(&ctx->have_pending_signals, memory_order_relaxed) == 0)
        break;
      atomic_store(&ctx->in_blocking_func, 0, memory_order_relaxed);
      ProcessPendingSignals(thr);
    }
    // When we are in a "blocking call", we process signals asynchronously
    // (right when they arrive). In this context we do not expect to be
    // executing any user/runtime code. The known interceptor sequence when
    // this is not true is: pthread_join -> munmap(stack). It's fine
    // to ignore munmap in this case -- we handle stack shadow separately.
    thr->ignore_interceptors++;
  }

  ~BlockingCall() {
    thr->ignore_interceptors--;
    atomic_store(&ctx->in_blocking_func, 0, memory_order_relaxed);
  }

  ThreadState *thr;
  ThreadSignalContext *ctx;
};

TREC_INTERCEPTOR(int, pause, int fake) {
  SCOPED_TREC_INTERCEPTOR(pause, fake);
  return BLOCK_REAL(pause)(fake);
}

static void at_exit_wrapper() {
  AtExitCtx *ctx;
  {
    // Ensure thread-safety.
    interceptor_ctx()->atexit_mu.Lock();

    // Pop AtExitCtx from the top of the stack of callback functions
    uptr element = interceptor_ctx()->AtExitStack.Size() - 1;
    ctx = interceptor_ctx()->AtExitStack[element];
    interceptor_ctx()->AtExitStack.PopBack();
    interceptor_ctx()->atexit_mu.Unlock();
  }
  ((void (*)())ctx->f)();
  InternalFree(ctx);
}

static void cxa_at_exit_wrapper(void *arg) {
  AtExitCtx *ctx = (AtExitCtx *)arg;
  ((void (*)(void *arg))ctx->f)(ctx->arg);
  InternalFree(ctx);
}

static int setup_at_exit_wrapper(ThreadState *thr, uptr pc, void (*f)(),
                                 void *arg, void *dso);

#if !SANITIZER_ANDROID
TREC_INTERCEPTOR(int, atexit, void (*f)()) {
  // We want to setup the atexit callback even if we are in ignored lib
  // or after fork.
  SCOPED_INTERCEPTOR_RAW(atexit, f);
  return setup_at_exit_wrapper(thr, pc, (void (*)())f, 0, 0);
}
#endif

TREC_INTERCEPTOR(int, __cxa_atexit, void (*f)(void *a), void *arg, void *dso) {
  SCOPED_TREC_INTERCEPTOR(__cxa_atexit, f, arg, dso);
  return setup_at_exit_wrapper(thr, pc, (void (*)())f, arg, dso);
}

static int setup_at_exit_wrapper(ThreadState *thr, uptr pc, void (*f)(),
                                 void *arg, void *dso) {
  AtExitCtx *ctx = (AtExitCtx *)InternalAlloc(sizeof(AtExitCtx));
  ctx->f = f;
  ctx->arg = arg;
  // Memory allocation in __cxa_atexit will race with free during exit,
  // because we do not see synchronization around atexit callback list.
  int res;
  if (!dso) {
    // NetBSD does not preserve the 2nd argument if dso is equal to 0
    // Store ctx in a local stack-like structure

    res = REAL(__cxa_atexit)((void (*)(void *a))at_exit_wrapper, 0, 0);
    // Push AtExitCtx on the top of the stack of callback functions
    if (!res) {
      interceptor_ctx()->AtExitStack.PushBack(ctx);
    }
  } else {
    res = REAL(__cxa_atexit)(cxa_at_exit_wrapper, ctx, dso);
  }
  return res;
}

#if !SANITIZER_MAC && !SANITIZER_NETBSD
static void on_exit_wrapper(int status, void *arg) {
  ThreadState *thr = cur_thread();
  uptr pc = 0;
  AtExitCtx *ctx = (AtExitCtx *)arg;
  ((void (*)(int status, void *arg))ctx->f)(status, ctx->arg);
  InternalFree(ctx);
}
TREC_INTERCEPTOR(int, on_exit, void (*f)(int, void *), void *arg) {
  SCOPED_TREC_INTERCEPTOR(on_exit, f, arg);
  AtExitCtx *ctx = (AtExitCtx *)InternalAlloc(sizeof(AtExitCtx));
  ctx->f = (void (*)())f;
  ctx->arg = arg;
  // Memory allocation in __cxa_atexit will race with free during exit,
  // because we do not see synchronization around atexit callback list.
  int res = REAL(on_exit)(on_exit_wrapper, ctx);
  return res;
}
#define TREC_MAYBE_INTERCEPT_ON_EXIT TREC_INTERCEPT(on_exit)
#else
#define TREC_MAYBE_INTERCEPT_ON_EXIT
#endif

// __cxa_guard_acquire and friends need to be intercepted in a special way -
// regular interceptors will break statically-linked libstdc++. Linux
// interceptors are especially defined as weak functions (so that they don't
// cause link errors when user defines them as well). So they silently
// auto-disable themselves when such symbol is already present in the binary. If
// we link libstdc++ statically, it will bring own __cxa_guard_acquire which
// will silently replace our interceptor.  That's why on Linux we simply export
// these interceptors with INTERFACE_ATTRIBUTE.
// On OS X, we don't support statically linking, so we just use a regular
// interceptor.
#if SANITIZER_MAC
#define STDCXX_INTERCEPTOR TREC_INTERCEPTOR
#else
#define STDCXX_INTERCEPTOR(rettype, name, ...) \
  extern "C" rettype INTERFACE_ATTRIBUTE name(__VA_ARGS__)
#endif

namespace __trec {
void DestroyThreadState() {
  ThreadState *thr = cur_thread();
  Processor *proc = thr->proc();
  ThreadFinish(thr);
  ProcUnwire(proc, thr);
  ProcDestroy(proc);
  DTLS_Destroy();
  cur_thread_finalize();
}

void PlatformCleanUpThreadState(ThreadState *thr) {
  ThreadSignalContext *sctx = thr->signal_ctx;
  if (sctx) {
    thr->signal_ctx = 0;
    UnmapOrDie(sctx, sizeof(*sctx));
  }
}
}  // namespace __trec

#if !SANITIZER_MAC && !SANITIZER_NETBSD && !SANITIZER_FREEBSD
static void thread_finalize(void *v) {
  uptr iter = (uptr)v;
  if (iter > 1) {
    if (pthread_setspecific(interceptor_ctx()->finalize_key,
                            (void *)(iter - 1))) {
      Printf("TraceRecorder: failed to set thread key\n");
      Die();
    }
    return;
  }
  DestroyThreadState();
}
#endif

struct ThreadParam {
  void *(*callback)(void *arg);
  void *param;
  atomic_uintptr_t tid;
  __trec_metadata::FuncEnterMeta entry_meta;
  Vector<__trec_metadata::FuncParamMeta> parameters;
  bool isEnterMetaVaild = false;
  char debug_info[512];
  int debug_info_size = 0;
};

extern "C" void *__trec_thread_start_func(void *arg) {
  ThreadParam *p = (ThreadParam *)arg;
  void *(*callback)(void *arg) = p->callback;
  void *param = p->param;
  int tid = 0;
  {
    cur_thread_init();
    ThreadState *thr = cur_thread();
    // Thread-local state is not initialized yet.
    ScopedIgnoreInterceptors ignore;
#if !SANITIZER_MAC && !SANITIZER_NETBSD && !SANITIZER_FREEBSD
    if (pthread_setspecific(interceptor_ctx()->finalize_key,
                            (void *)GetPthreadDestructorIterations())) {
      Printf("TraceRecorder: failed to set thread key\n");
      Die();
    }
#endif
    while ((tid = atomic_load(&p->tid, memory_order_acquire)) == 0)
      internal_sched_yield();
    Processor *proc = ProcCreate();
    ProcWire(proc, thr);
    ThreadStart(thr, tid, GetTid(), ThreadType::Regular);
    thr->tctx->isFuncEnterMetaVaild = p->isEnterMetaVaild;
    thr->tctx->entry_meta = p->entry_meta;
    thr->tctx->parammetas.Resize(0);
    for (int idx = 0; idx < p->parameters.Size(); idx++) {
      thr->tctx->parammetas.PushBack(p->parameters[idx]);
    }
    internal_memcpy(thr->tctx->dbg_temp_buffer, p->debug_info,
                    p->debug_info_size);
    thr->tctx->dbg_temp_buffer_size = p->debug_info_size;
    atomic_store(&p->tid, 0, memory_order_release);
  }
  void *res = callback(param);
  // Prevent the callback from being tail called,
  // it mixes up stack traces.
  volatile int foo = 42;
  foo++;
  return res;
}

TREC_INTERCEPTOR(int, pthread_create, void *th, void *attr,
                 void *(*callback)(void *), void *param) {
  ScopedIgnoreInterceptors ignore;
  SCOPED_INTERCEPTOR_RAW(pthread_create, th, attr, callback, param);

  ThreadParam p;
  p.callback = callback;
  p.param = param;
  p.debug_info_size = 0;
  p.isEnterMetaVaild = false;
  if (thr->tctx->isFuncEnterMetaVaild) {
    p.entry_meta = thr->tctx->entry_meta;
    for (int idx = 0; idx < thr->tctx->parammetas.Size(); idx++) {
      if (thr->tctx->parammetas[idx].id == 4) {
        __trec_metadata::FuncParamMeta meta = thr->tctx->parammetas[idx];
        meta.id = 1;
        p.parameters.PushBack(meta);
      }
    }
    p.entry_meta.parammeta_cnt = p.parameters.Size();
    p.entry_meta.arg_size -= 3;
    p.isEnterMetaVaild = true;
    thr->tctx->parammetas.Resize(0);
    thr->tctx->isFuncEnterMetaVaild = false;
  }
  if (thr->tctx->dbg_temp_buffer_size) {
    p.debug_info_size =
        min(thr->tctx->dbg_temp_buffer_size, (__sanitizer::u64)512);
    internal_memcpy(p.debug_info, thr->tctx->dbg_temp_buffer,
                    p.debug_info_size);
    thr->tctx->dbg_temp_buffer_size = 0;
  }
  atomic_store(&p.tid, 0, memory_order_relaxed);

  if (ctx->after_multithreaded_fork) {
    if (flags()->die_after_fork) {
      Report(
          "TraceRecorder: starting new threads after multi-threaded "
          "fork is not supported. Dying (set die_after_fork=0 to override)\n");
      Die();
    } else {
      VPrintf(1,
              "TraceRecorder: starting new threads after multi-threaded "
              "fork is not supported (pid %d). Continuing because of "
              "die_after_fork=0, but you are on your own\n",
              internal_getpid());
    }
  }
  __sanitizer_pthread_attr_t myattr;
  if (attr == 0) {
    pthread_attr_init(&myattr);
    attr = &myattr;
  }
  int detached = 0;
  pthread_attr_getdetachstate(attr, &detached);
  AdjustStackSize(attr);

  int res = -1;
  {
    // Otherwise we see false positives in pthread stack manipulation.
    res = REAL(pthread_create)(th, attr, __trec_thread_start_func, &p);
  }
  if (res == 0) {
    int backup = thr->ignore_interceptors;
    thr->ignore_interceptors = 0;
    int tid =
        ThreadCreate(thr, caller_pc, *(uptr *)th, IsStateDetached(detached));
    thr->ignore_interceptors = backup;
    CHECK_NE(tid, 0);
    // Synchronization on p.tid serves two purposes:
    // 1. ThreadCreate must finish before the new thread starts.
    //    Otherwise the new thread can call pthread_detach, but the pthread_t
    //    identifier is not yet registered in ThreadRegistry by ThreadCreate.
    // 2. ThreadStart must finish before this thread continues.
    //    Otherwise, this thread can call pthread_detach and reset thr->sync
    //    before the new thread got a chance to acquire from it in ThreadStart.
    atomic_store(&p.tid, tid, memory_order_release);
    while (atomic_load(&p.tid, memory_order_acquire) != 0)
      internal_sched_yield();
  }
  if (attr == &myattr)
    pthread_attr_destroy(&myattr);
  return res;
}

TREC_INTERCEPTOR(int, pthread_join, void *th, void **ret) {
  SCOPED_INTERCEPTOR_RAW(pthread_join, th, ret);
  int tid = ThreadConsumeTid(thr, caller_pc, (uptr)th);
  int res = BLOCK_REAL(pthread_join)(th, ret);
  if (res == 0) {
    ThreadJoin(thr, caller_pc, tid);
  }
  return res;
}

DEFINE_REAL_PTHREAD_FUNCTIONS

TREC_INTERCEPTOR(int, pthread_detach, void *th) {
  SCOPED_INTERCEPTOR_RAW(pthread_detach, th);
  int tid = ThreadConsumeTid(thr, caller_pc, (uptr)th);
  int res = REAL(pthread_detach)(th);
  if (res == 0) {
    ThreadDetach(thr, caller_pc, tid);
  }
  return res;
}

TREC_INTERCEPTOR(void, pthread_exit, void *retval) {
  {
    SCOPED_INTERCEPTOR_RAW(pthread_exit, retval);
#if !SANITIZER_MAC && !SANITIZER_ANDROID
    CHECK_EQ(thr, &cur_thread_placeholder);
#endif
  }
  DestroyThreadState();
  REAL(pthread_exit)(retval);
}

#if SANITIZER_LINUX
TREC_INTERCEPTOR(int, pthread_tryjoin_np, void *th, void **ret) {
  SCOPED_INTERCEPTOR_RAW(pthread_tryjoin_np, th, ret);
  int tid = ThreadConsumeTid(thr, caller_pc, (uptr)th);
  int res = REAL(pthread_tryjoin_np)(th, ret);
  if (res == 0)
    ThreadJoin(thr, caller_pc, tid);
  else
    ThreadNotJoined(thr, caller_pc, tid, (uptr)th);
  return res;
}

TREC_INTERCEPTOR(int, pthread_timedjoin_np, void *th, void **ret,
                 const struct timespec *abstime) {
  SCOPED_INTERCEPTOR_RAW(pthread_timedjoin_np, th, ret, abstime);
  int tid = ThreadConsumeTid(thr, caller_pc, (uptr)th);
  int res = BLOCK_REAL(pthread_timedjoin_np)(th, ret, abstime);
  if (res == 0)
    ThreadJoin(thr, caller_pc, tid);
  else
    ThreadNotJoined(thr, caller_pc, tid, (uptr)th);
  return res;
}
#endif

// Problem:
// NPTL implementation of pthread_cond has 2 versions (2.2.5 and 2.3.2).
// pthread_cond_t has different size in the different versions.
// If call new REAL functions for old pthread_cond_t, they will corrupt memory
// after pthread_cond_t (old cond is smaller).
// If we call old REAL functions for new pthread_cond_t, we will lose  some
// functionality (e.g. old functions do not support waiting against
// CLOCK_REALTIME).
// Proper handling would require to have 2 versions of interceptors as well.
// But this is messy, in particular requires linker scripts when sanitizer
// runtime is linked into a shared library.
// Instead we assume we don't have dynamic libraries built against old
// pthread (2.2.5 is dated by 2002). And provide legacy_pthread_cond flag
// that allows to work with old libraries (but this mode does not support
// some features, e.g. pthread_condattr_getpshared).
static void *init_cond(void *c, bool force = false) {
  // sizeof(pthread_cond_t) >= sizeof(uptr) in both versions.
  // So we allocate additional memory on the side large enough to hold
  // any pthread_cond_t object. Always call new REAL functions, but pass
  // the aux object to them.
  // Note: the code assumes that PTHREAD_COND_INITIALIZER initializes
  // first word of pthread_cond_t to zero.
  // It's all relevant only for linux.
  if (!common_flags()->legacy_pthread_cond)
    return c;
  atomic_uintptr_t *p = (atomic_uintptr_t *)c;
  uptr cond = atomic_load(p, memory_order_acquire);
  if (!force && cond != 0)
    return (void *)cond;
  void *newcond = WRAP(malloc)(pthread_cond_t_sz);
  internal_memset(newcond, 0, pthread_cond_t_sz);
  if (atomic_compare_exchange_strong(p, &cond, (uptr)newcond,
                                     memory_order_acq_rel))
    return newcond;
  WRAP(free)(newcond);
  return (void *)cond;
}

namespace {

template <class Fn>
struct CondMutexUnlockCtx {
  ScopedInterceptor *si;
  ThreadState *thr;
  uptr pc;
  void *m;
  void *c;
  const Fn &fn;

  int Cancel() const { return fn(); }
  void Unlock() const;
};

template <class Fn>
void CondMutexUnlockCtx<Fn>::Unlock() const {
  // pthread_cond_wait interceptor has enabled async signal delivery
  // (see BlockingCall below). Disable async signals since we are running
  // trec code. Also ScopedInterceptor and BlockingCall destructors won't run
  // since the thread is cancelled, so we have to manually execute them
  // (the thread still can run some user code due to pthread_cleanup_push).
  ThreadSignalContext *ctx = SigCtx(thr);
  CHECK_EQ(atomic_load(&ctx->in_blocking_func, memory_order_relaxed), 1);
  atomic_store(&ctx->in_blocking_func, 0, memory_order_relaxed);
  // Undo BlockingCall ctor effects.
  thr->ignore_interceptors--;
  si->~ScopedInterceptor();
}
}  // namespace


#if !SANITIZER_MAC
TREC_INTERCEPTOR(int, pthread_barrier_init, void *b, void *a, unsigned count) {
  SCOPED_TREC_INTERCEPTOR(pthread_barrier_init, b, a, count);
  int res = REAL(pthread_barrier_init)(b, a, count);
  return res;
}

TREC_INTERCEPTOR(int, pthread_barrier_destroy, void *b) {
  SCOPED_TREC_INTERCEPTOR(pthread_barrier_destroy, b);
  int res = REAL(pthread_barrier_destroy)(b);
  return res;
}

TREC_INTERCEPTOR(int, pthread_barrier_wait, void *b) {
  SCOPED_TREC_INTERCEPTOR(pthread_barrier_wait, b);
  int res = REAL(pthread_barrier_wait)(b);
  return res;
}
#endif

TREC_INTERCEPTOR(int, pthread_once, void *o, void (*f)()) {
  SCOPED_INTERCEPTOR_RAW(pthread_once, o, f);
  if (o == 0 || f == 0)
    return errno_EINVAL;
  atomic_uint32_t *a;

  if (SANITIZER_NETBSD)
    a = static_cast<atomic_uint32_t *>(
        (void *)((char *)o + __sanitizer::pthread_mutex_t_sz));
  else
    a = static_cast<atomic_uint32_t *>(o);

  u32 v = atomic_load(a, memory_order_acquire);
  if (v == 0 &&
      atomic_compare_exchange_strong(a, &v, 1, memory_order_relaxed)) {
    (*f)();
    atomic_store(a, 2, memory_order_release);
  } else {
    while (v != 2) {
      internal_sched_yield();
      v = atomic_load(a, memory_order_acquire);
    }
  }
  return 0;
}

static void FlushStreams() {
  // Flushing all the streams here may freeze the process if a child thread is
  // performing file stream operations at the same time.
  fflush(stdout);
  fflush(stderr);
}

// The following functions are intercepted merely to process pending signals.
// If program blocks signal X, we must deliver the signal before the function
// returns. Similarly, if program unblocks a signal (or returns from sigsuspend)
// it's better to deliver the signal straight away.

TREC_INTERCEPTOR(int, pthread_sigmask, int how, const __sanitizer_sigset_t *set,
                 __sanitizer_sigset_t *oldset) {
  SCOPED_TREC_INTERCEPTOR(pthread_sigmask, how, set, oldset);
  return REAL(pthread_sigmask)(how, set, oldset);
}

namespace __trec {

static void CallUserSignalHandler(ThreadState *thr, bool sync, bool acquire,
                                  bool sigact, int sig,
                                  __sanitizer_siginfo *info, void *uctx) {
  __sanitizer_sigaction *sigactions = interceptor_ctx()->sigactions;
  // Ensure that the handler does not spoil errno.
  const int saved_errno = errno;
  errno = 99;
  // This code races with sigaction. Be careful to not read sa_sigaction twice.
  // Also need to remember pc for reporting before the call,
  // because the handler can reset it.
  volatile uptr pc =
      sigact ? (uptr)sigactions[sig].sigaction : (uptr)sigactions[sig].handler;
  if (pc != sig_dfl && pc != sig_ign) {
    if (sigact)
      ((__sanitizer_sigactionhandler_ptr)pc)(sig, info, uctx);
    else
      ((__sanitizer_sighandler_ptr)pc)(sig);
  }
  errno = saved_errno;
}

void ProcessPendingSignals(ThreadState *thr) {
  ThreadSignalContext *sctx = SigCtx(thr);
  if (sctx == 0 ||
      atomic_load(&sctx->have_pending_signals, memory_order_relaxed) == 0)
    return;
  atomic_store(&sctx->have_pending_signals, 0, memory_order_relaxed);
  atomic_fetch_add(&thr->in_signal_handler, 1, memory_order_relaxed);
  internal_sigfillset(&sctx->emptyset);
  int res = REAL(pthread_sigmask)(SIG_SETMASK, &sctx->emptyset, &sctx->oldset);
  CHECK_EQ(res, 0);
  for (int sig = 0; sig < kSigCount; sig++) {
    SignalDesc *signal = &sctx->pending_signals[sig];
    if (signal->armed) {
      signal->armed = false;
      CallUserSignalHandler(thr, false, true, signal->sigaction, sig,
                            &signal->siginfo, &signal->ctx);
    }
  }
  res = REAL(pthread_sigmask)(SIG_SETMASK, &sctx->oldset, 0);
  CHECK_EQ(res, 0);
  atomic_fetch_add(&thr->in_signal_handler, -1, memory_order_relaxed);
}

}  // namespace __trec


TREC_INTERCEPTOR(int, pthread_kill, void *tid, int sig) {
  SCOPED_TREC_INTERCEPTOR(pthread_kill, tid, sig);
  ThreadSignalContext *sctx = SigCtx(thr);
  CHECK_NE(sctx, 0);
  int prev = sctx->int_signal_send;
  if (tid == pthread_self()) {
    sctx->int_signal_send = sig;
  }
  int res = REAL(pthread_kill)(tid, sig);
  if (tid == pthread_self()) {
    CHECK_EQ(sctx->int_signal_send, sig);
    sctx->int_signal_send = prev;
  }
  return res;
}

TREC_INTERCEPTOR(int, fork, int fake) {
  SCOPED_INTERCEPTOR_RAW(fork, fake);
  ForkBefore(thr, pc);
  int pid;
  {
    // On OS X, REAL(fork) can call intercepted functions (OSSpinLockLock), and
    // we'll assert in CheckNoLocks() unless we ignore interceptors.
    ScopedIgnoreInterceptors ignore;
    pid = REAL(fork)(fake);
  }

  if (pid == 0) {
    // child
    ForkChildAfter(thr, pc);
  } else if (pid > 0) {
    // parent
    ForkParentAfter(thr, pc);
  } else {
    // error
    ForkParentAfter(thr, pc);
  }
  return pid;
}

TREC_INTERCEPTOR(int, vfork, int fake) {
  // Some programs (e.g. openjdk) call close for all file descriptors
  // in the child process. Under trec it leads to false positives, because
  // address space is shared, so the parent process also thinks that
  // the descriptors are closed (while they are actually not).
  // This leads to false positives due to missed synchronization.
  // Strictly saying this is undefined behavior, because vfork child is not
  // allowed to call any functions other than exec/exit. But this is what
  // openjdk does, so we want to handle it.
  // We could disable interceptors in the child process. But it's not possible
  // to simply intercept and wrap vfork, because vfork child is not allowed
  // to return from the function that calls vfork, and that's exactly what
  // we would do. So this would require some assembly trickery as well.
  // Instead we simply turn vfork into fork.
  return WRAP(fork)(fake);
}

static int OnExit(ThreadState *thr) {
  int status = Finalize(thr);
  FlushStreams();
  return status;
}

struct TrecInterceptorContext {
  ThreadState *thr;
  const uptr caller_pc;
  const uptr pc;
};

#if !SANITIZER_MAC
static void HandleRecvmsg(ThreadState *thr, uptr pc, __sanitizer_msghdr *msg) {
  int fds[64];
  int cnt = ExtractRecvmsgFDs(msg, fds, ARRAY_SIZE(fds));
}
#endif

#include "sanitizer_common/sanitizer_platform_interceptors.h"
// Causes interceptor recursion (getaddrinfo() and fopen())
#undef SANITIZER_INTERCEPT_GETADDRINFO
// We define our own.
#if SANITIZER_INTERCEPT_TLS_GET_ADDR
#define NEED_TLS_GET_ADDR
#endif
#undef SANITIZER_INTERCEPT_TLS_GET_ADDR
#undef SANITIZER_INTERCEPT_PTHREAD_SIGMASK

#define COMMON_INTERCEPT_FUNCTION(name) INTERCEPT_FUNCTION(name)
#define COMMON_INTERCEPT_FUNCTION_VER(name, ver) \
  INTERCEPT_FUNCTION_VER(name, ver)
#define COMMON_INTERCEPT_FUNCTION_VER_UNVERSIONED_FALLBACK(name, ver) \
  (INTERCEPT_FUNCTION_VER(name, ver) || INTERCEPT_FUNCTION(name))

#define COMMON_INTERCEPTOR_ENTER(ctx, func, ...)      \
  SCOPED_TREC_INTERCEPTOR(func, __VA_ARGS__);         \
  TrecInterceptorContext _ctx = {thr, caller_pc, pc}; \
  ctx = (void *)&_ctx;                                \
  (void)ctx;

#define COMMON_INTERCEPTOR_ENTER_NOIGNORE(ctx, func, ...) \
  SCOPED_INTERCEPTOR_RAW(func, __VA_ARGS__);              \
  TrecInterceptorContext _ctx = {thr, caller_pc, pc};     \
  ctx = (void *)&_ctx;                                    \
  (void)ctx;

void Acquire(ThreadState *thr, uptr pc, uptr addr) {}
void FdAcquire(ThreadState *thr, uptr pc, int fd) {}
void FdRelease(ThreadState *thr, uptr pc, int fd) {}
void FdSocketAccept(ThreadState *thr, uptr pc, int fd, int newfd) {}
void FdAccess(ThreadState *thr, uptr pc, int fd) {}
uptr Dir2addr(const char *path) {
  (void)path;
  static u64 addr;
  return (uptr)&addr;
}

#define COMMON_INTERCEPTOR_FD_ACQUIRE(ctx, fd) \
  FdAcquire(((TrecInterceptorContext *)ctx)->thr, pc, fd)

#define COMMON_INTERCEPTOR_FD_RELEASE(ctx, fd) \
  FdRelease(((TrecInterceptorContext *)ctx)->thr, pc, fd)

#define COMMON_INTERCEPTOR_FD_ACCESS(ctx, fd) \
  FdAccess(((TrecInterceptorContext *)ctx)->thr, pc, fd)

#define COMMON_INTERCEPTOR_FD_SOCKET_ACCEPT(ctx, fd, newfd) \
  FdSocketAccept(((TrecInterceptorContext *)ctx)->thr, pc, fd, newfd)

#define COMMON_INTERCEPTOR_SET_THREAD_NAME(ctx, name) \
  ThreadSetName(((TrecInterceptorContext *)ctx)->thr, name)
#define COMMON_INTERCEPTOR_DIR_ACQUIRE(ctx, path) \
  Acquire(((TrecInterceptorContext *)ctx)->thr, pc, Dir2addr(path))

#define COMMON_INTERCEPTOR_SET_PTHREAD_NAME(ctx, thread, name) \
  __trec::ctx->thread_registry->SetThreadNameByUserId(thread, name)

#define COMMON_INTERCEPTOR_BLOCK_REAL(name) BLOCK_REAL(name)

#define COMMON_INTERCEPTOR_ON_EXIT(ctx) \
  OnExit(((TrecInterceptorContext *)ctx)->thr)

#define COMMON_INTERCEPTOR_MUTEX_PRE_LOCK(ctx, m)    \
  MutexPreLock(((TrecInterceptorContext *)ctx)->thr, \
               ((TrecInterceptorContext *)ctx)->caller_pc, (uptr)m)

#define COMMON_INTERCEPTOR_MUTEX_POST_LOCK(ctx, m)                   \
  MutexPostLock(((TrecInterceptorContext *)ctx)->thr,                \
                ((TrecInterceptorContext *)ctx)->caller_pc, (uptr)m, \
                {0x8000, 1})

#define COMMON_INTERCEPTOR_MUTEX_UNLOCK(ctx, m)                    \
  MutexUnlock(((TrecInterceptorContext *)ctx)->thr,                \
              ((TrecInterceptorContext *)ctx)->caller_pc, (uptr)m, \
              {0x8000, 1})

#define COMMON_INTERCEPTOR_MUTEX_REPAIR(ctx, m)     \
  MutexRepair(((TrecInterceptorContext *)ctx)->thr, \
              ((TrecInterceptorContext *)ctx)->pc, (uptr)m)

#define COMMON_INTERCEPTOR_MUTEX_INVALID(ctx, m)           \
  MutexInvalidAccess(((TrecInterceptorContext *)ctx)->thr, \
                     ((TrecInterceptorContext *)ctx)->pc, (uptr)m)

#if !SANITIZER_MAC
#define COMMON_INTERCEPTOR_HANDLE_RECVMSG(ctx, msg)   \
  HandleRecvmsg(((TrecInterceptorContext *)ctx)->thr, \
                ((TrecInterceptorContext *)ctx)->pc, msg)
#endif

#define COMMON_INTERCEPTOR_GET_TLS_RANGE(begin, end) \
  if (TrecThread *t = GetCurrentThread()) {          \
    *begin = t->tls_begin();                         \
    *end = t->tls_end();                             \
  } else {                                           \
    *begin = *end = 0;                               \
  }

#include "sanitizer_common/sanitizer_signal_interceptors.inc"
static int sigaction_impl(int sig, const __sanitizer_sigaction *act,
                          __sanitizer_sigaction *old);
static __sanitizer_sighandler_ptr signal_impl(int sig,
                                              __sanitizer_sighandler_ptr h);

#define SIGNAL_INTERCEPTOR_SIGACTION_IMPL(signo, act, oldact) \
  { return sigaction_impl(signo, act, oldact); }

#define SIGNAL_INTERCEPTOR_SIGNAL_IMPL(func, signo, handler) \
  { return (uptr)signal_impl(signo, (__sanitizer_sighandler_ptr)handler); }

#define TREC_SYSCALL()             \
  ThreadState *thr = cur_thread(); \
  if (thr->ignore_interceptors)    \
    return;                        \
  ScopedSyscall scoped_syscall(thr) /**/
struct ScopedSyscall {
  ThreadState *thr;

  explicit ScopedSyscall(ThreadState *thr) : thr(thr) { Initialize(thr); }

  ~ScopedSyscall() { ProcessPendingSignals(thr); }
};

#if !SANITIZER_FREEBSD && !SANITIZER_MAC
static void syscall_pre_fork(uptr pc) {
  TREC_SYSCALL();
  ForkBefore(thr, pc);
}

static void syscall_post_fork(uptr pc, int pid) {
  TREC_SYSCALL();
  if (pid == 0) {
    // child
    ForkChildAfter(thr, pc);
  } else if (pid > 0) {
    // parent
    ForkParentAfter(thr, pc);
  } else {
    // error
    ForkParentAfter(thr, pc);
  }
}
#endif

#define COMMON_SYSCALL_PRE_READ_RANGE(p, s, ...)                              \
  syscall_access_range(StackTrace::GetPreviousInstructionPc(GET_CALLER_PC()), \
                       (uptr)(p), (uptr)(s), false, ##__VA_ARGS__)

#define COMMON_SYSCALL_PRE_WRITE_RANGE(p, s, ...)                             \
  syscall_access_range(StackTrace::GetPreviousInstructionPc(GET_CALLER_PC()), \
                       (uptr)(p), (uptr)(s), true, ##__VA_ARGS__)

#define COMMON_SYSCALL_POST_READ_RANGE(p, s) \
  do {                                       \
    (void)(p);                               \
    (void)(s);                               \
  } while (false)

#define COMMON_SYSCALL_POST_WRITE_RANGE(p, s) \
  do {                                        \
    (void)(p);                                \
    (void)(s);                                \
  } while (false)

#define COMMON_SYSCALL_ACQUIRE(addr)                                     \
  syscall_acquire(StackTrace::GetPreviousInstructionPc(GET_CALLER_PC()), \
                  (uptr)(addr))

#define COMMON_SYSCALL_RELEASE(addr)                                     \
  syscall_release(StackTrace::GetPreviousInstructionPc(GET_CALLER_PC()), \
                  (uptr)(addr))

#define COMMON_SYSCALL_PRE_FORK() \
  syscall_pre_fork(StackTrace::GetPreviousInstructionPc(GET_CALLER_PC()))

#define COMMON_SYSCALL_POST_FORK(res) \
  syscall_post_fork(StackTrace::GetPreviousInstructionPc(GET_CALLER_PC()), res)

namespace __trec {

static void finalize(void *arg) {
  ThreadState *thr = cur_thread();
  int status = Finalize(thr);
  // Make sure the output is not lost.
  FlushStreams();
  if (status)
    Die();
}

#if !SANITIZER_MAC && !SANITIZER_ANDROID
static void unreachable() {
  Report("FATAL: TraceRecorder: unreachable called\n");
  Die();
}
#endif

// Define default implementation since interception of libdispatch  is optional.
SANITIZER_WEAK_ATTRIBUTE void InitializeLibdispatchInterceptors() {}

void InitializeInterceptors() {
  // Instruct libc malloc to consume less memory.
#if SANITIZER_GLIBC
  mallopt(1, 0);           // M_MXFAST
  mallopt(-3, 32 * 1024);  // M_MMAP_THRESHOLD
#endif

  new (interceptor_ctx()) InterceptorContext();
  //InitializeCommonInterceptors();
  InitializeSignalInterceptors();
  InitializeLibdispatchInterceptors();


  TREC_INTERCEPT(pthread_create);
  TREC_INTERCEPT(pthread_join);
  TREC_INTERCEPT(pthread_detach);
  TREC_INTERCEPT(pthread_exit);
#if SANITIZER_LINUX
  TREC_INTERCEPT(pthread_tryjoin_np);
  TREC_INTERCEPT(pthread_timedjoin_np);
#endif

  TREC_INTERCEPT(pthread_barrier_init);
  TREC_INTERCEPT(pthread_barrier_destroy);
  TREC_INTERCEPT(pthread_barrier_wait);

  TREC_INTERCEPT(pthread_once);

  TREC_INTERCEPT(pthread_sigmask);
  TREC_INTERCEPT(pthread_kill);
  
  TREC_INTERCEPT(fork);
  TREC_INTERCEPT(vfork);
  TREC_MAYBE_INTERCEPT_ON_EXIT;
  TREC_INTERCEPT(__cxa_atexit);
  //TREC_INTERCEPT(_exit);


#if !SANITIZER_MAC && !SANITIZER_ANDROID
  // Need to setup it, because interceptors check that the function is resolved.
  // But atexit is emitted directly into the module, so can't be resolved.
  REAL(atexit) = (int (*)(void (*)()))unreachable;
#endif

  if (REAL(__cxa_atexit)(&finalize, 0, 0)) {
    Printf("TraceRecorder: failed to setup atexit callback\n");
    Die();
  }
#if !SANITIZER_MAC && !SANITIZER_NETBSD && !SANITIZER_FREEBSD
  if (pthread_key_create(&interceptor_ctx()->finalize_key, &thread_finalize)) {
    Printf("TraceRecorder: failed to create thread key\n");
    Die();
  }
#endif

}

}  // namespace __trec

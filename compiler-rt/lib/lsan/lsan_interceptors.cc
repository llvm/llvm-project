//=-- lsan_interceptors.cc ------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of LeakSanitizer.
// Interceptors for standalone LSan.
//
//===----------------------------------------------------------------------===//

#include "interception/interception.h"
#include "sanitizer_common/sanitizer_allocator.h"
#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_linux.h"
#include "sanitizer_common/sanitizer_platform_interceptors.h"
#include "sanitizer_common/sanitizer_platform_limits_posix.h"
#include "sanitizer_common/sanitizer_posix.h"
#include "sanitizer_common/sanitizer_tls_get_addr.h"
#include "lsan.h"
#include "lsan_allocator.h"
#include "lsan_common.h"
#include "lsan_thread.h"

#include <stddef.h>

using namespace __lsan;

extern "C" {
int pthread_attr_init(void *attr);
int pthread_attr_destroy(void *attr);
int pthread_attr_getdetachstate(void *attr, int *v);
int pthread_key_create(unsigned *key, void (*destructor)(void* v));
int pthread_setspecific(unsigned key, const void *v);
}

///// Malloc/free interceptors. /////

namespace std {
  struct nothrow_t;
}

#if !SANITIZER_MAC
INTERCEPTOR(void*, malloc, uptr size) {
  ENSURE_LSAN_INITED;
  GET_STACK_TRACE_MALLOC;
  return lsan_malloc(size, stack);
}

INTERCEPTOR(void, free, void *p) {
  ENSURE_LSAN_INITED;
  lsan_free(p);
}

INTERCEPTOR(void*, calloc, uptr nmemb, uptr size) {
  if (lsan_init_is_running) {
    // Hack: dlsym calls calloc before REAL(calloc) is retrieved from dlsym.
    const uptr kCallocPoolSize = 1024;
    static uptr calloc_memory_for_dlsym[kCallocPoolSize];
    static uptr allocated;
    uptr size_in_words = ((nmemb * size) + kWordSize - 1) / kWordSize;
    void *mem = (void*)&calloc_memory_for_dlsym[allocated];
    allocated += size_in_words;
    CHECK(allocated < kCallocPoolSize);
    return mem;
  }
  if (CallocShouldReturnNullDueToOverflow(size, nmemb)) return nullptr;
  ENSURE_LSAN_INITED;
  GET_STACK_TRACE_MALLOC;
  return lsan_calloc(nmemb, size, stack);
}

INTERCEPTOR(void*, realloc, void *q, uptr size) {
  ENSURE_LSAN_INITED;
  GET_STACK_TRACE_MALLOC;
  return lsan_realloc(q, size, stack);
}

INTERCEPTOR(int, posix_memalign, void **memptr, uptr alignment, uptr size) {
  ENSURE_LSAN_INITED;
  GET_STACK_TRACE_MALLOC;
  *memptr = lsan_memalign(alignment, size, stack);
  // FIXME: Return ENOMEM if user requested more than max alloc size.
  return 0;
}

INTERCEPTOR(void*, valloc, uptr size) {
  ENSURE_LSAN_INITED;
  GET_STACK_TRACE_MALLOC;
  return lsan_valloc(size, stack);
}
#endif

#if SANITIZER_INTERCEPT_MEMALIGN
INTERCEPTOR(void*, memalign, uptr alignment, uptr size) {
  ENSURE_LSAN_INITED;
  GET_STACK_TRACE_MALLOC;
  return lsan_memalign(alignment, size, stack);
}
#define LSAN_MAYBE_INTERCEPT_MEMALIGN INTERCEPT_FUNCTION(memalign)

INTERCEPTOR(void *, __libc_memalign, uptr alignment, uptr size) {
  ENSURE_LSAN_INITED;
  GET_STACK_TRACE_MALLOC;
  void *res = lsan_memalign(alignment, size, stack);
  DTLS_on_libc_memalign(res, size);
  return res;
}
#define LSAN_MAYBE_INTERCEPT___LIBC_MEMALIGN INTERCEPT_FUNCTION(__libc_memalign)
#else
#define LSAN_MAYBE_INTERCEPT_MEMALIGN
#define LSAN_MAYBE_INTERCEPT___LIBC_MEMALIGN
#endif // SANITIZER_INTERCEPT_MEMALIGN

#if SANITIZER_INTERCEPT_ALIGNED_ALLOC
INTERCEPTOR(void*, aligned_alloc, uptr alignment, uptr size) {
  ENSURE_LSAN_INITED;
  GET_STACK_TRACE_MALLOC;
  return lsan_memalign(alignment, size, stack);
}
#define LSAN_MAYBE_INTERCEPT_ALIGNED_ALLOC INTERCEPT_FUNCTION(aligned_alloc)
#else
#define LSAN_MAYBE_INTERCEPT_ALIGNED_ALLOC
#endif

#if SANITIZER_INTERCEPT_MALLOC_USABLE_SIZE
INTERCEPTOR(uptr, malloc_usable_size, void *ptr) {
  ENSURE_LSAN_INITED;
  return GetMallocUsableSize(ptr);
}
#define LSAN_MAYBE_INTERCEPT_MALLOC_USABLE_SIZE \
        INTERCEPT_FUNCTION(malloc_usable_size)
#else
#define LSAN_MAYBE_INTERCEPT_MALLOC_USABLE_SIZE
#endif

#if SANITIZER_INTERCEPT_MALLOPT_AND_MALLINFO
struct fake_mallinfo {
  int x[10];
};

INTERCEPTOR(struct fake_mallinfo, mallinfo, void) {
  struct fake_mallinfo res;
  internal_memset(&res, 0, sizeof(res));
  return res;
}
#define LSAN_MAYBE_INTERCEPT_MALLINFO INTERCEPT_FUNCTION(mallinfo)

INTERCEPTOR(int, mallopt, int cmd, int value) {
  return -1;
}
#define LSAN_MAYBE_INTERCEPT_MALLOPT INTERCEPT_FUNCTION(mallopt)
#else
#define LSAN_MAYBE_INTERCEPT_MALLINFO
#define LSAN_MAYBE_INTERCEPT_MALLOPT
#endif // SANITIZER_INTERCEPT_MALLOPT_AND_MALLINFO

#if SANITIZER_INTERCEPT_PVALLOC
INTERCEPTOR(void*, pvalloc, uptr size) {
  ENSURE_LSAN_INITED;
  GET_STACK_TRACE_MALLOC;
  uptr PageSize = GetPageSizeCached();
  size = RoundUpTo(size, PageSize);
  if (size == 0) {
    // pvalloc(0) should allocate one page.
    size = PageSize;
  }
  return Allocate(stack, size, GetPageSizeCached(), kAlwaysClearMemory);
}
#define LSAN_MAYBE_INTERCEPT_PVALLOC INTERCEPT_FUNCTION(pvalloc)
#else
#define LSAN_MAYBE_INTERCEPT_PVALLOC
#endif // SANITIZER_INTERCEPT_PVALLOC

#if SANITIZER_INTERCEPT_CFREE
INTERCEPTOR(void, cfree, void *p) ALIAS(WRAPPER_NAME(free));
#define LSAN_MAYBE_INTERCEPT_CFREE INTERCEPT_FUNCTION(cfree)
#else
#define LSAN_MAYBE_INTERCEPT_CFREE
#endif // SANITIZER_INTERCEPT_CFREE

#if SANITIZER_INTERCEPT_MCHECK_MPROBE
INTERCEPTOR(int, mcheck, void (*abortfunc)(int mstatus)) {
  return 0;
}

INTERCEPTOR(int, mcheck_pedantic, void (*abortfunc)(int mstatus)) {
  return 0;
}

INTERCEPTOR(int, mprobe, void *ptr) {
  return 0;
}
#endif // SANITIZER_INTERCEPT_MCHECK_MPROBE

#define OPERATOR_NEW_BODY                              \
  ENSURE_LSAN_INITED;                                  \
  GET_STACK_TRACE_MALLOC;                              \
  return Allocate(stack, size, 1, kAlwaysClearMemory);

INTERCEPTOR_ATTRIBUTE
void *operator new(size_t size) { OPERATOR_NEW_BODY; }
INTERCEPTOR_ATTRIBUTE
void *operator new[](size_t size) { OPERATOR_NEW_BODY; }
INTERCEPTOR_ATTRIBUTE
void *operator new(size_t size, std::nothrow_t const&) { OPERATOR_NEW_BODY; }
INTERCEPTOR_ATTRIBUTE
void *operator new[](size_t size, std::nothrow_t const&) { OPERATOR_NEW_BODY; }

#define OPERATOR_DELETE_BODY \
  ENSURE_LSAN_INITED;        \
  Deallocate(ptr);

INTERCEPTOR_ATTRIBUTE
void operator delete(void *ptr) NOEXCEPT { OPERATOR_DELETE_BODY; }
INTERCEPTOR_ATTRIBUTE
void operator delete[](void *ptr) NOEXCEPT { OPERATOR_DELETE_BODY; }
INTERCEPTOR_ATTRIBUTE
void operator delete(void *ptr, std::nothrow_t const&) { OPERATOR_DELETE_BODY; }
INTERCEPTOR_ATTRIBUTE
void operator delete[](void *ptr, std::nothrow_t const &) {
  OPERATOR_DELETE_BODY;
}

///// Thread initialization and finalization. /////

static unsigned g_thread_finalize_key;

static void thread_finalize(void *v) {
  uptr iter = (uptr)v;
  if (iter > 1) {
    if (pthread_setspecific(g_thread_finalize_key, (void*)(iter - 1))) {
      Report("LeakSanitizer: failed to set thread key.\n");
      Die();
    }
    return;
  }
  ThreadFinish();
}

struct ThreadParam {
  void *(*callback)(void *arg);
  void *param;
  atomic_uintptr_t tid;
};

extern "C" void *__lsan_thread_start_func(void *arg) {
  ThreadParam *p = (ThreadParam*)arg;
  void* (*callback)(void *arg) = p->callback;
  void *param = p->param;
  // Wait until the last iteration to maximize the chance that we are the last
  // destructor to run.
  if (pthread_setspecific(g_thread_finalize_key,
                          (void*)GetPthreadDestructorIterations())) {
    Report("LeakSanitizer: failed to set thread key.\n");
    Die();
  }
  int tid = 0;
  while ((tid = atomic_load(&p->tid, memory_order_acquire)) == 0)
    internal_sched_yield();
  SetCurrentThread(tid);
  ThreadStart(tid, GetTid());
  atomic_store(&p->tid, 0, memory_order_release);
  return callback(param);
}

INTERCEPTOR(int, pthread_create, void *th, void *attr,
            void *(*callback)(void *), void *param) {
  ENSURE_LSAN_INITED;
  EnsureMainThreadIDIsCorrect();
  __sanitizer_pthread_attr_t myattr;
  if (!attr) {
    pthread_attr_init(&myattr);
    attr = &myattr;
  }
  AdjustStackSize(attr);
  int detached = 0;
  pthread_attr_getdetachstate(attr, &detached);
  ThreadParam p;
  p.callback = callback;
  p.param = param;
  atomic_store(&p.tid, 0, memory_order_relaxed);
  int res;
  {
    // Ignore all allocations made by pthread_create: thread stack/TLS may be
    // stored by pthread for future reuse even after thread destruction, and
    // the linked list it's stored in doesn't even hold valid pointers to the
    // objects, the latter are calculated by obscure pointer arithmetic.
    ScopedInterceptorDisabler disabler;
    res = REAL(pthread_create)(th, attr, __lsan_thread_start_func, &p);
  }
  if (res == 0) {
    int tid = ThreadCreate(GetCurrentThread(), *(uptr *)th,
                           IsStateDetached(detached));
    CHECK_NE(tid, 0);
    atomic_store(&p.tid, tid, memory_order_release);
    while (atomic_load(&p.tid, memory_order_acquire) != 0)
      internal_sched_yield();
  }
  if (attr == &myattr)
    pthread_attr_destroy(&myattr);
  return res;
}

INTERCEPTOR(int, pthread_join, void *th, void **ret) {
  ENSURE_LSAN_INITED;
  int tid = ThreadTid((uptr)th);
  int res = REAL(pthread_join)(th, ret);
  if (res == 0)
    ThreadJoin(tid);
  return res;
}

namespace __lsan {

void InitializeInterceptors() {
  INTERCEPT_FUNCTION(malloc);
  INTERCEPT_FUNCTION(free);
  LSAN_MAYBE_INTERCEPT_CFREE;
  INTERCEPT_FUNCTION(calloc);
  INTERCEPT_FUNCTION(realloc);
  LSAN_MAYBE_INTERCEPT_MEMALIGN;
  LSAN_MAYBE_INTERCEPT___LIBC_MEMALIGN;
  LSAN_MAYBE_INTERCEPT_ALIGNED_ALLOC;
  INTERCEPT_FUNCTION(posix_memalign);
  INTERCEPT_FUNCTION(valloc);
  LSAN_MAYBE_INTERCEPT_PVALLOC;
  LSAN_MAYBE_INTERCEPT_MALLOC_USABLE_SIZE;
  LSAN_MAYBE_INTERCEPT_MALLINFO;
  LSAN_MAYBE_INTERCEPT_MALLOPT;
  INTERCEPT_FUNCTION(pthread_create);
  INTERCEPT_FUNCTION(pthread_join);

  if (pthread_key_create(&g_thread_finalize_key, &thread_finalize)) {
    Report("LeakSanitizer: failed to create thread key.\n");
    Die();
  }
}

} // namespace __lsan

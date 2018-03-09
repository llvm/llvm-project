//===-- hwasan_interceptors.cc ----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of HWAddressSanitizer.
//
// Interceptors for standard library functions.
//
// FIXME: move as many interceptors as possible into
// sanitizer_common/sanitizer_common_interceptors.h
//===----------------------------------------------------------------------===//

#include "interception/interception.h"
#include "hwasan.h"
#include "hwasan_thread.h"
#include "hwasan_poisoning.h"
#include "sanitizer_common/sanitizer_platform_limits_posix.h"
#include "sanitizer_common/sanitizer_allocator.h"
#include "sanitizer_common/sanitizer_allocator_interface.h"
#include "sanitizer_common/sanitizer_allocator_internal.h"
#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_errno.h"
#include "sanitizer_common/sanitizer_stackdepot.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_linux.h"
#include "sanitizer_common/sanitizer_tls_get_addr.h"

#include <stdarg.h>
// ACHTUNG! No other system header includes in this file.
// Ideally, we should get rid of stdarg.h as well.

using namespace __hwasan;

using __sanitizer::memory_order;
using __sanitizer::atomic_load;
using __sanitizer::atomic_store;
using __sanitizer::atomic_uintptr_t;

DECLARE_REAL(SIZE_T, strlen, const char *s)
DECLARE_REAL(SIZE_T, strnlen, const char *s, SIZE_T maxlen)
DECLARE_REAL(void *, memcpy, void *dest, const void *src, uptr n)
DECLARE_REAL(void *, memset, void *dest, int c, uptr n)

bool IsInInterceptorScope() {
  HwasanThread *t = GetCurrentThread();
  return t && t->InInterceptorScope();
}

struct InterceptorScope {
  InterceptorScope() {
    HwasanThread *t = GetCurrentThread();
    if (t)
      t->EnterInterceptorScope();
  }
  ~InterceptorScope() {
    HwasanThread *t = GetCurrentThread();
    if (t)
      t->LeaveInterceptorScope();
  }
};

static uptr allocated_for_dlsym;
static const uptr kDlsymAllocPoolSize = 1024;
static uptr alloc_memory_for_dlsym[kDlsymAllocPoolSize];

static bool IsInDlsymAllocPool(const void *ptr) {
  uptr off = (uptr)ptr - (uptr)alloc_memory_for_dlsym;
  return off < sizeof(alloc_memory_for_dlsym);
}

static void *AllocateFromLocalPool(uptr size_in_bytes) {
  uptr size_in_words = RoundUpTo(size_in_bytes, kWordSize) / kWordSize;
  void *mem = (void *)&alloc_memory_for_dlsym[allocated_for_dlsym];
  allocated_for_dlsym += size_in_words;
  CHECK_LT(allocated_for_dlsym, kDlsymAllocPoolSize);
  return mem;
}

#define ENSURE_HWASAN_INITED() do { \
  CHECK(!hwasan_init_is_running); \
  if (!hwasan_inited) { \
    __hwasan_init(); \
  } \
} while (0)



#define HWASAN_READ_RANGE(ctx, offset, size) \
  CHECK_UNPOISONED(offset, size)
#define HWASAN_WRITE_RANGE(ctx, offset, size) \
  CHECK_UNPOISONED(offset, size)



// Check that [x, x+n) range is unpoisoned.
#define CHECK_UNPOISONED_0(x, n)                                       \
  do {                                                                 \
    sptr __offset = __hwasan_test_shadow(x, n);                         \
    if (__hwasan::IsInSymbolizer()) break;                              \
    if (__offset >= 0) {                                               \
      GET_CALLER_PC_BP_SP;                                             \
      (void)sp;                                                        \
      ReportInvalidAccessInsideAddressRange(__func__, x, n, __offset); \
      __hwasan::PrintWarning(pc, bp);                                   \
      if (__hwasan::flags()->halt_on_error) {                           \
        Printf("Exiting\n");                                           \
        Die();                                                         \
      }                                                                \
    }                                                                  \
  } while (0)

// Check that [x, x+n) range is unpoisoned unless we are in a nested
// interceptor.
#define CHECK_UNPOISONED(x, n)                             \
  do {                                                     \
    if (!IsInInterceptorScope()) CHECK_UNPOISONED_0(x, n); \
  } while (0)

#define CHECK_UNPOISONED_STRING_OF_LEN(x, len, n)               \
  CHECK_UNPOISONED((x),                                         \
    common_flags()->strict_string_checks ? (len) + 1 : (n) )


INTERCEPTOR(int, posix_memalign, void **memptr, SIZE_T alignment, SIZE_T size) {
  GET_MALLOC_STACK_TRACE;
  CHECK_NE(memptr, 0);
  int res = hwasan_posix_memalign(memptr, alignment, size, &stack);
  return res;
}

#if !SANITIZER_FREEBSD && !SANITIZER_NETBSD
INTERCEPTOR(void *, memalign, SIZE_T alignment, SIZE_T size) {
  GET_MALLOC_STACK_TRACE;
  return hwasan_memalign(alignment, size, &stack);
}
#define HWASAN_MAYBE_INTERCEPT_MEMALIGN INTERCEPT_FUNCTION(memalign)
#else
#define HWASAN_MAYBE_INTERCEPT_MEMALIGN
#endif

INTERCEPTOR(void *, aligned_alloc, SIZE_T alignment, SIZE_T size) {
  GET_MALLOC_STACK_TRACE;
  return hwasan_aligned_alloc(alignment, size, &stack);
}

INTERCEPTOR(void *, __libc_memalign, SIZE_T alignment, SIZE_T size) {
  GET_MALLOC_STACK_TRACE;
  void *ptr = hwasan_memalign(alignment, size, &stack);
  if (ptr)
    DTLS_on_libc_memalign(ptr, size);
  return ptr;
}

INTERCEPTOR(void *, valloc, SIZE_T size) {
  GET_MALLOC_STACK_TRACE;
  return hwasan_valloc(size, &stack);
}

#if !SANITIZER_FREEBSD && !SANITIZER_NETBSD
INTERCEPTOR(void *, pvalloc, SIZE_T size) {
  GET_MALLOC_STACK_TRACE;
  return hwasan_pvalloc(size, &stack);
}
#define HWASAN_MAYBE_INTERCEPT_PVALLOC INTERCEPT_FUNCTION(pvalloc)
#else
#define HWASAN_MAYBE_INTERCEPT_PVALLOC
#endif

INTERCEPTOR(void, free, void *ptr) {
  GET_MALLOC_STACK_TRACE;
  if (!ptr || UNLIKELY(IsInDlsymAllocPool(ptr))) return;
  HwasanDeallocate(&stack, ptr);
}

#if !SANITIZER_FREEBSD && !SANITIZER_NETBSD
INTERCEPTOR(void, cfree, void *ptr) {
  GET_MALLOC_STACK_TRACE;
  if (!ptr || UNLIKELY(IsInDlsymAllocPool(ptr))) return;
  HwasanDeallocate(&stack, ptr);
}
#define HWASAN_MAYBE_INTERCEPT_CFREE INTERCEPT_FUNCTION(cfree)
#else
#define HWASAN_MAYBE_INTERCEPT_CFREE
#endif

INTERCEPTOR(uptr, malloc_usable_size, void *ptr) {
  return __sanitizer_get_allocated_size(ptr);
}

#if !SANITIZER_FREEBSD && !SANITIZER_NETBSD
// This function actually returns a struct by value, but we can't unpoison a
// temporary! The following is equivalent on all supported platforms but
// aarch64 (which uses a different register for sret value).  We have a test
// to confirm that.
INTERCEPTOR(void, mallinfo, __sanitizer_mallinfo *sret) {
#ifdef __aarch64__
  uptr r8;
  asm volatile("mov %0,x8" : "=r" (r8));
  sret = reinterpret_cast<__sanitizer_mallinfo*>(r8);
#endif
  REAL(memset)(sret, 0, sizeof(*sret));
}
#define HWASAN_MAYBE_INTERCEPT_MALLINFO INTERCEPT_FUNCTION(mallinfo)
#else
#define HWASAN_MAYBE_INTERCEPT_MALLINFO
#endif

#if !SANITIZER_FREEBSD && !SANITIZER_NETBSD
INTERCEPTOR(int, mallopt, int cmd, int value) {
  return -1;
}
#define HWASAN_MAYBE_INTERCEPT_MALLOPT INTERCEPT_FUNCTION(mallopt)
#else
#define HWASAN_MAYBE_INTERCEPT_MALLOPT
#endif

#if !SANITIZER_FREEBSD && !SANITIZER_NETBSD
INTERCEPTOR(void, malloc_stats, void) {
  // FIXME: implement, but don't call REAL(malloc_stats)!
}
#define HWASAN_MAYBE_INTERCEPT_MALLOC_STATS INTERCEPT_FUNCTION(malloc_stats)
#else
#define HWASAN_MAYBE_INTERCEPT_MALLOC_STATS
#endif


INTERCEPTOR(void *, calloc, SIZE_T nmemb, SIZE_T size) {
  GET_MALLOC_STACK_TRACE;
  if (UNLIKELY(!hwasan_inited))
    // Hack: dlsym calls calloc before REAL(calloc) is retrieved from dlsym.
    return AllocateFromLocalPool(nmemb * size);
  return hwasan_calloc(nmemb, size, &stack);
}

INTERCEPTOR(void *, realloc, void *ptr, SIZE_T size) {
  GET_MALLOC_STACK_TRACE;
  if (UNLIKELY(IsInDlsymAllocPool(ptr))) {
    uptr offset = (uptr)ptr - (uptr)alloc_memory_for_dlsym;
    uptr copy_size = Min(size, kDlsymAllocPoolSize - offset);
    void *new_ptr;
    if (UNLIKELY(!hwasan_inited)) {
      new_ptr = AllocateFromLocalPool(copy_size);
    } else {
      copy_size = size;
      new_ptr = hwasan_malloc(copy_size, &stack);
    }
    internal_memcpy(new_ptr, ptr, copy_size);
    return new_ptr;
  }
  return hwasan_realloc(ptr, size, &stack);
}

INTERCEPTOR(void *, malloc, SIZE_T size) {
  GET_MALLOC_STACK_TRACE;
  if (UNLIKELY(!hwasan_inited))
    // Hack: dlsym calls malloc before REAL(malloc) is retrieved from dlsym.
    return AllocateFromLocalPool(size);
  return hwasan_malloc(size, &stack);
}


INTERCEPTOR(void *, mmap, void *addr, SIZE_T length, int prot, int flags,
            int fd, OFF_T offset) {
  if (hwasan_init_is_running)
    return REAL(mmap)(addr, length, prot, flags, fd, offset);
  ENSURE_HWASAN_INITED();
  if (addr && !MEM_IS_APP(addr)) {
    if (flags & map_fixed) {
      errno = errno_EINVAL;
      return (void *)-1;
    } else {
      addr = nullptr;
    }
  }
  void *res = REAL(mmap)(addr, length, prot, flags, fd, offset);
  return res;
}

#if !SANITIZER_FREEBSD && !SANITIZER_NETBSD
INTERCEPTOR(void *, mmap64, void *addr, SIZE_T length, int prot, int flags,
            int fd, OFF64_T offset) {
  ENSURE_HWASAN_INITED();
  if (addr && !MEM_IS_APP(addr)) {
    if (flags & map_fixed) {
      errno = errno_EINVAL;
      return (void *)-1;
    } else {
      addr = nullptr;
    }
  }
  void *res = REAL(mmap64)(addr, length, prot, flags, fd, offset);
  return res;
}
#define HWASAN_MAYBE_INTERCEPT_MMAP64 INTERCEPT_FUNCTION(mmap64)
#else
#define HWASAN_MAYBE_INTERCEPT_MMAP64
#endif

extern "C" int pthread_attr_init(void *attr);
extern "C" int pthread_attr_destroy(void *attr);

static void *HwasanThreadStartFunc(void *arg) {
  HwasanThread *t = (HwasanThread *)arg;
  SetCurrentThread(t);
  return t->ThreadStart();
}

INTERCEPTOR(int, pthread_create, void *th, void *attr, void *(*callback)(void*),
            void * param) {
  ENSURE_HWASAN_INITED(); // for GetTlsSize()
  __sanitizer_pthread_attr_t myattr;
  if (!attr) {
    pthread_attr_init(&myattr);
    attr = &myattr;
  }

  AdjustStackSize(attr);

  HwasanThread *t = HwasanThread::Create(callback, param);

  int res = REAL(pthread_create)(th, attr, HwasanThreadStartFunc, t);

  if (attr == &myattr)
    pthread_attr_destroy(&myattr);
  return res;
}

static void BeforeFork() {
  StackDepotLockAll();
}

static void AfterFork() {
  StackDepotUnlockAll();
}

INTERCEPTOR(int, fork, void) {
  ENSURE_HWASAN_INITED();
  BeforeFork();
  int pid = REAL(fork)();
  AfterFork();
  return pid;
}


struct HwasanInterceptorContext {
  bool in_interceptor_scope;
};

namespace __hwasan {

int OnExit() {
  // FIXME: ask frontend whether we need to return failure.
  return 0;
}

} // namespace __hwasan

// A version of CHECK_UNPOISONED using a saved scope value. Used in common
// interceptors.
#define CHECK_UNPOISONED_CTX(ctx, x, n)                         \
  do {                                                          \
    if (!((HwasanInterceptorContext *)ctx)->in_interceptor_scope) \
      CHECK_UNPOISONED_0(x, n);                                 \
  } while (0)

#define HWASAN_INTERCEPT_FUNC(name)                                       \
  do {                                                                  \
    if ((!INTERCEPT_FUNCTION(name) || !REAL(name)))                     \
      VReport(1, "HWAddressSanitizer: failed to intercept '" #name "'\n"); \
  } while (0)

#define HWASAN_INTERCEPT_FUNC_VER(name, ver)                                    \
  do {                                                                        \
    if ((!INTERCEPT_FUNCTION_VER(name, ver) || !REAL(name)))                  \
      VReport(                                                                \
          1, "HWAddressSanitizer: failed to intercept '" #name "@@" #ver "'\n"); \
  } while (0)

#define COMMON_INTERCEPT_FUNCTION(name) HWASAN_INTERCEPT_FUNC(name)
#define COMMON_INTERCEPT_FUNCTION_VER(name, ver)                          \
  HWASAN_INTERCEPT_FUNC_VER(name, ver)
#define COMMON_INTERCEPTOR_WRITE_RANGE(ctx, ptr, size) \
  CHECK_UNPOISONED_CTX(ctx, ptr, size)
#define COMMON_INTERCEPTOR_READ_RANGE(ctx, ptr, size) \
  CHECK_UNPOISONED_CTX(ctx, ptr, size)
#define COMMON_INTERCEPTOR_INITIALIZE_RANGE(ptr, size) \
  HWASAN_WRITE_RANGE(ctx, ptr, size)
#define COMMON_INTERCEPTOR_ENTER(ctx, func, ...)                  \
  if (hwasan_init_is_running) return REAL(func)(__VA_ARGS__);       \
  ENSURE_HWASAN_INITED();                                           \
  HwasanInterceptorContext hwasan_ctx = {IsInInterceptorScope()};     \
  ctx = (void *)&hwasan_ctx;                                        \
  (void)ctx;                                                      \
  InterceptorScope interceptor_scope;
#define COMMON_INTERCEPTOR_DIR_ACQUIRE(ctx, path) \
  do {                                            \
  } while (false)
#define COMMON_INTERCEPTOR_FD_ACQUIRE(ctx, fd) \
  do {                                         \
  } while (false)
#define COMMON_INTERCEPTOR_FD_RELEASE(ctx, fd) \
  do {                                         \
  } while (false)
#define COMMON_INTERCEPTOR_FD_SOCKET_ACCEPT(ctx, fd, newfd) \
  do {                                                      \
  } while (false)
#define COMMON_INTERCEPTOR_SET_THREAD_NAME(ctx, name) \
  do {                                                \
  } while (false)  // FIXME
#define COMMON_INTERCEPTOR_SET_PTHREAD_NAME(ctx, thread, name) \
  do {                                                         \
  } while (false)  // FIXME
#define COMMON_INTERCEPTOR_BLOCK_REAL(name) REAL(name)
#define COMMON_INTERCEPTOR_ON_EXIT(ctx) OnExit()

#define COMMON_INTERCEPTOR_GET_TLS_RANGE(begin, end)                           \
  if (HwasanThread *t = GetCurrentThread()) {                                    \
    *begin = t->tls_begin();                                                   \
    *end = t->tls_end();                                                       \
  } else {                                                                     \
    *begin = *end = 0;                                                         \
  }

#include "sanitizer_common/sanitizer_platform_interceptors.h"
#include "sanitizer_common/sanitizer_common_interceptors.inc"
#include "sanitizer_common/sanitizer_signal_interceptors.inc"

#define COMMON_SYSCALL_PRE_READ_RANGE(p, s) CHECK_UNPOISONED(p, s)
#define COMMON_SYSCALL_PRE_WRITE_RANGE(p, s) \
  do {                                       \
    (void)(p);                               \
    (void)(s);                               \
  } while (false)
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
#include "sanitizer_common/sanitizer_common_syscalls.inc"



namespace __hwasan {

void InitializeInterceptors() {
  static int inited = 0;
  CHECK_EQ(inited, 0);
  InitializeCommonInterceptors();
  InitializeSignalInterceptors();

  INTERCEPT_FUNCTION(mmap);
  HWASAN_MAYBE_INTERCEPT_MMAP64;
  INTERCEPT_FUNCTION(posix_memalign);
  HWASAN_MAYBE_INTERCEPT_MEMALIGN;
  INTERCEPT_FUNCTION(__libc_memalign);
  INTERCEPT_FUNCTION(valloc);
  HWASAN_MAYBE_INTERCEPT_PVALLOC;
  INTERCEPT_FUNCTION(malloc);
  INTERCEPT_FUNCTION(calloc);
  INTERCEPT_FUNCTION(realloc);
  INTERCEPT_FUNCTION(free);
  HWASAN_MAYBE_INTERCEPT_CFREE;
  INTERCEPT_FUNCTION(malloc_usable_size);
  HWASAN_MAYBE_INTERCEPT_MALLINFO;
  HWASAN_MAYBE_INTERCEPT_MALLOPT;
  HWASAN_MAYBE_INTERCEPT_MALLOC_STATS;
  INTERCEPT_FUNCTION(pthread_create);
  INTERCEPT_FUNCTION(fork);

  inited = 1;
}
} // namespace __hwasan

//===-- asan_interceptors.cc ----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// Intercept various libc functions.
//===----------------------------------------------------------------------===//

#include "asan_interceptors.h"
#include "asan_allocator.h"
#include "asan_internal.h"
#include "asan_mapping.h"
#include "asan_poisoning.h"
#include "asan_report.h"
#include "asan_stack.h"
#include "asan_stats.h"
#include "asan_suppressions.h"
#include "lsan/lsan_common.h"
#include "sanitizer_common/sanitizer_libc.h"

#if SANITIZER_POSIX
#include "sanitizer_common/sanitizer_posix.h"
#endif

#if defined(__i386) && SANITIZER_LINUX
#define ASAN_PTHREAD_CREATE_VERSION "GLIBC_2.1"
#elif defined(__mips__) && SANITIZER_LINUX
#define ASAN_PTHREAD_CREATE_VERSION "GLIBC_2.2"
#endif

namespace __asan {

// Return true if we can quickly decide that the region is unpoisoned.
// We assume that a redzone is at least 16 bytes.
static inline bool QuickCheckForUnpoisonedRegion(uptr beg, uptr size) {
  if (size == 0) return true;
  if (size <= 32)
    return !AddressIsPoisoned(beg) &&
           !AddressIsPoisoned(beg + size - 1) &&
           !AddressIsPoisoned(beg + size / 2);
  if (size <= 64)
    return !AddressIsPoisoned(beg) &&
           !AddressIsPoisoned(beg + size / 4) &&
           !AddressIsPoisoned(beg + size - 1) &&
           !AddressIsPoisoned(beg + 3 * size / 4) &&
           !AddressIsPoisoned(beg + size / 2);
  return false;
}

struct AsanInterceptorContext {
  const char *interceptor_name;
};

// We implement ACCESS_MEMORY_RANGE, ASAN_READ_RANGE,
// and ASAN_WRITE_RANGE as macro instead of function so
// that no extra frames are created, and stack trace contains
// relevant information only.
// We check all shadow bytes.
#define ACCESS_MEMORY_RANGE(ctx, offset, size, isWrite) do {            \
    uptr __offset = (uptr)(offset);                                     \
    uptr __size = (uptr)(size);                                         \
    uptr __bad = 0;                                                     \
    if (__offset > __offset + __size) {                                 \
      GET_STACK_TRACE_FATAL_HERE;                                       \
      ReportStringFunctionSizeOverflow(__offset, __size, &stack);       \
    }                                                                   \
    if (!QuickCheckForUnpoisonedRegion(__offset, __size) &&             \
        (__bad = __asan_region_is_poisoned(__offset, __size))) {        \
      AsanInterceptorContext *_ctx = (AsanInterceptorContext *)ctx;     \
      bool suppressed = false;                                          \
      if (_ctx) {                                                       \
        suppressed = IsInterceptorSuppressed(_ctx->interceptor_name);   \
        if (!suppressed && HaveStackTraceBasedSuppressions()) {         \
          GET_STACK_TRACE_FATAL_HERE;                                   \
          suppressed = IsStackTraceSuppressed(&stack);                  \
        }                                                               \
      }                                                                 \
      if (!suppressed) {                                                \
        GET_CURRENT_PC_BP_SP;                                           \
        ReportGenericError(pc, bp, sp, __bad, isWrite, __size, 0, false);\
      }                                                                 \
    }                                                                   \
  } while (0)

// memcpy is called during __asan_init() from the internals of printf(...).
// We do not treat memcpy with to==from as a bug.
// See http://llvm.org/bugs/show_bug.cgi?id=11763.
#define ASAN_MEMCPY_IMPL(ctx, to, from, size)                           \
  do {                                                                  \
    if (UNLIKELY(!asan_inited)) return internal_memcpy(to, from, size); \
    if (asan_init_is_running) {                                         \
      return REAL(memcpy)(to, from, size);                              \
    }                                                                   \
    ENSURE_ASAN_INITED();                                               \
    if (flags()->replace_intrin) {                                      \
      if (to != from) {                                                 \
        CHECK_RANGES_OVERLAP("memcpy", to, size, from, size);           \
      }                                                                 \
      ASAN_READ_RANGE(ctx, from, size);                                 \
      ASAN_WRITE_RANGE(ctx, to, size);                                  \
    }                                                                   \
    return REAL(memcpy)(to, from, size);                                \
  } while (0)

// memset is called inside Printf.
#define ASAN_MEMSET_IMPL(ctx, block, c, size)                           \
  do {                                                                  \
    if (UNLIKELY(!asan_inited)) return internal_memset(block, c, size); \
    if (asan_init_is_running) {                                         \
      return REAL(memset)(block, c, size);                              \
    }                                                                   \
    ENSURE_ASAN_INITED();                                               \
    if (flags()->replace_intrin) {                                      \
      ASAN_WRITE_RANGE(ctx, block, size);                               \
    }                                                                   \
    return REAL(memset)(block, c, size);                                \
  } while (0)

#define ASAN_MEMMOVE_IMPL(ctx, to, from, size)                           \
  do {                                                                   \
    if (UNLIKELY(!asan_inited)) return internal_memmove(to, from, size); \
    ENSURE_ASAN_INITED();                                                \
    if (flags()->replace_intrin) {                                       \
      ASAN_READ_RANGE(ctx, from, size);                                  \
      ASAN_WRITE_RANGE(ctx, to, size);                                   \
    }                                                                    \
    return internal_memmove(to, from, size);                             \
  } while (0)

#define ASAN_READ_RANGE(ctx, offset, size) \
  ACCESS_MEMORY_RANGE(ctx, offset, size, false)
#define ASAN_WRITE_RANGE(ctx, offset, size) \
  ACCESS_MEMORY_RANGE(ctx, offset, size, true)

#define ASAN_READ_STRING_OF_LEN(ctx, s, len, n)                 \
  ASAN_READ_RANGE((ctx), (s),                                   \
    common_flags()->strict_string_checks ? (len) + 1 : (n))

#define ASAN_READ_STRING(ctx, s, n)                             \
  ASAN_READ_STRING_OF_LEN((ctx), (s), REAL(strlen)(s), (n))

// Behavior of functions like "memcpy" or "strcpy" is undefined
// if memory intervals overlap. We report error in this case.
// Macro is used to avoid creation of new frames.
static inline bool RangesOverlap(const char *offset1, uptr length1,
                                 const char *offset2, uptr length2) {
  return !((offset1 + length1 <= offset2) || (offset2 + length2 <= offset1));
}
#define CHECK_RANGES_OVERLAP(name, _offset1, length1, _offset2, length2) do { \
  const char *offset1 = (const char*)_offset1; \
  const char *offset2 = (const char*)_offset2; \
  if (RangesOverlap(offset1, length1, offset2, length2)) { \
    GET_STACK_TRACE_FATAL_HERE; \
    ReportStringFunctionMemoryRangesOverlap(name, offset1, length1, \
                                            offset2, length2, &stack); \
  } \
} while (0)

static inline uptr MaybeRealStrnlen(const char *s, uptr maxlen) {
#if SANITIZER_INTERCEPT_STRNLEN
  if (REAL(strnlen)) {
    return REAL(strnlen)(s, maxlen);
  }
#endif
  return internal_strnlen(s, maxlen);
}

void SetThreadName(const char *name) {
  AsanThread *t = GetCurrentThread();
  if (t)
    asanThreadRegistry().SetThreadName(t->tid(), name);
}

int OnExit() {
  // FIXME: ask frontend whether we need to return failure.
  return 0;
}

} // namespace __asan

// ---------------------- Wrappers ---------------- {{{1
using namespace __asan;  // NOLINT

DECLARE_REAL_AND_INTERCEPTOR(void *, malloc, uptr)
DECLARE_REAL_AND_INTERCEPTOR(void, free, void *)

#define ASAN_INTERCEPTOR_ENTER(ctx, func)                                      \
  AsanInterceptorContext _ctx = {#func};                                       \
  ctx = (void *)&_ctx;                                                         \
  (void) ctx;                                                                  \

#define COMMON_INTERCEPT_FUNCTION(name) ASAN_INTERCEPT_FUNC(name)
#define COMMON_INTERCEPT_FUNCTION_VER(name, ver)                          \
  ASAN_INTERCEPT_FUNC_VER(name, ver)
#define COMMON_INTERCEPTOR_WRITE_RANGE(ctx, ptr, size) \
  ASAN_WRITE_RANGE(ctx, ptr, size)
#define COMMON_INTERCEPTOR_READ_RANGE(ctx, ptr, size) \
  ASAN_READ_RANGE(ctx, ptr, size)
#define COMMON_INTERCEPTOR_ENTER(ctx, func, ...)                               \
  ASAN_INTERCEPTOR_ENTER(ctx, func);                                           \
  do {                                                                         \
    if (asan_init_is_running)                                                  \
      return REAL(func)(__VA_ARGS__);                                          \
    if (SANITIZER_MAC && UNLIKELY(!asan_inited))                               \
      return REAL(func)(__VA_ARGS__);                                          \
    ENSURE_ASAN_INITED();                                                      \
  } while (false)
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
#define COMMON_INTERCEPTOR_SET_THREAD_NAME(ctx, name) SetThreadName(name)
// Should be asanThreadRegistry().SetThreadNameByUserId(thread, name)
// But asan does not remember UserId's for threads (pthread_t);
// and remembers all ever existed threads, so the linear search by UserId
// can be slow.
#define COMMON_INTERCEPTOR_SET_PTHREAD_NAME(ctx, thread, name) \
  do {                                                         \
  } while (false)
#define COMMON_INTERCEPTOR_BLOCK_REAL(name) REAL(name)
// Strict init-order checking is dlopen-hostile:
// https://github.com/google/sanitizers/issues/178
#define COMMON_INTERCEPTOR_ON_DLOPEN(filename, flag)                           \
  do {                                                                         \
    if (flags()->strict_init_order)                                            \
      StopInitOrderChecking();                                                 \
    CheckNoDeepBind(filename, flag);                                           \
  } while (false)
#define COMMON_INTERCEPTOR_ON_EXIT(ctx) OnExit()
#define COMMON_INTERCEPTOR_LIBRARY_LOADED(filename, handle)
#define COMMON_INTERCEPTOR_LIBRARY_UNLOADED()
#define COMMON_INTERCEPTOR_NOTHING_IS_INITIALIZED (!asan_inited)
#define COMMON_INTERCEPTOR_GET_TLS_RANGE(begin, end)                           \
  if (AsanThread *t = GetCurrentThread()) {                                    \
    *begin = t->tls_begin();                                                   \
    *end = t->tls_end();                                                       \
  } else {                                                                     \
    *begin = *end = 0;                                                         \
  }

#define COMMON_INTERCEPTOR_MEMMOVE_IMPL(ctx, to, from, size) \
  do {                                                       \
    ASAN_INTERCEPTOR_ENTER(ctx, memmove);                    \
    ASAN_MEMMOVE_IMPL(ctx, to, from, size);                  \
  } while (false)

#define COMMON_INTERCEPTOR_MEMCPY_IMPL(ctx, to, from, size) \
  do {                                                      \
    ASAN_INTERCEPTOR_ENTER(ctx, memcpy);                    \
    ASAN_MEMCPY_IMPL(ctx, to, from, size);                  \
  } while (false)

#define COMMON_INTERCEPTOR_MEMSET_IMPL(ctx, block, c, size) \
  do {                                                      \
    ASAN_INTERCEPTOR_ENTER(ctx, memset);                    \
    ASAN_MEMSET_IMPL(ctx, block, c, size);                  \
  } while (false)

#include "sanitizer_common/sanitizer_common_interceptors.inc"

// Syscall interceptors don't have contexts, we don't support suppressions
// for them.
#define COMMON_SYSCALL_PRE_READ_RANGE(p, s) ASAN_READ_RANGE(nullptr, p, s)
#define COMMON_SYSCALL_PRE_WRITE_RANGE(p, s) ASAN_WRITE_RANGE(nullptr, p, s)
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

struct ThreadStartParam {
  atomic_uintptr_t t;
  atomic_uintptr_t is_registered;
};

#if ASAN_INTERCEPT_PTHREAD_CREATE
static thread_return_t THREAD_CALLING_CONV asan_thread_start(void *arg) {
  ThreadStartParam *param = reinterpret_cast<ThreadStartParam *>(arg);
  AsanThread *t = nullptr;
  while ((t = reinterpret_cast<AsanThread *>(
              atomic_load(&param->t, memory_order_acquire))) == nullptr)
    internal_sched_yield();
  SetCurrentThread(t);
  return t->ThreadStart(GetTid(), &param->is_registered);
}

INTERCEPTOR(int, pthread_create, void *thread,
    void *attr, void *(*start_routine)(void*), void *arg) {
  EnsureMainThreadIDIsCorrect();
  // Strict init-order checking is thread-hostile.
  if (flags()->strict_init_order)
    StopInitOrderChecking();
  GET_STACK_TRACE_THREAD;
  int detached = 0;
  if (attr)
    REAL(pthread_attr_getdetachstate)(attr, &detached);
  ThreadStartParam param;
  atomic_store(&param.t, 0, memory_order_relaxed);
  atomic_store(&param.is_registered, 0, memory_order_relaxed);
  int result;
  {
    // Ignore all allocations made by pthread_create: thread stack/TLS may be
    // stored by pthread for future reuse even after thread destruction, and
    // the linked list it's stored in doesn't even hold valid pointers to the
    // objects, the latter are calculated by obscure pointer arithmetic.
#if CAN_SANITIZE_LEAKS
    __lsan::ScopedInterceptorDisabler disabler;
#endif
    result = REAL(pthread_create)(thread, attr, asan_thread_start, &param);
  }
  if (result == 0) {
    u32 current_tid = GetCurrentTidOrInvalid();
    AsanThread *t =
        AsanThread::Create(start_routine, arg, current_tid, &stack, detached);
    atomic_store(&param.t, reinterpret_cast<uptr>(t), memory_order_release);
    // Wait until the AsanThread object is initialized and the ThreadRegistry
    // entry is in "started" state. One reason for this is that after this
    // interceptor exits, the child thread's stack may be the only thing holding
    // the |arg| pointer. This may cause LSan to report a leak if leak checking
    // happens at a point when the interceptor has already exited, but the stack
    // range for the child thread is not yet known.
    while (atomic_load(&param.is_registered, memory_order_acquire) == 0)
      internal_sched_yield();
  }
  return result;
}

INTERCEPTOR(int, pthread_join, void *t, void **arg) {
  return real_pthread_join(t, arg);
}

DEFINE_REAL_PTHREAD_FUNCTIONS
#endif  // ASAN_INTERCEPT_PTHREAD_CREATE

#if ASAN_INTERCEPT_SIGNAL_AND_SIGACTION

#if SANITIZER_ANDROID
INTERCEPTOR(void*, bsd_signal, int signum, void *handler) {
  if (GetHandleSignalMode(signum) != kHandleSignalExclusive)
    return REAL(bsd_signal)(signum, handler);
  return 0;
}
#endif

INTERCEPTOR(void*, signal, int signum, void *handler) {
  if (GetHandleSignalMode(signum) != kHandleSignalExclusive)
    return REAL(signal)(signum, handler);
  return nullptr;
}

INTERCEPTOR(int, sigaction, int signum, const struct sigaction *act,
                            struct sigaction *oldact) {
  if (GetHandleSignalMode(signum) != kHandleSignalExclusive)
    return REAL(sigaction)(signum, act, oldact);
  return 0;
}

namespace __sanitizer {
int real_sigaction(int signum, const void *act, void *oldact) {
  return REAL(sigaction)(signum, (const struct sigaction *)act,
                         (struct sigaction *)oldact);
}
} // namespace __sanitizer

#elif SANITIZER_POSIX
// We need to have defined REAL(sigaction) on posix systems.
DEFINE_REAL(int, sigaction, int signum, const struct sigaction *act,
    struct sigaction *oldact)
#endif  // ASAN_INTERCEPT_SIGNAL_AND_SIGACTION

#if ASAN_INTERCEPT_SWAPCONTEXT
static void ClearShadowMemoryForContextStack(uptr stack, uptr ssize) {
  // Align to page size.
  uptr PageSize = GetPageSizeCached();
  uptr bottom = stack & ~(PageSize - 1);
  ssize += stack - bottom;
  ssize = RoundUpTo(ssize, PageSize);
  static const uptr kMaxSaneContextStackSize = 1 << 22;  // 4 Mb
  if (AddrIsInMem(bottom) && ssize && ssize <= kMaxSaneContextStackSize) {
    PoisonShadow(bottom, ssize, 0);
  }
}

INTERCEPTOR(int, swapcontext, struct ucontext_t *oucp,
            struct ucontext_t *ucp) {
  static bool reported_warning = false;
  if (!reported_warning) {
    Report("WARNING: ASan doesn't fully support makecontext/swapcontext "
           "functions and may produce false positives in some cases!\n");
    reported_warning = true;
  }
  // Clear shadow memory for new context (it may share stack
  // with current context).
  uptr stack, ssize;
  ReadContextStack(ucp, &stack, &ssize);
  ClearShadowMemoryForContextStack(stack, ssize);
  int res = REAL(swapcontext)(oucp, ucp);
  // swapcontext technically does not return, but program may swap context to
  // "oucp" later, that would look as if swapcontext() returned 0.
  // We need to clear shadow for ucp once again, as it may be in arbitrary
  // state.
  ClearShadowMemoryForContextStack(stack, ssize);
  return res;
}
#endif  // ASAN_INTERCEPT_SWAPCONTEXT

INTERCEPTOR(void, longjmp, void *env, int val) {
  __asan_handle_no_return();
  REAL(longjmp)(env, val);
}

#if ASAN_INTERCEPT__LONGJMP
INTERCEPTOR(void, _longjmp, void *env, int val) {
  __asan_handle_no_return();
  REAL(_longjmp)(env, val);
}
#endif

#if ASAN_INTERCEPT___LONGJMP_CHK
INTERCEPTOR(void, __longjmp_chk, void *env, int val) {
  __asan_handle_no_return();
  REAL(__longjmp_chk)(env, val);
}
#endif

#if ASAN_INTERCEPT_SIGLONGJMP
INTERCEPTOR(void, siglongjmp, void *env, int val) {
  __asan_handle_no_return();
  REAL(siglongjmp)(env, val);
}
#endif

#if ASAN_INTERCEPT___CXA_THROW
INTERCEPTOR(void, __cxa_throw, void *a, void *b, void *c) {
  CHECK(REAL(__cxa_throw));
  __asan_handle_no_return();
  REAL(__cxa_throw)(a, b, c);
}
#endif

void *__asan_memcpy(void *to, const void *from, uptr size) {
  ASAN_MEMCPY_IMPL(nullptr, to, from, size);
}

void *__asan_memset(void *block, int c, uptr size) {
  ASAN_MEMSET_IMPL(nullptr, block, c, size);
}

void *__asan_memmove(void *to, const void *from, uptr size) {
  ASAN_MEMMOVE_IMPL(nullptr, to, from, size);
}

#if ASAN_INTERCEPT_INDEX
# if ASAN_USE_ALIAS_ATTRIBUTE_FOR_INDEX
INTERCEPTOR(char*, index, const char *string, int c)
  ALIAS(WRAPPER_NAME(strchr));
# else
#  if SANITIZER_MAC
DECLARE_REAL(char*, index, const char *string, int c)
OVERRIDE_FUNCTION(index, strchr);
#  else
DEFINE_REAL(char*, index, const char *string, int c)
#  endif
# endif
#endif  // ASAN_INTERCEPT_INDEX

// For both strcat() and strncat() we need to check the validity of |to|
// argument irrespective of the |from| length.
INTERCEPTOR(char*, strcat, char *to, const char *from) {  // NOLINT
  void *ctx;
  ASAN_INTERCEPTOR_ENTER(ctx, strcat);  // NOLINT
  ENSURE_ASAN_INITED();
  if (flags()->replace_str) {
    uptr from_length = REAL(strlen)(from);
    ASAN_READ_RANGE(ctx, from, from_length + 1);
    uptr to_length = REAL(strlen)(to);
    ASAN_READ_STRING_OF_LEN(ctx, to, to_length, to_length);
    ASAN_WRITE_RANGE(ctx, to + to_length, from_length + 1);
    // If the copying actually happens, the |from| string should not overlap
    // with the resulting string starting at |to|, which has a length of
    // to_length + from_length + 1.
    if (from_length > 0) {
      CHECK_RANGES_OVERLAP("strcat", to, from_length + to_length + 1,
                           from, from_length + 1);
    }
  }
  return REAL(strcat)(to, from);  // NOLINT
}

INTERCEPTOR(char*, strncat, char *to, const char *from, uptr size) {
  void *ctx;
  ASAN_INTERCEPTOR_ENTER(ctx, strncat);
  ENSURE_ASAN_INITED();
  if (flags()->replace_str) {
    uptr from_length = MaybeRealStrnlen(from, size);
    uptr copy_length = Min(size, from_length + 1);
    ASAN_READ_RANGE(ctx, from, copy_length);
    uptr to_length = REAL(strlen)(to);
    ASAN_READ_STRING_OF_LEN(ctx, to, to_length, to_length);
    ASAN_WRITE_RANGE(ctx, to + to_length, from_length + 1);
    if (from_length > 0) {
      CHECK_RANGES_OVERLAP("strncat", to, to_length + copy_length + 1,
                           from, copy_length);
    }
  }
  return REAL(strncat)(to, from, size);
}

INTERCEPTOR(char*, strcpy, char *to, const char *from) {  // NOLINT
  void *ctx;
  ASAN_INTERCEPTOR_ENTER(ctx, strcpy);  // NOLINT
#if SANITIZER_MAC
  if (UNLIKELY(!asan_inited)) return REAL(strcpy)(to, from);  // NOLINT
#endif
  // strcpy is called from malloc_default_purgeable_zone()
  // in __asan::ReplaceSystemAlloc() on Mac.
  if (asan_init_is_running) {
    return REAL(strcpy)(to, from);  // NOLINT
  }
  ENSURE_ASAN_INITED();
  if (flags()->replace_str) {
    uptr from_size = REAL(strlen)(from) + 1;
    CHECK_RANGES_OVERLAP("strcpy", to, from_size, from, from_size);
    ASAN_READ_RANGE(ctx, from, from_size);
    ASAN_WRITE_RANGE(ctx, to, from_size);
  }
  return REAL(strcpy)(to, from);  // NOLINT
}

INTERCEPTOR(char*, strdup, const char *s) {
  void *ctx;
  ASAN_INTERCEPTOR_ENTER(ctx, strdup);
  if (UNLIKELY(!asan_inited)) return internal_strdup(s);
  ENSURE_ASAN_INITED();
  uptr length = REAL(strlen)(s);
  if (flags()->replace_str) {
    ASAN_READ_RANGE(ctx, s, length + 1);
  }
  GET_STACK_TRACE_MALLOC;
  void *new_mem = asan_malloc(length + 1, &stack);
  REAL(memcpy)(new_mem, s, length + 1);
  return reinterpret_cast<char*>(new_mem);
}

#if ASAN_INTERCEPT___STRDUP
INTERCEPTOR(char*, __strdup, const char *s) {
  void *ctx;
  ASAN_INTERCEPTOR_ENTER(ctx, strdup);
  if (UNLIKELY(!asan_inited)) return internal_strdup(s);
  ENSURE_ASAN_INITED();
  uptr length = REAL(strlen)(s);
  if (flags()->replace_str) {
    ASAN_READ_RANGE(ctx, s, length + 1);
  }
  GET_STACK_TRACE_MALLOC;
  void *new_mem = asan_malloc(length + 1, &stack);
  REAL(memcpy)(new_mem, s, length + 1);
  return reinterpret_cast<char*>(new_mem);
}
#endif // ASAN_INTERCEPT___STRDUP

INTERCEPTOR(char*, strncpy, char *to, const char *from, uptr size) {
  void *ctx;
  ASAN_INTERCEPTOR_ENTER(ctx, strncpy);
  ENSURE_ASAN_INITED();
  if (flags()->replace_str) {
    uptr from_size = Min(size, MaybeRealStrnlen(from, size) + 1);
    CHECK_RANGES_OVERLAP("strncpy", to, from_size, from, from_size);
    ASAN_READ_RANGE(ctx, from, from_size);
    ASAN_WRITE_RANGE(ctx, to, size);
  }
  return REAL(strncpy)(to, from, size);
}

INTERCEPTOR(long, strtol, const char *nptr,  // NOLINT
            char **endptr, int base) {
  void *ctx;
  ASAN_INTERCEPTOR_ENTER(ctx, strtol);
  ENSURE_ASAN_INITED();
  if (!flags()->replace_str) {
    return REAL(strtol)(nptr, endptr, base);
  }
  char *real_endptr;
  long result = REAL(strtol)(nptr, &real_endptr, base);  // NOLINT
  StrtolFixAndCheck(ctx, nptr, endptr, real_endptr, base);
  return result;
}

INTERCEPTOR(int, atoi, const char *nptr) {
  void *ctx;
  ASAN_INTERCEPTOR_ENTER(ctx, atoi);
#if SANITIZER_MAC
  if (UNLIKELY(!asan_inited)) return REAL(atoi)(nptr);
#endif
  ENSURE_ASAN_INITED();
  if (!flags()->replace_str) {
    return REAL(atoi)(nptr);
  }
  char *real_endptr;
  // "man atoi" tells that behavior of atoi(nptr) is the same as
  // strtol(nptr, 0, 10), i.e. it sets errno to ERANGE if the
  // parsed integer can't be stored in *long* type (even if it's
  // different from int). So, we just imitate this behavior.
  int result = REAL(strtol)(nptr, &real_endptr, 10);
  FixRealStrtolEndptr(nptr, &real_endptr);
  ASAN_READ_STRING(ctx, nptr, (real_endptr - nptr) + 1);
  return result;
}

INTERCEPTOR(long, atol, const char *nptr) {  // NOLINT
  void *ctx;
  ASAN_INTERCEPTOR_ENTER(ctx, atol);
#if SANITIZER_MAC
  if (UNLIKELY(!asan_inited)) return REAL(atol)(nptr);
#endif
  ENSURE_ASAN_INITED();
  if (!flags()->replace_str) {
    return REAL(atol)(nptr);
  }
  char *real_endptr;
  long result = REAL(strtol)(nptr, &real_endptr, 10);  // NOLINT
  FixRealStrtolEndptr(nptr, &real_endptr);
  ASAN_READ_STRING(ctx, nptr, (real_endptr - nptr) + 1);
  return result;
}

#if ASAN_INTERCEPT_ATOLL_AND_STRTOLL
INTERCEPTOR(long long, strtoll, const char *nptr,  // NOLINT
            char **endptr, int base) {
  void *ctx;
  ASAN_INTERCEPTOR_ENTER(ctx, strtoll);
  ENSURE_ASAN_INITED();
  if (!flags()->replace_str) {
    return REAL(strtoll)(nptr, endptr, base);
  }
  char *real_endptr;
  long long result = REAL(strtoll)(nptr, &real_endptr, base);  // NOLINT
  StrtolFixAndCheck(ctx, nptr, endptr, real_endptr, base);
  return result;
}

INTERCEPTOR(long long, atoll, const char *nptr) {  // NOLINT
  void *ctx;
  ASAN_INTERCEPTOR_ENTER(ctx, atoll);
  ENSURE_ASAN_INITED();
  if (!flags()->replace_str) {
    return REAL(atoll)(nptr);
  }
  char *real_endptr;
  long long result = REAL(strtoll)(nptr, &real_endptr, 10);  // NOLINT
  FixRealStrtolEndptr(nptr, &real_endptr);
  ASAN_READ_STRING(ctx, nptr, (real_endptr - nptr) + 1);
  return result;
}
#endif  // ASAN_INTERCEPT_ATOLL_AND_STRTOLL

#if ASAN_INTERCEPT___CXA_ATEXIT
static void AtCxaAtexit(void *unused) {
  (void)unused;
  StopInitOrderChecking();
}

INTERCEPTOR(int, __cxa_atexit, void (*func)(void *), void *arg,
            void *dso_handle) {
#if SANITIZER_MAC
  if (UNLIKELY(!asan_inited)) return REAL(__cxa_atexit)(func, arg, dso_handle);
#endif
  ENSURE_ASAN_INITED();
  int res = REAL(__cxa_atexit)(func, arg, dso_handle);
  REAL(__cxa_atexit)(AtCxaAtexit, nullptr, nullptr);
  return res;
}
#endif  // ASAN_INTERCEPT___CXA_ATEXIT

#if ASAN_INTERCEPT_FORK
INTERCEPTOR(int, fork, void) {
  ENSURE_ASAN_INITED();
  int pid = REAL(fork)();
  return pid;
}
#endif  // ASAN_INTERCEPT_FORK

// ---------------------- InitializeAsanInterceptors ---------------- {{{1
namespace __asan {
void InitializeAsanInterceptors() {
  static bool was_called_once;
  CHECK(!was_called_once);
  was_called_once = true;
  InitializeCommonInterceptors();

  // Intercept str* functions.
  ASAN_INTERCEPT_FUNC(strcat);  // NOLINT
  ASAN_INTERCEPT_FUNC(strcpy);  // NOLINT
  ASAN_INTERCEPT_FUNC(strncat);
  ASAN_INTERCEPT_FUNC(strncpy);
  ASAN_INTERCEPT_FUNC(strdup);
#if ASAN_INTERCEPT___STRDUP
  ASAN_INTERCEPT_FUNC(__strdup);
#endif
#if ASAN_INTERCEPT_INDEX && ASAN_USE_ALIAS_ATTRIBUTE_FOR_INDEX
  ASAN_INTERCEPT_FUNC(index);
#endif

  ASAN_INTERCEPT_FUNC(atoi);
  ASAN_INTERCEPT_FUNC(atol);
  ASAN_INTERCEPT_FUNC(strtol);
#if ASAN_INTERCEPT_ATOLL_AND_STRTOLL
  ASAN_INTERCEPT_FUNC(atoll);
  ASAN_INTERCEPT_FUNC(strtoll);
#endif

  // Intecept signal- and jump-related functions.
  ASAN_INTERCEPT_FUNC(longjmp);
#if ASAN_INTERCEPT_SIGNAL_AND_SIGACTION
  ASAN_INTERCEPT_FUNC(sigaction);
#if SANITIZER_ANDROID
  ASAN_INTERCEPT_FUNC(bsd_signal);
#endif
  ASAN_INTERCEPT_FUNC(signal);
#endif
#if ASAN_INTERCEPT_SWAPCONTEXT
  ASAN_INTERCEPT_FUNC(swapcontext);
#endif
#if ASAN_INTERCEPT__LONGJMP
  ASAN_INTERCEPT_FUNC(_longjmp);
#endif
#if ASAN_INTERCEPT___LONGJMP_CHK
  ASAN_INTERCEPT_FUNC(__longjmp_chk);
#endif
#if ASAN_INTERCEPT_SIGLONGJMP
  ASAN_INTERCEPT_FUNC(siglongjmp);
#endif

  // Intercept exception handling functions.
#if ASAN_INTERCEPT___CXA_THROW
  ASAN_INTERCEPT_FUNC(__cxa_throw);
#endif

  // Intercept threading-related functions
#if ASAN_INTERCEPT_PTHREAD_CREATE
#if defined(ASAN_PTHREAD_CREATE_VERSION)
  ASAN_INTERCEPT_FUNC_VER(pthread_create, ASAN_PTHREAD_CREATE_VERSION);
#else
  ASAN_INTERCEPT_FUNC(pthread_create);
#endif
  ASAN_INTERCEPT_FUNC(pthread_join);
#endif

  // Intercept atexit function.
#if ASAN_INTERCEPT___CXA_ATEXIT
  ASAN_INTERCEPT_FUNC(__cxa_atexit);
#endif

#if ASAN_INTERCEPT_FORK
  ASAN_INTERCEPT_FUNC(fork);
#endif

  InitializePlatformInterceptors();

  VReport(1, "AddressSanitizer: libc interceptors initialized\n");
}

} // namespace __asan

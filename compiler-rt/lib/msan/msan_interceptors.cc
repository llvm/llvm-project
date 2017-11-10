//===-- msan_interceptors.cc ----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of MemorySanitizer.
//
// Interceptors for standard library functions.
//
// FIXME: move as many interceptors as possible into
// sanitizer_common/sanitizer_common_interceptors.h
//===----------------------------------------------------------------------===//

#include "interception/interception.h"
#include "msan.h"
#include "msan_chained_origin_depot.h"
#include "msan_origin.h"
#include "msan_thread.h"
#include "msan_poisoning.h"
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

using namespace __msan;

using __sanitizer::memory_order;
using __sanitizer::atomic_load;
using __sanitizer::atomic_store;
using __sanitizer::atomic_uintptr_t;

DECLARE_REAL(SIZE_T, strlen, const char *s)
DECLARE_REAL(SIZE_T, strnlen, const char *s, SIZE_T maxlen)
DECLARE_REAL(void *, memcpy, void *dest, const void *src, uptr n)
DECLARE_REAL(void *, memset, void *dest, int c, uptr n)

// True if this is a nested interceptor.
static THREADLOCAL int in_interceptor_scope;

struct InterceptorScope {
  InterceptorScope() { ++in_interceptor_scope; }
  ~InterceptorScope() { --in_interceptor_scope; }
};

bool IsInInterceptorScope() {
  return in_interceptor_scope;
}

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

#define ENSURE_MSAN_INITED() do { \
  CHECK(!msan_init_is_running); \
  if (!msan_inited) { \
    __msan_init(); \
  } \
} while (0)

// Check that [x, x+n) range is unpoisoned.
#define CHECK_UNPOISONED_0(x, n)                                               \
  do {                                                                         \
    sptr offset = __msan_test_shadow(x, n);                                    \
    if (__msan::IsInSymbolizer())                                              \
      break;                                                                   \
    if (offset >= 0 && __msan::flags()->report_umrs) {                         \
      GET_CALLER_PC_BP_SP;                                                     \
      (void) sp;                                                               \
      ReportUMRInsideAddressRange(__func__, x, n, offset);                     \
      __msan::PrintWarningWithOrigin(                                          \
          pc, bp, __msan_get_origin((const char *)x + offset));                \
      if (__msan::flags()->halt_on_error) {                                    \
        Printf("Exiting\n");                                                   \
        Die();                                                                 \
      }                                                                        \
    }                                                                          \
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

#define CHECK_UNPOISONED_STRING(x, n)                           \
    CHECK_UNPOISONED_STRING_OF_LEN((x), internal_strlen(x), (n))

#if !SANITIZER_FREEBSD && !SANITIZER_NETBSD
INTERCEPTOR(SIZE_T, fread_unlocked, void *ptr, SIZE_T size, SIZE_T nmemb,
            void *file) {
  ENSURE_MSAN_INITED();
  SIZE_T res = REAL(fread_unlocked)(ptr, size, nmemb, file);
  if (res > 0)
    __msan_unpoison(ptr, res *size);
  return res;
}
#define MSAN_MAYBE_INTERCEPT_FREAD_UNLOCKED INTERCEPT_FUNCTION(fread_unlocked)
#else
#define MSAN_MAYBE_INTERCEPT_FREAD_UNLOCKED
#endif

INTERCEPTOR(SSIZE_T, readlink, const char *path, char *buf, SIZE_T bufsiz) {
  ENSURE_MSAN_INITED();
  CHECK_UNPOISONED_STRING(path, 0);
  SSIZE_T res = REAL(readlink)(path, buf, bufsiz);
  if (res > 0)
    __msan_unpoison(buf, res);
  return res;
}

INTERCEPTOR(void *, mempcpy, void *dest, const void *src, SIZE_T n) {
  return (char *)__msan_memcpy(dest, src, n) + n;
}

INTERCEPTOR(void *, memccpy, void *dest, const void *src, int c, SIZE_T n) {
  ENSURE_MSAN_INITED();
  void *res = REAL(memccpy)(dest, src, c, n);
  CHECK(!res || (res >= dest && res <= (char *)dest + n));
  SIZE_T sz = res ? (char *)res - (char *)dest : n;
  CHECK_UNPOISONED(src, sz);
  __msan_unpoison(dest, sz);
  return res;
}

INTERCEPTOR(void *, bcopy, const void *src, void *dest, SIZE_T n) {
  return __msan_memmove(dest, src, n);
}

INTERCEPTOR(int, posix_memalign, void **memptr, SIZE_T alignment, SIZE_T size) {
  GET_MALLOC_STACK_TRACE;
  CHECK_NE(memptr, 0);
  int res = msan_posix_memalign(memptr, alignment, size, &stack);
  if (!res)
    __msan_unpoison(memptr, sizeof(*memptr));
  return res;
}

#if !SANITIZER_FREEBSD && !SANITIZER_NETBSD
INTERCEPTOR(void *, memalign, SIZE_T alignment, SIZE_T size) {
  GET_MALLOC_STACK_TRACE;
  return msan_memalign(alignment, size, &stack);
}
#define MSAN_MAYBE_INTERCEPT_MEMALIGN INTERCEPT_FUNCTION(memalign)
#else
#define MSAN_MAYBE_INTERCEPT_MEMALIGN
#endif

INTERCEPTOR(void *, aligned_alloc, SIZE_T alignment, SIZE_T size) {
  GET_MALLOC_STACK_TRACE;
  return msan_aligned_alloc(alignment, size, &stack);
}

INTERCEPTOR(void *, __libc_memalign, SIZE_T alignment, SIZE_T size) {
  GET_MALLOC_STACK_TRACE;
  void *ptr = msan_memalign(alignment, size, &stack);
  if (ptr)
    DTLS_on_libc_memalign(ptr, size);
  return ptr;
}

INTERCEPTOR(void *, valloc, SIZE_T size) {
  GET_MALLOC_STACK_TRACE;
  return msan_valloc(size, &stack);
}

#if !SANITIZER_FREEBSD && !SANITIZER_NETBSD
INTERCEPTOR(void *, pvalloc, SIZE_T size) {
  GET_MALLOC_STACK_TRACE;
  return msan_pvalloc(size, &stack);
}
#define MSAN_MAYBE_INTERCEPT_PVALLOC INTERCEPT_FUNCTION(pvalloc)
#else
#define MSAN_MAYBE_INTERCEPT_PVALLOC
#endif

INTERCEPTOR(void, free, void *ptr) {
  GET_MALLOC_STACK_TRACE;
  if (!ptr || UNLIKELY(IsInDlsymAllocPool(ptr))) return;
  MsanDeallocate(&stack, ptr);
}

#if !SANITIZER_FREEBSD && !SANITIZER_NETBSD
INTERCEPTOR(void, cfree, void *ptr) {
  GET_MALLOC_STACK_TRACE;
  if (!ptr || UNLIKELY(IsInDlsymAllocPool(ptr))) return;
  MsanDeallocate(&stack, ptr);
}
#define MSAN_MAYBE_INTERCEPT_CFREE INTERCEPT_FUNCTION(cfree)
#else
#define MSAN_MAYBE_INTERCEPT_CFREE
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
  __msan_unpoison(sret, sizeof(*sret));
}
#define MSAN_MAYBE_INTERCEPT_MALLINFO INTERCEPT_FUNCTION(mallinfo)
#else
#define MSAN_MAYBE_INTERCEPT_MALLINFO
#endif

#if !SANITIZER_FREEBSD && !SANITIZER_NETBSD
INTERCEPTOR(int, mallopt, int cmd, int value) {
  return -1;
}
#define MSAN_MAYBE_INTERCEPT_MALLOPT INTERCEPT_FUNCTION(mallopt)
#else
#define MSAN_MAYBE_INTERCEPT_MALLOPT
#endif

#if !SANITIZER_FREEBSD && !SANITIZER_NETBSD
INTERCEPTOR(void, malloc_stats, void) {
  // FIXME: implement, but don't call REAL(malloc_stats)!
}
#define MSAN_MAYBE_INTERCEPT_MALLOC_STATS INTERCEPT_FUNCTION(malloc_stats)
#else
#define MSAN_MAYBE_INTERCEPT_MALLOC_STATS
#endif

INTERCEPTOR(char *, strcpy, char *dest, const char *src) {  // NOLINT
  ENSURE_MSAN_INITED();
  GET_STORE_STACK_TRACE;
  SIZE_T n = REAL(strlen)(src);
  CHECK_UNPOISONED_STRING(src + n, 0);
  char *res = REAL(strcpy)(dest, src);  // NOLINT
  CopyShadowAndOrigin(dest, src, n + 1, &stack);
  return res;
}

INTERCEPTOR(char *, strncpy, char *dest, const char *src, SIZE_T n) {  // NOLINT
  ENSURE_MSAN_INITED();
  GET_STORE_STACK_TRACE;
  SIZE_T copy_size = REAL(strnlen)(src, n);
  if (copy_size < n)
    copy_size++;  // trailing \0
  char *res = REAL(strncpy)(dest, src, n);  // NOLINT
  CopyShadowAndOrigin(dest, src, copy_size, &stack);
  __msan_unpoison(dest + copy_size, n - copy_size);
  return res;
}

INTERCEPTOR(char *, stpcpy, char *dest, const char *src) {  // NOLINT
  ENSURE_MSAN_INITED();
  GET_STORE_STACK_TRACE;
  SIZE_T n = REAL(strlen)(src);
  CHECK_UNPOISONED_STRING(src + n, 0);
  char *res = REAL(stpcpy)(dest, src);  // NOLINT
  CopyShadowAndOrigin(dest, src, n + 1, &stack);
  return res;
}

INTERCEPTOR(char *, strdup, char *src) {
  ENSURE_MSAN_INITED();
  GET_STORE_STACK_TRACE;
  // On FreeBSD strdup() leverages strlen().
  InterceptorScope interceptor_scope;
  SIZE_T n = REAL(strlen)(src);
  CHECK_UNPOISONED_STRING(src + n, 0);
  char *res = REAL(strdup)(src);
  CopyShadowAndOrigin(res, src, n + 1, &stack);
  return res;
}

#if !SANITIZER_FREEBSD && !SANITIZER_NETBSD
INTERCEPTOR(char *, __strdup, char *src) {
  ENSURE_MSAN_INITED();
  GET_STORE_STACK_TRACE;
  SIZE_T n = REAL(strlen)(src);
  CHECK_UNPOISONED_STRING(src + n, 0);
  char *res = REAL(__strdup)(src);
  CopyShadowAndOrigin(res, src, n + 1, &stack);
  return res;
}
#define MSAN_MAYBE_INTERCEPT___STRDUP INTERCEPT_FUNCTION(__strdup)
#else
#define MSAN_MAYBE_INTERCEPT___STRDUP
#endif

INTERCEPTOR(char *, gcvt, double number, SIZE_T ndigit, char *buf) {
  ENSURE_MSAN_INITED();
  char *res = REAL(gcvt)(number, ndigit, buf);
  SIZE_T n = REAL(strlen)(buf);
  __msan_unpoison(buf, n + 1);
  return res;
}

INTERCEPTOR(char *, strcat, char *dest, const char *src) {  // NOLINT
  ENSURE_MSAN_INITED();
  GET_STORE_STACK_TRACE;
  SIZE_T src_size = REAL(strlen)(src);
  SIZE_T dest_size = REAL(strlen)(dest);
  CHECK_UNPOISONED_STRING(src + src_size, 0);
  CHECK_UNPOISONED_STRING(dest + dest_size, 0);
  char *res = REAL(strcat)(dest, src);  // NOLINT
  CopyShadowAndOrigin(dest + dest_size, src, src_size + 1, &stack);
  return res;
}

INTERCEPTOR(char *, strncat, char *dest, const char *src, SIZE_T n) {  // NOLINT
  ENSURE_MSAN_INITED();
  GET_STORE_STACK_TRACE;
  SIZE_T dest_size = REAL(strlen)(dest);
  SIZE_T copy_size = REAL(strnlen)(src, n);
  CHECK_UNPOISONED_STRING(dest + dest_size, 0);
  char *res = REAL(strncat)(dest, src, n);  // NOLINT
  CopyShadowAndOrigin(dest + dest_size, src, copy_size, &stack);
  __msan_unpoison(dest + dest_size + copy_size, 1); // \0
  return res;
}

// Hack: always pass nptr and endptr as part of __VA_ARGS_ to avoid having to
// deal with empty __VA_ARGS__ in the case of INTERCEPTOR_STRTO.
#define INTERCEPTOR_STRTO_BODY(ret_type, func, ...) \
  ENSURE_MSAN_INITED();                             \
  ret_type res = REAL(func)(__VA_ARGS__);           \
  __msan_unpoison(endptr, sizeof(*endptr));         \
  return res;

#define INTERCEPTOR_STRTO(ret_type, func, char_type)                       \
  INTERCEPTOR(ret_type, func, const char_type *nptr, char_type **endptr) { \
    INTERCEPTOR_STRTO_BODY(ret_type, func, nptr, endptr);                  \
  }

#define INTERCEPTOR_STRTO_BASE(ret_type, func, char_type)                \
  INTERCEPTOR(ret_type, func, const char_type *nptr, char_type **endptr, \
              int base) {                                                \
    INTERCEPTOR_STRTO_BODY(ret_type, func, nptr, endptr, base);          \
  }

#define INTERCEPTOR_STRTO_LOC(ret_type, func, char_type)                 \
  INTERCEPTOR(ret_type, func, const char_type *nptr, char_type **endptr, \
              void *loc) {                                               \
    INTERCEPTOR_STRTO_BODY(ret_type, func, nptr, endptr, loc);           \
  }

#define INTERCEPTOR_STRTO_BASE_LOC(ret_type, func, char_type)            \
  INTERCEPTOR(ret_type, func, const char_type *nptr, char_type **endptr, \
              int base, void *loc) {                                     \
    INTERCEPTOR_STRTO_BODY(ret_type, func, nptr, endptr, base, loc);     \
  }

#define INTERCEPTORS_STRTO(ret_type, func, char_type)      \
  INTERCEPTOR_STRTO(ret_type, func, char_type)             \
  INTERCEPTOR_STRTO_LOC(ret_type, func##_l, char_type)     \
  INTERCEPTOR_STRTO_LOC(ret_type, __##func##_l, char_type) \
  INTERCEPTOR_STRTO_LOC(ret_type, __##func##_internal, char_type)

#define INTERCEPTORS_STRTO_BASE(ret_type, func, char_type)      \
  INTERCEPTOR_STRTO_BASE(ret_type, func, char_type)             \
  INTERCEPTOR_STRTO_BASE_LOC(ret_type, func##_l, char_type)     \
  INTERCEPTOR_STRTO_BASE_LOC(ret_type, __##func##_l, char_type) \
  INTERCEPTOR_STRTO_BASE_LOC(ret_type, __##func##_internal, char_type)

INTERCEPTORS_STRTO(double, strtod, char)                     // NOLINT
INTERCEPTORS_STRTO(float, strtof, char)                      // NOLINT
INTERCEPTORS_STRTO(long double, strtold, char)               // NOLINT
INTERCEPTORS_STRTO_BASE(long, strtol, char)                  // NOLINT
INTERCEPTORS_STRTO_BASE(long long, strtoll, char)            // NOLINT
INTERCEPTORS_STRTO_BASE(unsigned long, strtoul, char)        // NOLINT
INTERCEPTORS_STRTO_BASE(unsigned long long, strtoull, char)  // NOLINT

INTERCEPTORS_STRTO(double, wcstod, wchar_t)                     // NOLINT
INTERCEPTORS_STRTO(float, wcstof, wchar_t)                      // NOLINT
INTERCEPTORS_STRTO(long double, wcstold, wchar_t)               // NOLINT
INTERCEPTORS_STRTO_BASE(long, wcstol, wchar_t)                  // NOLINT
INTERCEPTORS_STRTO_BASE(long long, wcstoll, wchar_t)            // NOLINT
INTERCEPTORS_STRTO_BASE(unsigned long, wcstoul, wchar_t)        // NOLINT
INTERCEPTORS_STRTO_BASE(unsigned long long, wcstoull, wchar_t)  // NOLINT

#define INTERCEPT_STRTO(func) \
  INTERCEPT_FUNCTION(func); \
  INTERCEPT_FUNCTION(func##_l); \
  INTERCEPT_FUNCTION(__##func##_l); \
  INTERCEPT_FUNCTION(__##func##_internal);


// FIXME: support *wprintf in common format interceptors.
INTERCEPTOR(int, vswprintf, void *str, uptr size, void *format, va_list ap) {
  ENSURE_MSAN_INITED();
  int res = REAL(vswprintf)(str, size, format, ap);
  if (res >= 0) {
    __msan_unpoison(str, 4 * (res + 1));
  }
  return res;
}

INTERCEPTOR(int, swprintf, void *str, uptr size, void *format, ...) {
  ENSURE_MSAN_INITED();
  va_list ap;
  va_start(ap, format);
  int res = vswprintf(str, size, format, ap);
  va_end(ap);
  return res;
}

INTERCEPTOR(SIZE_T, strxfrm, char *dest, const char *src, SIZE_T n) {
  ENSURE_MSAN_INITED();
  CHECK_UNPOISONED(src, REAL(strlen)(src) + 1);
  SIZE_T res = REAL(strxfrm)(dest, src, n);
  if (res < n) __msan_unpoison(dest, res + 1);
  return res;
}

INTERCEPTOR(SIZE_T, strxfrm_l, char *dest, const char *src, SIZE_T n,
            void *loc) {
  ENSURE_MSAN_INITED();
  CHECK_UNPOISONED(src, REAL(strlen)(src) + 1);
  SIZE_T res = REAL(strxfrm_l)(dest, src, n, loc);
  if (res < n) __msan_unpoison(dest, res + 1);
  return res;
}

#if SANITIZER_LINUX
INTERCEPTOR(SIZE_T, __strxfrm_l, char *dest, const char *src, SIZE_T n,
            void *loc) {
  ENSURE_MSAN_INITED();
  CHECK_UNPOISONED(src, REAL(strlen)(src) + 1);
  SIZE_T res = REAL(__strxfrm_l)(dest, src, n, loc);
  if (res < n) __msan_unpoison(dest, res + 1);
  return res;
}
#define MSAN_MAYBE_INTERCEPT___STRXFRM_L INTERCEPT_FUNCTION(__strxfrm_l)
#else
#define MSAN_MAYBE_INTERCEPT___STRXFRM_L
#endif

#define INTERCEPTOR_STRFTIME_BODY(char_type, ret_type, func, s, ...) \
  ENSURE_MSAN_INITED();                                              \
  ret_type res = REAL(func)(s, __VA_ARGS__);                         \
  if (s) __msan_unpoison(s, sizeof(char_type) * (res + 1));          \
  return res;

INTERCEPTOR(SIZE_T, strftime, char *s, SIZE_T max, const char *format,
            __sanitizer_tm *tm) {
  INTERCEPTOR_STRFTIME_BODY(char, SIZE_T, strftime, s, max, format, tm);
}

INTERCEPTOR(SIZE_T, strftime_l, char *s, SIZE_T max, const char *format,
            __sanitizer_tm *tm, void *loc) {
  INTERCEPTOR_STRFTIME_BODY(char, SIZE_T, strftime_l, s, max, format, tm, loc);
}

#if !SANITIZER_FREEBSD && !SANITIZER_NETBSD
INTERCEPTOR(SIZE_T, __strftime_l, char *s, SIZE_T max, const char *format,
            __sanitizer_tm *tm, void *loc) {
  INTERCEPTOR_STRFTIME_BODY(char, SIZE_T, __strftime_l, s, max, format, tm,
                            loc);
}
#define MSAN_MAYBE_INTERCEPT___STRFTIME_L INTERCEPT_FUNCTION(__strftime_l)
#else
#define MSAN_MAYBE_INTERCEPT___STRFTIME_L
#endif

INTERCEPTOR(SIZE_T, wcsftime, wchar_t *s, SIZE_T max, const wchar_t *format,
            __sanitizer_tm *tm) {
  INTERCEPTOR_STRFTIME_BODY(wchar_t, SIZE_T, wcsftime, s, max, format, tm);
}

INTERCEPTOR(SIZE_T, wcsftime_l, wchar_t *s, SIZE_T max, const wchar_t *format,
            __sanitizer_tm *tm, void *loc) {
  INTERCEPTOR_STRFTIME_BODY(wchar_t, SIZE_T, wcsftime_l, s, max, format, tm,
                            loc);
}

#if !SANITIZER_FREEBSD && !SANITIZER_NETBSD
INTERCEPTOR(SIZE_T, __wcsftime_l, wchar_t *s, SIZE_T max, const wchar_t *format,
            __sanitizer_tm *tm, void *loc) {
  INTERCEPTOR_STRFTIME_BODY(wchar_t, SIZE_T, __wcsftime_l, s, max, format, tm,
                            loc);
}
#define MSAN_MAYBE_INTERCEPT___WCSFTIME_L INTERCEPT_FUNCTION(__wcsftime_l)
#else
#define MSAN_MAYBE_INTERCEPT___WCSFTIME_L
#endif

INTERCEPTOR(int, mbtowc, wchar_t *dest, const char *src, SIZE_T n) {
  ENSURE_MSAN_INITED();
  int res = REAL(mbtowc)(dest, src, n);
  if (res != -1 && dest) __msan_unpoison(dest, sizeof(wchar_t));
  return res;
}

INTERCEPTOR(int, mbrtowc, wchar_t *dest, const char *src, SIZE_T n, void *ps) {
  ENSURE_MSAN_INITED();
  SIZE_T res = REAL(mbrtowc)(dest, src, n, ps);
  if (res != (SIZE_T)-1 && dest) __msan_unpoison(dest, sizeof(wchar_t));
  return res;
}

// wchar_t *wmemcpy(wchar_t *dest, const wchar_t *src, SIZE_T n);
INTERCEPTOR(wchar_t *, wmemcpy, wchar_t *dest, const wchar_t *src, SIZE_T n) {
  ENSURE_MSAN_INITED();
  GET_STORE_STACK_TRACE;
  wchar_t *res = REAL(wmemcpy)(dest, src, n);
  CopyShadowAndOrigin(dest, src, n * sizeof(wchar_t), &stack);
  return res;
}

INTERCEPTOR(wchar_t *, wmempcpy, wchar_t *dest, const wchar_t *src, SIZE_T n) {
  ENSURE_MSAN_INITED();
  GET_STORE_STACK_TRACE;
  wchar_t *res = REAL(wmempcpy)(dest, src, n);
  CopyShadowAndOrigin(dest, src, n * sizeof(wchar_t), &stack);
  return res;
}

INTERCEPTOR(wchar_t *, wmemset, wchar_t *s, wchar_t c, SIZE_T n) {
  CHECK(MEM_IS_APP(s));
  ENSURE_MSAN_INITED();
  wchar_t *res = REAL(wmemset)(s, c, n);
  __msan_unpoison(s, n * sizeof(wchar_t));
  return res;
}

INTERCEPTOR(wchar_t *, wmemmove, wchar_t *dest, const wchar_t *src, SIZE_T n) {
  ENSURE_MSAN_INITED();
  GET_STORE_STACK_TRACE;
  wchar_t *res = REAL(wmemmove)(dest, src, n);
  MoveShadowAndOrigin(dest, src, n * sizeof(wchar_t), &stack);
  return res;
}

INTERCEPTOR(int, wcscmp, const wchar_t *s1, const wchar_t *s2) {
  ENSURE_MSAN_INITED();
  int res = REAL(wcscmp)(s1, s2);
  return res;
}

INTERCEPTOR(int, gettimeofday, void *tv, void *tz) {
  ENSURE_MSAN_INITED();
  int res = REAL(gettimeofday)(tv, tz);
  if (tv)
    __msan_unpoison(tv, 16);
  if (tz)
    __msan_unpoison(tz, 8);
  return res;
}

INTERCEPTOR(char *, fcvt, double x, int a, int *b, int *c) {
  ENSURE_MSAN_INITED();
  char *res = REAL(fcvt)(x, a, b, c);
  __msan_unpoison(b, sizeof(*b));
  __msan_unpoison(c, sizeof(*c));
  if (res) __msan_unpoison(res, REAL(strlen)(res) + 1);
  return res;
}

INTERCEPTOR(char *, getenv, char *name) {
  if (msan_init_is_running)
    return REAL(getenv)(name);
  ENSURE_MSAN_INITED();
  char *res = REAL(getenv)(name);
  if (res) __msan_unpoison(res, REAL(strlen)(res) + 1);
  return res;
}

extern char **environ;

static void UnpoisonEnviron() {
  char **envp = environ;
  for (; *envp; ++envp) {
    __msan_unpoison(envp, sizeof(*envp));
    __msan_unpoison(*envp, REAL(strlen)(*envp) + 1);
  }
  // Trailing NULL pointer.
  __msan_unpoison(envp, sizeof(*envp));
}

INTERCEPTOR(int, setenv, const char *name, const char *value, int overwrite) {
  ENSURE_MSAN_INITED();
  CHECK_UNPOISONED_STRING(name, 0);
  int res = REAL(setenv)(name, value, overwrite);
  if (!res) UnpoisonEnviron();
  return res;
}

INTERCEPTOR(int, putenv, char *string) {
  ENSURE_MSAN_INITED();
  int res = REAL(putenv)(string);
  if (!res) UnpoisonEnviron();
  return res;
}

#if !SANITIZER_FREEBSD && !SANITIZER_NETBSD
INTERCEPTOR(int, __fxstat, int magic, int fd, void *buf) {
  ENSURE_MSAN_INITED();
  int res = REAL(__fxstat)(magic, fd, buf);
  if (!res)
    __msan_unpoison(buf, __sanitizer::struct_stat_sz);
  return res;
}
#define MSAN_MAYBE_INTERCEPT___FXSTAT INTERCEPT_FUNCTION(__fxstat)
#else
#define MSAN_MAYBE_INTERCEPT___FXSTAT
#endif

#if !SANITIZER_FREEBSD && !SANITIZER_NETBSD
INTERCEPTOR(int, __fxstat64, int magic, int fd, void *buf) {
  ENSURE_MSAN_INITED();
  int res = REAL(__fxstat64)(magic, fd, buf);
  if (!res)
    __msan_unpoison(buf, __sanitizer::struct_stat64_sz);
  return res;
}
#define MSAN_MAYBE_INTERCEPT___FXSTAT64 INTERCEPT_FUNCTION(__fxstat64)
#else
#define MSAN_MAYBE_INTERCEPT___FXSTAT64
#endif

#if SANITIZER_FREEBSD && !SANITIZER_NETBSD
INTERCEPTOR(int, fstatat, int fd, char *pathname, void *buf, int flags) {
  ENSURE_MSAN_INITED();
  int res = REAL(fstatat)(fd, pathname, buf, flags);
  if (!res) __msan_unpoison(buf, __sanitizer::struct_stat_sz);
  return res;
}
# define MSAN_INTERCEPT_FSTATAT INTERCEPT_FUNCTION(fstatat)
#else
INTERCEPTOR(int, __fxstatat, int magic, int fd, char *pathname, void *buf,
            int flags) {
  ENSURE_MSAN_INITED();
  int res = REAL(__fxstatat)(magic, fd, pathname, buf, flags);
  if (!res) __msan_unpoison(buf, __sanitizer::struct_stat_sz);
  return res;
}
# define MSAN_INTERCEPT_FSTATAT INTERCEPT_FUNCTION(__fxstatat)
#endif

#if !SANITIZER_FREEBSD && !SANITIZER_NETBSD
INTERCEPTOR(int, __fxstatat64, int magic, int fd, char *pathname, void *buf,
            int flags) {
  ENSURE_MSAN_INITED();
  int res = REAL(__fxstatat64)(magic, fd, pathname, buf, flags);
  if (!res) __msan_unpoison(buf, __sanitizer::struct_stat64_sz);
  return res;
}
#define MSAN_MAYBE_INTERCEPT___FXSTATAT64 INTERCEPT_FUNCTION(__fxstatat64)
#else
#define MSAN_MAYBE_INTERCEPT___FXSTATAT64
#endif

INTERCEPTOR(int, pipe, int pipefd[2]) {
  if (msan_init_is_running)
    return REAL(pipe)(pipefd);
  ENSURE_MSAN_INITED();
  int res = REAL(pipe)(pipefd);
  if (!res)
    __msan_unpoison(pipefd, sizeof(int[2]));
  return res;
}

INTERCEPTOR(int, pipe2, int pipefd[2], int flags) {
  ENSURE_MSAN_INITED();
  int res = REAL(pipe2)(pipefd, flags);
  if (!res)
    __msan_unpoison(pipefd, sizeof(int[2]));
  return res;
}

INTERCEPTOR(int, socketpair, int domain, int type, int protocol, int sv[2]) {
  ENSURE_MSAN_INITED();
  int res = REAL(socketpair)(domain, type, protocol, sv);
  if (!res)
    __msan_unpoison(sv, sizeof(int[2]));
  return res;
}

INTERCEPTOR(char *, fgets, char *s, int size, void *stream) {
  ENSURE_MSAN_INITED();
  char *res = REAL(fgets)(s, size, stream);
  if (res)
    __msan_unpoison(s, REAL(strlen)(s) + 1);
  return res;
}

#if !SANITIZER_FREEBSD && !SANITIZER_NETBSD
INTERCEPTOR(char *, fgets_unlocked, char *s, int size, void *stream) {
  ENSURE_MSAN_INITED();
  char *res = REAL(fgets_unlocked)(s, size, stream);
  if (res)
    __msan_unpoison(s, REAL(strlen)(s) + 1);
  return res;
}
#define MSAN_MAYBE_INTERCEPT_FGETS_UNLOCKED INTERCEPT_FUNCTION(fgets_unlocked)
#else
#define MSAN_MAYBE_INTERCEPT_FGETS_UNLOCKED
#endif

INTERCEPTOR(int, getrlimit, int resource, void *rlim) {
  if (msan_init_is_running)
    return REAL(getrlimit)(resource, rlim);
  ENSURE_MSAN_INITED();
  int res = REAL(getrlimit)(resource, rlim);
  if (!res)
    __msan_unpoison(rlim, __sanitizer::struct_rlimit_sz);
  return res;
}

#if !SANITIZER_FREEBSD && !SANITIZER_NETBSD
INTERCEPTOR(int, getrlimit64, int resource, void *rlim) {
  if (msan_init_is_running) return REAL(getrlimit64)(resource, rlim);
  ENSURE_MSAN_INITED();
  int res = REAL(getrlimit64)(resource, rlim);
  if (!res) __msan_unpoison(rlim, __sanitizer::struct_rlimit64_sz);
  return res;
}

INTERCEPTOR(int, prlimit, int pid, int resource, void *new_rlimit,
            void *old_rlimit) {
  if (msan_init_is_running)
    return REAL(prlimit)(pid, resource, new_rlimit, old_rlimit);
  ENSURE_MSAN_INITED();
  CHECK_UNPOISONED(new_rlimit, __sanitizer::struct_rlimit_sz);
  int res = REAL(prlimit)(pid, resource, new_rlimit, old_rlimit);
  if (!res) __msan_unpoison(old_rlimit, __sanitizer::struct_rlimit_sz);
  return res;
}

INTERCEPTOR(int, prlimit64, int pid, int resource, void *new_rlimit,
            void *old_rlimit) {
  if (msan_init_is_running)
    return REAL(prlimit64)(pid, resource, new_rlimit, old_rlimit);
  ENSURE_MSAN_INITED();
  CHECK_UNPOISONED(new_rlimit, __sanitizer::struct_rlimit64_sz);
  int res = REAL(prlimit64)(pid, resource, new_rlimit, old_rlimit);
  if (!res) __msan_unpoison(old_rlimit, __sanitizer::struct_rlimit64_sz);
  return res;
}

#define MSAN_MAYBE_INTERCEPT_GETRLIMIT64 INTERCEPT_FUNCTION(getrlimit64)
#define MSAN_MAYBE_INTERCEPT_PRLIMIT INTERCEPT_FUNCTION(prlimit)
#define MSAN_MAYBE_INTERCEPT_PRLIMIT64 INTERCEPT_FUNCTION(prlimit64)
#else
#define MSAN_MAYBE_INTERCEPT_GETRLIMIT64
#define MSAN_MAYBE_INTERCEPT_PRLIMIT
#define MSAN_MAYBE_INTERCEPT_PRLIMIT64
#endif

#if SANITIZER_FREEBSD
// FreeBSD's <sys/utsname.h> define uname() as
// static __inline int uname(struct utsname *name) {
//   return __xuname(SYS_NMLN, (void*)name);
// }
INTERCEPTOR(int, __xuname, int size, void *utsname) {
  ENSURE_MSAN_INITED();
  int res = REAL(__xuname)(size, utsname);
  if (!res)
    __msan_unpoison(utsname, __sanitizer::struct_utsname_sz);
  return res;
}
#define MSAN_INTERCEPT_UNAME INTERCEPT_FUNCTION(__xuname)
#else
INTERCEPTOR(int, uname, struct utsname *utsname) {
  ENSURE_MSAN_INITED();
  int res = REAL(uname)(utsname);
  if (!res)
    __msan_unpoison(utsname, __sanitizer::struct_utsname_sz);
  return res;
}
#define MSAN_INTERCEPT_UNAME INTERCEPT_FUNCTION(uname)
#endif

INTERCEPTOR(int, gethostname, char *name, SIZE_T len) {
  ENSURE_MSAN_INITED();
  int res = REAL(gethostname)(name, len);
  if (!res) {
    SIZE_T real_len = REAL(strnlen)(name, len);
    if (real_len < len)
      ++real_len;
    __msan_unpoison(name, real_len);
  }
  return res;
}

#if !SANITIZER_FREEBSD && !SANITIZER_NETBSD
INTERCEPTOR(int, epoll_wait, int epfd, void *events, int maxevents,
    int timeout) {
  ENSURE_MSAN_INITED();
  int res = REAL(epoll_wait)(epfd, events, maxevents, timeout);
  if (res > 0) {
    __msan_unpoison(events, __sanitizer::struct_epoll_event_sz * res);
  }
  return res;
}
#define MSAN_MAYBE_INTERCEPT_EPOLL_WAIT INTERCEPT_FUNCTION(epoll_wait)
#else
#define MSAN_MAYBE_INTERCEPT_EPOLL_WAIT
#endif

#if !SANITIZER_FREEBSD && !SANITIZER_NETBSD
INTERCEPTOR(int, epoll_pwait, int epfd, void *events, int maxevents,
    int timeout, void *sigmask) {
  ENSURE_MSAN_INITED();
  int res = REAL(epoll_pwait)(epfd, events, maxevents, timeout, sigmask);
  if (res > 0) {
    __msan_unpoison(events, __sanitizer::struct_epoll_event_sz * res);
  }
  return res;
}
#define MSAN_MAYBE_INTERCEPT_EPOLL_PWAIT INTERCEPT_FUNCTION(epoll_pwait)
#else
#define MSAN_MAYBE_INTERCEPT_EPOLL_PWAIT
#endif

INTERCEPTOR(void *, calloc, SIZE_T nmemb, SIZE_T size) {
  GET_MALLOC_STACK_TRACE;
  if (UNLIKELY(!msan_inited))
    // Hack: dlsym calls calloc before REAL(calloc) is retrieved from dlsym.
    return AllocateFromLocalPool(nmemb * size);
  return msan_calloc(nmemb, size, &stack);
}

INTERCEPTOR(void *, realloc, void *ptr, SIZE_T size) {
  GET_MALLOC_STACK_TRACE;
  if (UNLIKELY(IsInDlsymAllocPool(ptr))) {
    uptr offset = (uptr)ptr - (uptr)alloc_memory_for_dlsym;
    uptr copy_size = Min(size, kDlsymAllocPoolSize - offset);
    void *new_ptr;
    if (UNLIKELY(!msan_inited)) {
      new_ptr = AllocateFromLocalPool(copy_size);
    } else {
      copy_size = size;
      new_ptr = msan_malloc(copy_size, &stack);
    }
    internal_memcpy(new_ptr, ptr, copy_size);
    return new_ptr;
  }
  return msan_realloc(ptr, size, &stack);
}

INTERCEPTOR(void *, malloc, SIZE_T size) {
  GET_MALLOC_STACK_TRACE;
  if (UNLIKELY(!msan_inited))
    // Hack: dlsym calls malloc before REAL(malloc) is retrieved from dlsym.
    return AllocateFromLocalPool(size);
  return msan_malloc(size, &stack);
}

void __msan_allocated_memory(const void *data, uptr size) {
  GET_MALLOC_STACK_TRACE;
  if (flags()->poison_in_malloc) {
    stack.tag = STACK_TRACE_TAG_POISON;
    PoisonMemory(data, size, &stack);
  }
}

void __msan_copy_shadow(void *dest, const void *src, uptr n) {
  GET_STORE_STACK_TRACE;
  MoveShadowAndOrigin(dest, src, n, &stack);
}

void __sanitizer_dtor_callback(const void *data, uptr size) {
  GET_MALLOC_STACK_TRACE;
  if (flags()->poison_in_dtor) {
    stack.tag = STACK_TRACE_TAG_POISON;
    PoisonMemory(data, size, &stack);
  }
}

INTERCEPTOR(void *, mmap, void *addr, SIZE_T length, int prot, int flags,
            int fd, OFF_T offset) {
  if (msan_init_is_running)
    return REAL(mmap)(addr, length, prot, flags, fd, offset);
  ENSURE_MSAN_INITED();
  if (addr && !MEM_IS_APP(addr)) {
    if (flags & map_fixed) {
      errno = errno_EINVAL;
      return (void *)-1;
    } else {
      addr = nullptr;
    }
  }
  void *res = REAL(mmap)(addr, length, prot, flags, fd, offset);
  if (res != (void*)-1)
    __msan_unpoison(res, RoundUpTo(length, GetPageSize()));
  return res;
}

#if !SANITIZER_FREEBSD && !SANITIZER_NETBSD
INTERCEPTOR(void *, mmap64, void *addr, SIZE_T length, int prot, int flags,
            int fd, OFF64_T offset) {
  ENSURE_MSAN_INITED();
  if (addr && !MEM_IS_APP(addr)) {
    if (flags & map_fixed) {
      errno = errno_EINVAL;
      return (void *)-1;
    } else {
      addr = nullptr;
    }
  }
  void *res = REAL(mmap64)(addr, length, prot, flags, fd, offset);
  if (res != (void*)-1)
    __msan_unpoison(res, RoundUpTo(length, GetPageSize()));
  return res;
}
#define MSAN_MAYBE_INTERCEPT_MMAP64 INTERCEPT_FUNCTION(mmap64)
#else
#define MSAN_MAYBE_INTERCEPT_MMAP64
#endif

INTERCEPTOR(int, getrusage, int who, void *usage) {
  ENSURE_MSAN_INITED();
  int res = REAL(getrusage)(who, usage);
  if (res == 0) {
    __msan_unpoison(usage, __sanitizer::struct_rusage_sz);
  }
  return res;
}

class SignalHandlerScope {
 public:
  SignalHandlerScope() {
    if (MsanThread *t = GetCurrentThread())
      t->EnterSignalHandler();
  }
  ~SignalHandlerScope() {
    if (MsanThread *t = GetCurrentThread())
      t->LeaveSignalHandler();
  }
};

// sigactions_mu guarantees atomicity of sigaction() and signal() calls.
// Access to sigactions[] is gone with relaxed atomics to avoid data race with
// the signal handler.
const int kMaxSignals = 1024;
static atomic_uintptr_t sigactions[kMaxSignals];
static StaticSpinMutex sigactions_mu;

static void SignalHandler(int signo) {
  SignalHandlerScope signal_handler_scope;
  ScopedThreadLocalStateBackup stlsb;
  UnpoisonParam(1);

  typedef void (*signal_cb)(int x);
  signal_cb cb =
      (signal_cb)atomic_load(&sigactions[signo], memory_order_relaxed);
  cb(signo);
}

static void SignalAction(int signo, void *si, void *uc) {
  SignalHandlerScope signal_handler_scope;
  ScopedThreadLocalStateBackup stlsb;
  UnpoisonParam(3);
  __msan_unpoison(si, sizeof(__sanitizer_sigaction));
  __msan_unpoison(uc, __sanitizer::ucontext_t_sz);

  typedef void (*sigaction_cb)(int, void *, void *);
  sigaction_cb cb =
      (sigaction_cb)atomic_load(&sigactions[signo], memory_order_relaxed);
  cb(signo, si, uc);
}

static void read_sigaction(const __sanitizer_sigaction *act) {
  CHECK_UNPOISONED(&act->sa_flags, sizeof(act->sa_flags));
  if (act->sa_flags & __sanitizer::sa_siginfo)
    CHECK_UNPOISONED(&act->sigaction, sizeof(act->sigaction));
  else
    CHECK_UNPOISONED(&act->handler, sizeof(act->handler));
  CHECK_UNPOISONED(&act->sa_mask, sizeof(act->sa_mask));
}

extern "C" int pthread_attr_init(void *attr);
extern "C" int pthread_attr_destroy(void *attr);

static void *MsanThreadStartFunc(void *arg) {
  MsanThread *t = (MsanThread *)arg;
  SetCurrentThread(t);
  return t->ThreadStart();
}

INTERCEPTOR(int, pthread_create, void *th, void *attr, void *(*callback)(void*),
            void * param) {
  ENSURE_MSAN_INITED(); // for GetTlsSize()
  __sanitizer_pthread_attr_t myattr;
  if (!attr) {
    pthread_attr_init(&myattr);
    attr = &myattr;
  }

  AdjustStackSize(attr);

  MsanThread *t = MsanThread::Create(callback, param);

  int res = REAL(pthread_create)(th, attr, MsanThreadStartFunc, t);

  if (attr == &myattr)
    pthread_attr_destroy(&myattr);
  if (!res) {
    __msan_unpoison(th, __sanitizer::pthread_t_sz);
  }
  return res;
}

INTERCEPTOR(int, pthread_key_create, __sanitizer_pthread_key_t *key,
            void (*dtor)(void *value)) {
  if (msan_init_is_running) return REAL(pthread_key_create)(key, dtor);
  ENSURE_MSAN_INITED();
  int res = REAL(pthread_key_create)(key, dtor);
  if (!res && key)
    __msan_unpoison(key, sizeof(*key));
  return res;
}

INTERCEPTOR(int, pthread_join, void *th, void **retval) {
  ENSURE_MSAN_INITED();
  int res = REAL(pthread_join)(th, retval);
  if (!res && retval)
    __msan_unpoison(retval, sizeof(*retval));
  return res;
}

extern char *tzname[2];

INTERCEPTOR(void, tzset, int fake) {
  ENSURE_MSAN_INITED();
  REAL(tzset)(fake);
  if (tzname[0])
    __msan_unpoison(tzname[0], REAL(strlen)(tzname[0]) + 1);
  if (tzname[1])
    __msan_unpoison(tzname[1], REAL(strlen)(tzname[1]) + 1);
  return;
}

struct MSanAtExitRecord {
  void (*func)(void *arg);
  void *arg;
};

void MSanAtExitWrapper(void *arg) {
  UnpoisonParam(1);
  MSanAtExitRecord *r = (MSanAtExitRecord *)arg;
  r->func(r->arg);
  InternalFree(r);
}

// Unpoison argument shadow for C++ module destructors.
INTERCEPTOR(int, __cxa_atexit, void (*func)(void *), void *arg,
            void *dso_handle) {
  if (msan_init_is_running) return REAL(__cxa_atexit)(func, arg, dso_handle);
  ENSURE_MSAN_INITED();
  MSanAtExitRecord *r =
      (MSanAtExitRecord *)InternalAlloc(sizeof(MSanAtExitRecord));
  r->func = func;
  r->arg = arg;
  return REAL(__cxa_atexit)(MSanAtExitWrapper, r, dso_handle);
}

DECLARE_REAL(int, shmctl, int shmid, int cmd, void *buf)

INTERCEPTOR(void *, shmat, int shmid, const void *shmaddr, int shmflg) {
  ENSURE_MSAN_INITED();
  void *p = REAL(shmat)(shmid, shmaddr, shmflg);
  if (p != (void *)-1) {
    __sanitizer_shmid_ds ds;
    int res = REAL(shmctl)(shmid, shmctl_ipc_stat, &ds);
    if (!res) {
      __msan_unpoison(p, ds.shm_segsz);
    }
  }
  return p;
}

static void BeforeFork() {
  StackDepotLockAll();
  ChainedOriginDepotLockAll();
}

static void AfterFork() {
  ChainedOriginDepotUnlockAll();
  StackDepotUnlockAll();
}

INTERCEPTOR(int, fork, void) {
  ENSURE_MSAN_INITED();
  BeforeFork();
  int pid = REAL(fork)();
  AfterFork();
  return pid;
}

INTERCEPTOR(int, openpty, int *amaster, int *aslave, char *name,
            const void *termp, const void *winp) {
  ENSURE_MSAN_INITED();
  InterceptorScope interceptor_scope;
  int res = REAL(openpty)(amaster, aslave, name, termp, winp);
  if (!res) {
    __msan_unpoison(amaster, sizeof(*amaster));
    __msan_unpoison(aslave, sizeof(*aslave));
  }
  return res;
}

INTERCEPTOR(int, forkpty, int *amaster, char *name, const void *termp,
            const void *winp) {
  ENSURE_MSAN_INITED();
  InterceptorScope interceptor_scope;
  int res = REAL(forkpty)(amaster, name, termp, winp);
  if (res != -1)
    __msan_unpoison(amaster, sizeof(*amaster));
  return res;
}

struct MSanInterceptorContext {
  bool in_interceptor_scope;
};

namespace __msan {

int OnExit() {
  // FIXME: ask frontend whether we need to return failure.
  return 0;
}

} // namespace __msan

// A version of CHECK_UNPOISONED using a saved scope value. Used in common
// interceptors.
#define CHECK_UNPOISONED_CTX(ctx, x, n)                         \
  do {                                                          \
    if (!((MSanInterceptorContext *)ctx)->in_interceptor_scope) \
      CHECK_UNPOISONED_0(x, n);                                 \
  } while (0)

#define MSAN_INTERCEPT_FUNC(name)                                       \
  do {                                                                  \
    if ((!INTERCEPT_FUNCTION(name) || !REAL(name)))                     \
      VReport(1, "MemorySanitizer: failed to intercept '" #name "'\n"); \
  } while (0)

#define MSAN_INTERCEPT_FUNC_VER(name, ver)                                    \
  do {                                                                        \
    if ((!INTERCEPT_FUNCTION_VER(name, ver) || !REAL(name)))                  \
      VReport(                                                                \
          1, "MemorySanitizer: failed to intercept '" #name "@@" #ver "'\n"); \
  } while (0)

#define COMMON_INTERCEPT_FUNCTION(name) MSAN_INTERCEPT_FUNC(name)
#define COMMON_INTERCEPT_FUNCTION_VER(name, ver)                          \
  MSAN_INTERCEPT_FUNC_VER(name, ver)
#define COMMON_INTERCEPTOR_UNPOISON_PARAM(count)  \
  UnpoisonParam(count)
#define COMMON_INTERCEPTOR_WRITE_RANGE(ctx, ptr, size) \
  __msan_unpoison(ptr, size)
#define COMMON_INTERCEPTOR_READ_RANGE(ctx, ptr, size) \
  CHECK_UNPOISONED_CTX(ctx, ptr, size)
#define COMMON_INTERCEPTOR_INITIALIZE_RANGE(ptr, size) \
  __msan_unpoison(ptr, size)
#define COMMON_INTERCEPTOR_ENTER(ctx, func, ...)                  \
  if (msan_init_is_running) return REAL(func)(__VA_ARGS__);       \
  ENSURE_MSAN_INITED();                                           \
  MSanInterceptorContext msan_ctx = {IsInInterceptorScope()};     \
  ctx = (void *)&msan_ctx;                                        \
  (void)ctx;                                                      \
  InterceptorScope interceptor_scope;                             \
  __msan_unpoison(__errno_location(), sizeof(int)); /* NOLINT */
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
#define COMMON_INTERCEPTOR_LIBRARY_LOADED(filename, handle)                    \
  do {                                                                         \
    link_map *map = GET_LINK_MAP_BY_DLOPEN_HANDLE((handle));                   \
    if (filename && map)                                                       \
      ForEachMappedRegion(map, __msan_unpoison);                               \
  } while (false)

#define COMMON_INTERCEPTOR_GET_TLS_RANGE(begin, end)                           \
  if (MsanThread *t = GetCurrentThread()) {                                    \
    *begin = t->tls_begin();                                                   \
    *end = t->tls_end();                                                       \
  } else {                                                                     \
    *begin = *end = 0;                                                         \
  }

#define COMMON_INTERCEPTOR_MEMSET_IMPL(ctx, block, c, size) \
  {                                                         \
    (void)ctx;                                              \
    return __msan_memset(block, c, size);                   \
  }
#define COMMON_INTERCEPTOR_MEMMOVE_IMPL(ctx, to, from, size) \
  {                                                          \
    (void)ctx;                                               \
    return __msan_memmove(to, from, size);                   \
  }
#define COMMON_INTERCEPTOR_MEMCPY_IMPL(ctx, to, from, size) \
  {                                                         \
    (void)ctx;                                              \
    return __msan_memcpy(to, from, size);                   \
  }

#define COMMON_INTERCEPTOR_COPY_STRING(ctx, to, from, size) \
  do {                                                      \
    GET_STORE_STACK_TRACE;                                  \
    CopyShadowAndOrigin(to, from, size, &stack);            \
    __msan_unpoison(to + size, 1);                          \
  } while (false)

#include "sanitizer_common/sanitizer_platform_interceptors.h"
#include "sanitizer_common/sanitizer_common_interceptors.inc"

static uptr signal_impl(int signo, uptr cb);
static int sigaction_impl(int signo, const __sanitizer_sigaction *act,
                          __sanitizer_sigaction *oldact);

#define SIGNAL_INTERCEPTOR_SIGACTION_IMPL(signo, act, oldact) \
  { return sigaction_impl(signo, act, oldact); }

#define SIGNAL_INTERCEPTOR_SIGNAL_IMPL(func, signo, handler) \
  {                                                          \
    handler = signal_impl(signo, handler);                   \
    return REAL(func)(signo, handler);                       \
  }

#include "sanitizer_common/sanitizer_signal_interceptors.inc"

static int sigaction_impl(int signo, const __sanitizer_sigaction *act,
                          __sanitizer_sigaction *oldact) {
  ENSURE_MSAN_INITED();
  if (act) read_sigaction(act);
  int res;
  if (flags()->wrap_signals) {
    SpinMutexLock lock(&sigactions_mu);
    CHECK_LT(signo, kMaxSignals);
    uptr old_cb = atomic_load(&sigactions[signo], memory_order_relaxed);
    __sanitizer_sigaction new_act;
    __sanitizer_sigaction *pnew_act = act ? &new_act : nullptr;
    if (act) {
      REAL(memcpy)(pnew_act, act, sizeof(__sanitizer_sigaction));
      uptr cb = (uptr)pnew_act->sigaction;
      uptr new_cb = (pnew_act->sa_flags & __sanitizer::sa_siginfo)
                        ? (uptr)SignalAction
                        : (uptr)SignalHandler;
      if (cb != __sanitizer::sig_ign && cb != __sanitizer::sig_dfl) {
        atomic_store(&sigactions[signo], cb, memory_order_relaxed);
        pnew_act->sigaction = (decltype(pnew_act->sigaction))new_cb;
      }
    }
    res = REAL(sigaction)(signo, pnew_act, oldact);
    if (res == 0 && oldact) {
      uptr cb = (uptr)oldact->sigaction;
      if (cb != __sanitizer::sig_ign && cb != __sanitizer::sig_dfl) {
        oldact->sigaction = (decltype(oldact->sigaction))old_cb;
      }
    }
  } else {
    res = REAL(sigaction)(signo, act, oldact);
  }

  if (res == 0 && oldact) {
    __msan_unpoison(oldact, sizeof(__sanitizer_sigaction));
  }
  return res;
}

static uptr signal_impl(int signo, uptr cb) {
  ENSURE_MSAN_INITED();
  if (flags()->wrap_signals) {
    CHECK_LT(signo, kMaxSignals);
    SpinMutexLock lock(&sigactions_mu);
    if (cb != __sanitizer::sig_ign && cb != __sanitizer::sig_dfl) {
      atomic_store(&sigactions[signo], cb, memory_order_relaxed);
      cb = (uptr)&SignalHandler;
    }
  }
  return cb;
}

#define COMMON_SYSCALL_PRE_READ_RANGE(p, s) CHECK_UNPOISONED(p, s)
#define COMMON_SYSCALL_PRE_WRITE_RANGE(p, s) \
  do {                                       \
  } while (false)
#define COMMON_SYSCALL_POST_READ_RANGE(p, s) \
  do {                                       \
  } while (false)
#define COMMON_SYSCALL_POST_WRITE_RANGE(p, s) __msan_unpoison(p, s)
#include "sanitizer_common/sanitizer_common_syscalls.inc"

struct dlinfo {
  char *dli_fname;
  void *dli_fbase;
  char *dli_sname;
  void *dli_saddr;
};

INTERCEPTOR(int, dladdr, void *addr, dlinfo *info) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, dladdr, addr, info);
  int res = REAL(dladdr)(addr, info);
  if (res != 0) {
    __msan_unpoison(info, sizeof(*info));
    if (info->dli_fname)
      __msan_unpoison(info->dli_fname, REAL(strlen)(info->dli_fname) + 1);
    if (info->dli_sname)
      __msan_unpoison(info->dli_sname, REAL(strlen)(info->dli_sname) + 1);
  }
  return res;
}

INTERCEPTOR(char *, dlerror, int fake) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, dlerror, fake);
  char *res = REAL(dlerror)(fake);
  if (res) __msan_unpoison(res, REAL(strlen)(res) + 1);
  return res;
}

typedef int (*dl_iterate_phdr_cb)(__sanitizer_dl_phdr_info *info, SIZE_T size,
                                  void *data);
struct dl_iterate_phdr_data {
  dl_iterate_phdr_cb callback;
  void *data;
};

static int msan_dl_iterate_phdr_cb(__sanitizer_dl_phdr_info *info, SIZE_T size,
                                   void *data) {
  if (info) {
    __msan_unpoison(info, size);
    if (info->dlpi_phdr && info->dlpi_phnum)
      __msan_unpoison(info->dlpi_phdr, struct_ElfW_Phdr_sz * info->dlpi_phnum);
    if (info->dlpi_name)
      __msan_unpoison(info->dlpi_name, REAL(strlen)(info->dlpi_name) + 1);
  }
  dl_iterate_phdr_data *cbdata = (dl_iterate_phdr_data *)data;
  UnpoisonParam(3);
  return cbdata->callback(info, size, cbdata->data);
}

INTERCEPTOR(int, dl_iterate_phdr, dl_iterate_phdr_cb callback, void *data) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, dl_iterate_phdr, callback, data);
  dl_iterate_phdr_data cbdata;
  cbdata.callback = callback;
  cbdata.data = data;
  int res = REAL(dl_iterate_phdr)(msan_dl_iterate_phdr_cb, (void *)&cbdata);
  return res;
}

// wchar_t *wcschr(const wchar_t *wcs, wchar_t wc);
INTERCEPTOR(wchar_t *, wcschr, void *s, wchar_t wc, void *ps) {
  ENSURE_MSAN_INITED();
  wchar_t *res = REAL(wcschr)(s, wc, ps);
  return res;
}

// wchar_t *wcscpy(wchar_t *dest, const wchar_t *src);
INTERCEPTOR(wchar_t *, wcscpy, wchar_t *dest, const wchar_t *src) {
  ENSURE_MSAN_INITED();
  GET_STORE_STACK_TRACE;
  wchar_t *res = REAL(wcscpy)(dest, src);
  CopyShadowAndOrigin(dest, src, sizeof(wchar_t) * (REAL(wcslen)(src) + 1),
                      &stack);
  return res;
}

INTERCEPTOR(wchar_t *, wcsncpy, wchar_t *dest, const wchar_t *src,
            SIZE_T n) {  // NOLINT
  ENSURE_MSAN_INITED();
  GET_STORE_STACK_TRACE;
  SIZE_T copy_size = REAL(wcsnlen)(src, n);
  if (copy_size < n) copy_size++;           // trailing \0
  wchar_t *res = REAL(wcsncpy)(dest, src, n);  // NOLINT
  CopyShadowAndOrigin(dest, src, copy_size * sizeof(wchar_t), &stack);
  __msan_unpoison(dest + copy_size, (n - copy_size) * sizeof(wchar_t));
  return res;
}

// These interface functions reside here so that they can use
// REAL(memset), etc.
void __msan_unpoison(const void *a, uptr size) {
  if (!MEM_IS_APP(a)) return;
  SetShadow(a, size, 0);
}

void __msan_poison(const void *a, uptr size) {
  if (!MEM_IS_APP(a)) return;
  SetShadow(a, size, __msan::flags()->poison_heap_with_zeroes ? 0 : -1);
}

void __msan_poison_stack(void *a, uptr size) {
  if (!MEM_IS_APP(a)) return;
  SetShadow(a, size, __msan::flags()->poison_stack_with_zeroes ? 0 : -1);
}

void __msan_clear_and_unpoison(void *a, uptr size) {
  REAL(memset)(a, 0, size);
  SetShadow(a, size, 0);
}

void *__msan_memcpy(void *dest, const void *src, SIZE_T n) {
  if (!msan_inited) return internal_memcpy(dest, src, n);
  if (msan_init_is_running || __msan::IsInSymbolizer())
    return REAL(memcpy)(dest, src, n);
  ENSURE_MSAN_INITED();
  GET_STORE_STACK_TRACE;
  void *res = REAL(memcpy)(dest, src, n);
  CopyShadowAndOrigin(dest, src, n, &stack);
  return res;
}

void *__msan_memset(void *s, int c, SIZE_T n) {
  if (!msan_inited) return internal_memset(s, c, n);
  if (msan_init_is_running) return REAL(memset)(s, c, n);
  ENSURE_MSAN_INITED();
  void *res = REAL(memset)(s, c, n);
  __msan_unpoison(s, n);
  return res;
}

void *__msan_memmove(void *dest, const void *src, SIZE_T n) {
  if (!msan_inited) return internal_memmove(dest, src, n);
  if (msan_init_is_running) return REAL(memmove)(dest, src, n);
  ENSURE_MSAN_INITED();
  GET_STORE_STACK_TRACE;
  void *res = REAL(memmove)(dest, src, n);
  MoveShadowAndOrigin(dest, src, n, &stack);
  return res;
}

void __msan_unpoison_string(const char* s) {
  if (!MEM_IS_APP(s)) return;
  __msan_unpoison(s, REAL(strlen)(s) + 1);
}

namespace __msan {

void InitializeInterceptors() {
  static int inited = 0;
  CHECK_EQ(inited, 0);
  InitializeCommonInterceptors();
  InitializeSignalInterceptors();

  INTERCEPT_FUNCTION(mmap);
  MSAN_MAYBE_INTERCEPT_MMAP64;
  INTERCEPT_FUNCTION(posix_memalign);
  MSAN_MAYBE_INTERCEPT_MEMALIGN;
  INTERCEPT_FUNCTION(__libc_memalign);
  INTERCEPT_FUNCTION(valloc);
  MSAN_MAYBE_INTERCEPT_PVALLOC;
  INTERCEPT_FUNCTION(malloc);
  INTERCEPT_FUNCTION(calloc);
  INTERCEPT_FUNCTION(realloc);
  INTERCEPT_FUNCTION(free);
  MSAN_MAYBE_INTERCEPT_CFREE;
  INTERCEPT_FUNCTION(malloc_usable_size);
  MSAN_MAYBE_INTERCEPT_MALLINFO;
  MSAN_MAYBE_INTERCEPT_MALLOPT;
  MSAN_MAYBE_INTERCEPT_MALLOC_STATS;
  INTERCEPT_FUNCTION(fread);
  MSAN_MAYBE_INTERCEPT_FREAD_UNLOCKED;
  INTERCEPT_FUNCTION(readlink);
  INTERCEPT_FUNCTION(memccpy);
  INTERCEPT_FUNCTION(mempcpy);
  INTERCEPT_FUNCTION(bcopy);
  INTERCEPT_FUNCTION(wmemset);
  INTERCEPT_FUNCTION(wmemcpy);
  INTERCEPT_FUNCTION(wmempcpy);
  INTERCEPT_FUNCTION(wmemmove);
  INTERCEPT_FUNCTION(strcpy);  // NOLINT
  INTERCEPT_FUNCTION(stpcpy);  // NOLINT
  INTERCEPT_FUNCTION(strdup);
  MSAN_MAYBE_INTERCEPT___STRDUP;
  INTERCEPT_FUNCTION(strncpy);  // NOLINT
  INTERCEPT_FUNCTION(gcvt);
  INTERCEPT_FUNCTION(strcat);  // NOLINT
  INTERCEPT_FUNCTION(strncat);  // NOLINT
  INTERCEPT_STRTO(strtod);
  INTERCEPT_STRTO(strtof);
  INTERCEPT_STRTO(strtold);
  INTERCEPT_STRTO(strtol);
  INTERCEPT_STRTO(strtoul);
  INTERCEPT_STRTO(strtoll);
  INTERCEPT_STRTO(strtoull);
  INTERCEPT_STRTO(wcstod);
  INTERCEPT_STRTO(wcstof);
  INTERCEPT_STRTO(wcstold);
  INTERCEPT_STRTO(wcstol);
  INTERCEPT_STRTO(wcstoul);
  INTERCEPT_STRTO(wcstoll);
  INTERCEPT_STRTO(wcstoull);
#ifdef SANITIZER_NLDBL_VERSION
  INTERCEPT_FUNCTION_VER(vswprintf, SANITIZER_NLDBL_VERSION);
  INTERCEPT_FUNCTION_VER(swprintf, SANITIZER_NLDBL_VERSION);
#else
  INTERCEPT_FUNCTION(vswprintf);
  INTERCEPT_FUNCTION(swprintf);
#endif
  INTERCEPT_FUNCTION(strxfrm);
  INTERCEPT_FUNCTION(strxfrm_l);
  MSAN_MAYBE_INTERCEPT___STRXFRM_L;
  INTERCEPT_FUNCTION(strftime);
  INTERCEPT_FUNCTION(strftime_l);
  MSAN_MAYBE_INTERCEPT___STRFTIME_L;
  INTERCEPT_FUNCTION(wcsftime);
  INTERCEPT_FUNCTION(wcsftime_l);
  MSAN_MAYBE_INTERCEPT___WCSFTIME_L;
  INTERCEPT_FUNCTION(mbtowc);
  INTERCEPT_FUNCTION(mbrtowc);
  INTERCEPT_FUNCTION(wcslen);
  INTERCEPT_FUNCTION(wcsnlen);
  INTERCEPT_FUNCTION(wcschr);
  INTERCEPT_FUNCTION(wcscpy);
  INTERCEPT_FUNCTION(wcsncpy);
  INTERCEPT_FUNCTION(wcscmp);
  INTERCEPT_FUNCTION(getenv);
  INTERCEPT_FUNCTION(setenv);
  INTERCEPT_FUNCTION(putenv);
  INTERCEPT_FUNCTION(gettimeofday);
  INTERCEPT_FUNCTION(fcvt);
  MSAN_MAYBE_INTERCEPT___FXSTAT;
  MSAN_INTERCEPT_FSTATAT;
  MSAN_MAYBE_INTERCEPT___FXSTAT64;
  MSAN_MAYBE_INTERCEPT___FXSTATAT64;
  INTERCEPT_FUNCTION(pipe);
  INTERCEPT_FUNCTION(pipe2);
  INTERCEPT_FUNCTION(socketpair);
  INTERCEPT_FUNCTION(fgets);
  MSAN_MAYBE_INTERCEPT_FGETS_UNLOCKED;
  INTERCEPT_FUNCTION(getrlimit);
  MSAN_MAYBE_INTERCEPT_GETRLIMIT64;
  MSAN_MAYBE_INTERCEPT_PRLIMIT;
  MSAN_MAYBE_INTERCEPT_PRLIMIT64;
  MSAN_INTERCEPT_UNAME;
  INTERCEPT_FUNCTION(gethostname);
  MSAN_MAYBE_INTERCEPT_EPOLL_WAIT;
  MSAN_MAYBE_INTERCEPT_EPOLL_PWAIT;
  INTERCEPT_FUNCTION(dladdr);
  INTERCEPT_FUNCTION(dlerror);
  INTERCEPT_FUNCTION(dl_iterate_phdr);
  INTERCEPT_FUNCTION(getrusage);
#if defined(__mips__)
  INTERCEPT_FUNCTION_VER(pthread_create, "GLIBC_2.2");
#else
  INTERCEPT_FUNCTION(pthread_create);
#endif
  INTERCEPT_FUNCTION(pthread_key_create);
  INTERCEPT_FUNCTION(pthread_join);
  INTERCEPT_FUNCTION(tzset);
  INTERCEPT_FUNCTION(__cxa_atexit);
  INTERCEPT_FUNCTION(shmat);
  INTERCEPT_FUNCTION(fork);
  INTERCEPT_FUNCTION(openpty);
  INTERCEPT_FUNCTION(forkpty);

  inited = 1;
}
} // namespace __msan

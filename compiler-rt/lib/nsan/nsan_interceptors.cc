//===-- nsan_interceptors.cc ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Interceptors for standard library functions.
//
// A note about `printf`: Make sure none of the interceptor code calls any
// part of the nsan framework that can call `printf`, since this could create
// a loop (`printf` itself uses the libc). printf-free functions are documented
// as such in nsan.h.
//
//===----------------------------------------------------------------------===//

#include "interception/interception.h"
#include "nsan/nsan.h"
#include "sanitizer_common/sanitizer_common.h"

#include <wchar.h>

#if SANITIZER_LINUX
extern "C" int mallopt(int param, int value);
#endif

using namespace __sanitizer;
using __nsan::NsanInitialized;
using __nsan::NsanInitIsRunning;

static constexpr uptr kEarlyAllocBufSize = 16384;
static uptr AllocatedBytes;
static char EarlyAllocBuf[kEarlyAllocBufSize];

static bool isInEarlyAllocBuf(const void *Ptr) {
  return ((uptr)Ptr >= (uptr)EarlyAllocBuf &&
          ((uptr)Ptr - (uptr)EarlyAllocBuf) < sizeof(EarlyAllocBuf));
}

static u8 *toU8Ptr(wchar_t *ptr) { return reinterpret_cast<u8 *>(ptr); }

static const u8 *toU8Ptr(const wchar_t *ptr) {
  return reinterpret_cast<const u8 *>(ptr);
}

template <typename T> T min(T a, T b) { return a < b ? a : b; }

// Handle allocation requests early (before all interceptors are setup). dlsym,
// for example, calls calloc.
static void *handleEarlyAlloc(uptr Size) {
  void *Mem = (void *)&EarlyAllocBuf[AllocatedBytes];
  AllocatedBytes += Size;
  CHECK_LT(AllocatedBytes, kEarlyAllocBufSize);
  return Mem;
}

INTERCEPTOR(void *, memset, void *Dst, int V, uptr Size) {
  // NOTE: This guard is needed because nsan's initialization code might call
  // memset.
  if (!NsanInitialized && REAL(memset) == nullptr)
    return internal_memset(Dst, V, Size);

  void *Res = REAL(memset)(Dst, V, Size);
  __nsan_set_value_unknown(static_cast<u8 *>(Dst), Size);
  return Res;
}

INTERCEPTOR(wchar_t *, wmemset, wchar_t *Dst, wchar_t V, uptr Size) {
  wchar_t *Res = REAL(wmemset)(Dst, V, Size);
  __nsan_set_value_unknown(toU8Ptr(Dst), sizeof(wchar_t) * Size);
  return Res;
}

INTERCEPTOR(void *, memmove, void *Dst, const void *Src, uptr Size) {
  // NOTE: This guard is needed because nsan's initialization code might call
  // memmove.
  if (!NsanInitialized && REAL(memmove) == nullptr)
    return internal_memmove(Dst, Src, Size);

  void *Res = REAL(memmove)(Dst, Src, Size);
  __nsan_copy_values(static_cast<u8 *>(Dst), static_cast<const u8 *>(Src),
                     Size);
  return Res;
}

INTERCEPTOR(wchar_t *, wmemmove, wchar_t *Dst, const wchar_t *Src, uptr Size) {
  wchar_t *Res = REAL(wmemmove)(Dst, Src, Size);
  __nsan_copy_values(toU8Ptr(Dst), toU8Ptr(Src), sizeof(wchar_t) * Size);
  return Res;
}

INTERCEPTOR(void *, memcpy, void *Dst, const void *Src, uptr Size) {
  // NOTE: This guard is needed because nsan's initialization code might call
  // memcpy.
  if (!NsanInitialized && REAL(memcpy) == nullptr) {
    // memmove is used here because on some platforms this will also
    // intercept the memmove implementation.
    return internal_memmove(Dst, Src, Size);
  }

  void *Res = REAL(memcpy)(Dst, Src, Size);
  __nsan_copy_values(static_cast<u8 *>(Dst), static_cast<const u8 *>(Src),
                     Size);
  return Res;
}

INTERCEPTOR(wchar_t *, wmemcpy, wchar_t *Dst, const wchar_t *Src, uptr Size) {
  wchar_t *Res = REAL(wmemcpy)(Dst, Src, Size);
  __nsan_copy_values(toU8Ptr(Dst), toU8Ptr(Src), sizeof(wchar_t) * Size);
  return Res;
}

INTERCEPTOR(void *, malloc, uptr Size) {
  // NOTE: This guard is needed because nsan's initialization code might call
  // malloc.
  if (NsanInitIsRunning && REAL(malloc) == nullptr)
    return handleEarlyAlloc(Size);

  void *Res = REAL(malloc)(Size);
  if (Res)
    __nsan_set_value_unknown(static_cast<u8 *>(Res), Size);
  return Res;
}

INTERCEPTOR(void *, realloc, void *Ptr, uptr Size) {
  void *Res = REAL(realloc)(Ptr, Size);
  // FIXME: We might want to copy the types from the original allocation
  // (although that would require that we know its size).
  if (Res)
    __nsan_set_value_unknown(static_cast<u8 *>(Res), Size);
  return Res;
}

INTERCEPTOR(void *, calloc, uptr Nmemb, uptr Size) {
  // NOTE: This guard is needed because nsan's initialization code might call
  // calloc.
  if (NsanInitIsRunning && REAL(calloc) == nullptr) {
    // Note: EarlyAllocBuf is initialized with zeros.
    return handleEarlyAlloc(Nmemb * Size);
  }

  void *Res = REAL(calloc)(Nmemb, Size);
  if (Res)
    __nsan_set_value_unknown(static_cast<u8 *>(Res), Nmemb * Size);
  return Res;
}

INTERCEPTOR(void, free, void *P) {
  // There are only a few early allocation requests, so we simply skip the free.
  if (isInEarlyAllocBuf(P))
    return;
  REAL(free)(P);
}

INTERCEPTOR(void *, valloc, uptr Size) {
  void *const Res = REAL(valloc)(Size);
  if (Res)
    __nsan_set_value_unknown(static_cast<u8 *>(Res), Size);
  return Res;
}

INTERCEPTOR(void *, memalign, uptr Alignment, uptr Size) {
  void *const Res = REAL(memalign)(Alignment, Size);
  if (Res)
    __nsan_set_value_unknown(static_cast<u8 *>(Res), Size);
  return Res;
}

INTERCEPTOR(void *, __libc_memalign, uptr Alignment, uptr Size) {
  void *const Res = REAL(__libc_memalign)(Alignment, Size);
  if (Res)
    __nsan_set_value_unknown(static_cast<u8 *>(Res), Size);
  return Res;
}

INTERCEPTOR(void *, pvalloc, uptr Size) {
  void *const Res = REAL(pvalloc)(Size);
  if (Res)
    __nsan_set_value_unknown(static_cast<u8 *>(Res), Size);
  return Res;
}

INTERCEPTOR(void *, aligned_alloc, uptr Alignment, uptr Size) {
  void *const Res = REAL(aligned_alloc)(Alignment, Size);
  if (Res)
    __nsan_set_value_unknown(static_cast<u8 *>(Res), Size);
  return Res;
}

INTERCEPTOR(int, posix_memalign, void **Memptr, uptr Alignment, uptr Size) {
  int Res = REAL(posix_memalign)(Memptr, Alignment, Size);
  if (Res == 0 && *Memptr)
    __nsan_set_value_unknown(static_cast<u8 *>(*Memptr), Size);
  return Res;
}

INTERCEPTOR(char *, strfry, char *S) {
  const auto Len = internal_strlen(S);
  char *Res = REAL(strfry)(S);
  if (Res)
    __nsan_set_value_unknown(reinterpret_cast<u8 *>(S), Len);
  return Res;
}

INTERCEPTOR(char *, strsep, char **Stringp, const char *Delim) {
  char *OrigStringp = REAL(strsep)(Stringp, Delim);
  if (Stringp != nullptr) {
    // The previous character has been overwritten with a '\0' char.
    __nsan_set_value_unknown(reinterpret_cast<u8 *>(*Stringp) - 1, 1);
  }
  return OrigStringp;
}

INTERCEPTOR(char *, strtok, char *Str, const char *Delim) {
  // This is overly conservative, but the probability that modern code is using
  // strtok on double data is essentially zero anyway.
  if (Str)
    __nsan_set_value_unknown(reinterpret_cast<u8 *>(Str), internal_strlen(Str));
  return REAL(strtok)(Str, Delim);
}

static void nsanCopyZeroTerminated(char *Dst, const char *Src, uptr N) {
  __nsan_copy_values(reinterpret_cast<u8 *>(Dst),
                     reinterpret_cast<const u8 *>(Src), N);     // Data.
  __nsan_set_value_unknown(reinterpret_cast<u8 *>(Dst) + N, 1); // Terminator.
}

static void nsanWCopyZeroTerminated(wchar_t *Dst, const wchar_t *Src, uptr N) {
  __nsan_copy_values(toU8Ptr(Dst), toU8Ptr(Src), sizeof(wchar_t) * N);
  __nsan_set_value_unknown(toU8Ptr(Dst + N), sizeof(wchar_t));
}

INTERCEPTOR(char *, strdup, const char *S) {
  char *Res = REAL(strdup)(S);
  if (Res) {
    nsanCopyZeroTerminated(Res, S, internal_strlen(S));
  }
  return Res;
}

INTERCEPTOR(wchar_t *, wcsdup, const wchar_t *S) {
  wchar_t *Res = REAL(wcsdup)(S);
  if (Res) {
    nsanWCopyZeroTerminated(Res, S, wcslen(S));
  }
  return Res;
}

INTERCEPTOR(char *, strndup, const char *S, uptr Size) {
  char *Res = REAL(strndup)(S, Size);
  if (Res) {
    nsanCopyZeroTerminated(Res, S, min(internal_strlen(S), Size));
  }
  return Res;
}

INTERCEPTOR(char *, strcpy, char *Dst, const char *Src) {
  char *Res = REAL(strcpy)(Dst, Src);
  nsanCopyZeroTerminated(Dst, Src, internal_strlen(Src));
  return Res;
}

INTERCEPTOR(wchar_t *, wcscpy, wchar_t *Dst, const wchar_t *Src) {
  wchar_t *Res = REAL(wcscpy)(Dst, Src);
  nsanWCopyZeroTerminated(Dst, Src, wcslen(Src));
  return Res;
}

INTERCEPTOR(char *, strncpy, char *Dst, const char *Src, uptr Size) {
  char *Res = REAL(strncpy)(Dst, Src, Size);
  nsanCopyZeroTerminated(Dst, Src, min(Size, internal_strlen(Src)));
  return Res;
}

INTERCEPTOR(char *, strcat, char *Dst, const char *Src) {
  const auto DstLenBeforeCat = internal_strlen(Dst);
  char *Res = REAL(strcat)(Dst, Src);
  nsanCopyZeroTerminated(Dst + DstLenBeforeCat, Src, internal_strlen(Src));
  return Res;
}

INTERCEPTOR(wchar_t *, wcscat, wchar_t *Dst, const wchar_t *Src) {
  const auto DstLenBeforeCat = wcslen(Dst);
  wchar_t *Res = REAL(wcscat)(Dst, Src);
  nsanWCopyZeroTerminated(Dst + DstLenBeforeCat, Src, wcslen(Src));
  return Res;
}

INTERCEPTOR(char *, strncat, char *Dst, const char *Src, uptr Size) {
  const auto DstLen = internal_strlen(Dst);
  char *Res = REAL(strncat)(Dst, Src, Size);
  nsanCopyZeroTerminated(Dst + DstLen, Src, min(Size, internal_strlen(Src)));
  return Res;
}

INTERCEPTOR(char *, stpcpy, char *Dst, const char *Src) {
  char *Res = REAL(stpcpy)(Dst, Src);
  nsanCopyZeroTerminated(Dst, Src, internal_strlen(Src));
  return Res;
}

INTERCEPTOR(wchar_t *, wcpcpy, wchar_t *Dst, const wchar_t *Src) {
  wchar_t *Res = REAL(wcpcpy)(Dst, Src);
  nsanWCopyZeroTerminated(Dst, Src, wcslen(Src));
  return Res;
}

INTERCEPTOR(uptr, strxfrm, char *Dst, const char *Src, uptr Size) {
  // This is overly conservative, but this function should very rarely be used.
  __nsan_set_value_unknown(reinterpret_cast<u8 *>(Dst), internal_strlen(Dst));
  const uptr Res = REAL(strxfrm)(Dst, Src, Size);
  return Res;
}

namespace __nsan {
void initializeInterceptors() {
  static bool Initialized = false;
  CHECK(!Initialized);

  // Instruct libc malloc to consume less memory.
#if SANITIZER_LINUX
  mallopt(1, 0);          // M_MXFAST
  mallopt(-3, 32 * 1024); // M_MMAP_THRESHOLD
#endif

  INTERCEPT_FUNCTION(malloc);
  INTERCEPT_FUNCTION(calloc);
  INTERCEPT_FUNCTION(free);
  INTERCEPT_FUNCTION(realloc);
  INTERCEPT_FUNCTION(valloc);
  INTERCEPT_FUNCTION(memalign);
  INTERCEPT_FUNCTION(__libc_memalign);
  INTERCEPT_FUNCTION(pvalloc);
  INTERCEPT_FUNCTION(aligned_alloc);
  INTERCEPT_FUNCTION(posix_memalign);

  INTERCEPT_FUNCTION(memset);
  INTERCEPT_FUNCTION(wmemset);
  INTERCEPT_FUNCTION(memmove);
  INTERCEPT_FUNCTION(wmemmove);
  INTERCEPT_FUNCTION(memcpy);
  INTERCEPT_FUNCTION(wmemcpy);

  INTERCEPT_FUNCTION(strdup);
  INTERCEPT_FUNCTION(wcsdup);
  INTERCEPT_FUNCTION(strndup);
  INTERCEPT_FUNCTION(stpcpy);
  INTERCEPT_FUNCTION(wcpcpy);
  INTERCEPT_FUNCTION(strcpy);
  INTERCEPT_FUNCTION(wcscpy);
  INTERCEPT_FUNCTION(strncpy);
  INTERCEPT_FUNCTION(strcat);
  INTERCEPT_FUNCTION(wcscat);
  INTERCEPT_FUNCTION(strncat);
  INTERCEPT_FUNCTION(strxfrm);

  INTERCEPT_FUNCTION(strfry);
  INTERCEPT_FUNCTION(strsep);
  INTERCEPT_FUNCTION(strtok);

  Initialized = 1;
}
} // end namespace __nsan

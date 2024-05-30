//===-- asan_malloc_win_thunk.cpp
//-----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// Windows-specific malloc interception.
// This is included statically for projects statically linking
// with the C Runtime (/MT, /MTd) in order to provide ASAN-aware
// versions of the C allocation functions.
//===----------------------------------------------------------------------===//

#ifdef SANITIZER_STATIC_RUNTIME_THUNK
#  include "..\sanitizer_common\sanitizer_allocator_interface.h"
// #include "asan_win_thunk_common.h"

// Preserve stack traces with noinline.
#  define STATIC_MALLOC_INTERFACE __declspec(noinline)

extern "C" {
__declspec(dllimport) size_t __cdecl __asan_msize(void *ptr);
__declspec(dllimport) void __cdecl __asan_free(void *const ptr);
__declspec(dllimport) void *__cdecl __asan_malloc(const size_t size);
__declspec(dllimport) void *__cdecl __asan_calloc(const size_t nmemb,
                                                  const size_t size);
__declspec(dllimport) void *__cdecl __asan_realloc(void *const ptr,
                                                   const size_t size);
__declspec(dllimport) void *__cdecl __asan_recalloc(void *const ptr,
                                                    const size_t nmemb,
                                                    const size_t size);

// Avoid tailcall optimization to preserve stack frames.
#  pragma optimize("", off)

// _msize
STATIC_MALLOC_INTERFACE size_t _msize(void *ptr) { return __asan_msize(ptr); }

STATIC_MALLOC_INTERFACE size_t _msize_base(void *ptr) {
  return __asan_msize(ptr);
}

STATIC_MALLOC_INTERFACE size_t _msize_dbg(void *ptr) {
  return __asan_msize(ptr);
}

// free
STATIC_MALLOC_INTERFACE void free(void *const ptr) { return __asan_free(ptr); }

STATIC_MALLOC_INTERFACE void _free_base(void *const ptr) {
  return __asan_free(ptr);
}

STATIC_MALLOC_INTERFACE void _free_dbg(void *const ptr) {
  return __asan_free(ptr);
}

// malloc
STATIC_MALLOC_INTERFACE void *malloc(const size_t size) {
  return __asan_malloc(size);
}

STATIC_MALLOC_INTERFACE void *_malloc_base(const size_t size) {
  return __asan_malloc(size);
}

STATIC_MALLOC_INTERFACE void *_malloc_dbg(const size_t size) {
  return __asan_malloc(size);
}

// calloc
STATIC_MALLOC_INTERFACE void *calloc(const size_t nmemb, const size_t size) {
  return __asan_calloc(nmemb, size);
}

STATIC_MALLOC_INTERFACE void *_calloc_base(const size_t nmemb,
                                           const size_t size) {
  return __asan_calloc(nmemb, size);
}

STATIC_MALLOC_INTERFACE void *_calloc_impl(const size_t nmemb,
                                           const size_t size,
                                           int *const errno_tmp) {
  // Provided by legacy msvcrt.
  (void)errno_tmp;

  return __asan_calloc(nmemb, size);
}

STATIC_MALLOC_INTERFACE void *_calloc_dbg(const size_t nmemb, const size_t size,
                                          int, const char *, int) {
  return __asan_calloc(nmemb, size);
}

// realloc
STATIC_MALLOC_INTERFACE void *realloc(void *const ptr, const size_t size) {
  return __asan_realloc(ptr, size);
}

STATIC_MALLOC_INTERFACE void *_realloc_base(void *const ptr,
                                            const size_t size) {
  return __asan_realloc(ptr, size);
}

STATIC_MALLOC_INTERFACE void *_realloc_dbg(void *const ptr, const size_t size,
                                           int, const char *, int) {
  return __asan_realloc(ptr, size);
}

// recalloc
STATIC_MALLOC_INTERFACE void *_recalloc(void *const ptr, const size_t nmemb,
                                        const size_t size) {
  return __asan_recalloc(ptr, nmemb, size);
}

STATIC_MALLOC_INTERFACE void *_recalloc_base(void *const ptr,
                                             const size_t nmemb,
                                             const size_t size) {
  return __asan_recalloc(ptr, nmemb, size);
}

STATIC_MALLOC_INTERFACE void *_recalloc_dbg(void *const ptr, const size_t nmemb,
                                            const size_t size, int,
                                            const char *, int) {
  return __asan_recalloc(ptr, nmemb, size);
}

// expand
STATIC_MALLOC_INTERFACE void *_expand(void *, size_t) {
  // _expand is used in realloc-like functions to resize the buffer if possible.
  // We don't want memory to stand still while resizing buffers, so return 0.
  return nullptr;
}

STATIC_MALLOC_INTERFACE void *_expand_dbg(void *, size_t, int, const char *,
                                          int) {
  return nullptr;
}

// We need to provide symbols for all the debug CRT functions if we decide to
// provide any. Most of these functions make no sense under ASan and so we
// make them no-ops.
long _CrtSetBreakAlloc(long const) { return ~0; }

void _CrtSetDbgBlockType(void *const, int const) { return; }

typedef int(__cdecl *CRT_ALLOC_HOOK)(int, void *, size_t, int, long,
                                     const unsigned char *, int);

CRT_ALLOC_HOOK _CrtGetAllocHook() { return nullptr; }

CRT_ALLOC_HOOK _CrtSetAllocHook(CRT_ALLOC_HOOK const hook) { return hook; }

int _CrtCheckMemory() { return 1; }

int _CrtSetDbgFlag(int const new_bits) { return new_bits; }

typedef void (*CrtDoForAllClientObjectsCallback)(void *, void *);

void _CrtDoForAllClientObjects(CrtDoForAllClientObjectsCallback const,
                               void *const) {
  return;
}

int _CrtIsValidPointer(void const *const p, unsigned int const, int const) {
  return p != nullptr;
}

int _CrtIsValidHeapPointer(void const *const block) {
  if (!block) {
    return 0;
  }

  return __sanitizer_get_ownership(block);
}

int _CrtIsMemoryBlock(void const *const, unsigned const, long *const,
                      char **const, int *const) {
  return 0;
}

int _CrtReportBlockType(void const *const) { return -1; }

typedef void(__cdecl *CRT_DUMP_CLIENT)(void *, size_t);

CRT_DUMP_CLIENT _CrtGetDumpClient() { return nullptr; }

CRT_DUMP_CLIENT _CrtSetDumpClient(CRT_DUMP_CLIENT new_client) {
  return new_client;
}

void _CrtMemCheckpoint(void *const) { return; }

int _CrtMemDifference(void *const, void const *const, void const *const) {
  return 0;
}

void _CrtMemDumpAllObjectsSince(void const *const) { return; }

int _CrtDumpMemoryLeaks() { return 0; }

void _CrtMemDumpStatistics(void const *const) { return; }

int _crtDbgFlag{0};
long _crtBreakAlloc{-1};
CRT_DUMP_CLIENT _pfnDumpClient{nullptr};

int *__p__crtDbgFlag() { return &_crtDbgFlag; }

long *__p__crtBreakAlloc() { return &_crtBreakAlloc; }

// TODO: These were added upstream but conflict with definitions in ucrtbased.
// int _CrtDbgReport(int, const char *, int, const char *, const char *, ...) {
//   ShowStatsAndAbort();
// }
//
// int _CrtDbgReportW(int reportType, const wchar_t *, int, const wchar_t *,
//                    const wchar_t *, ...) {
//   ShowStatsAndAbort();
// }
//
// int _CrtSetReportMode(int, int) { return 0; }

}  // extern "C"
#endif  // SANITIZER_STATIC_RUNTIME_THUNK

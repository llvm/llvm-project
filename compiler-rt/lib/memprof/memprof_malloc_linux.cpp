//===-- memprof_malloc_linux.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of MemProfiler, a memory profiler.
//
// Linux-specific malloc interception.
// We simply define functions like malloc, free, realloc, etc.
// They will replace the corresponding libc functions automagically.
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_platform.h"
#if !SANITIZER_LINUX
#error Unsupported OS
#endif

#include "memprof_allocator.h"
#include "memprof_interceptors.h"
#include "memprof_internal.h"
#include "memprof_stack.h"
#include "sanitizer_common/sanitizer_allocator_checks.h"
#include "sanitizer_common/sanitizer_errno.h"
#include "sanitizer_common/sanitizer_tls_get_addr.h"

// ---------------------- Replacement functions ---------------- {{{1
using namespace __memprof;

static uptr allocated_for_dlsym;
static uptr last_dlsym_alloc_size_in_words;
static const uptr kDlsymAllocPoolSize = 1024;
static uptr alloc_memory_for_dlsym[kDlsymAllocPoolSize];

static inline bool IsInDlsymAllocPool(const void *ptr) {
  uptr off = (uptr)ptr - (uptr)alloc_memory_for_dlsym;
  return off < allocated_for_dlsym * sizeof(alloc_memory_for_dlsym[0]);
}

static void *AllocateFromLocalPool(uptr size_in_bytes) {
  uptr size_in_words = RoundUpTo(size_in_bytes, kWordSize) / kWordSize;
  void *mem = (void *)&alloc_memory_for_dlsym[allocated_for_dlsym];
  last_dlsym_alloc_size_in_words = size_in_words;
  allocated_for_dlsym += size_in_words;
  CHECK_LT(allocated_for_dlsym, kDlsymAllocPoolSize);
  return mem;
}

static void DeallocateFromLocalPool(const void *ptr) {
  // Hack: since glibc 2.27 dlsym no longer uses stack-allocated memory to store
  // error messages and instead uses malloc followed by free. To avoid pool
  // exhaustion due to long object filenames, handle that special case here.
  uptr prev_offset = allocated_for_dlsym - last_dlsym_alloc_size_in_words;
  void *prev_mem = (void *)&alloc_memory_for_dlsym[prev_offset];
  if (prev_mem == ptr) {
    REAL(memset)(prev_mem, 0, last_dlsym_alloc_size_in_words * kWordSize);
    allocated_for_dlsym = prev_offset;
    last_dlsym_alloc_size_in_words = 0;
  }
}

static int PosixMemalignFromLocalPool(void **memptr, uptr alignment,
                                      uptr size_in_bytes) {
  if (UNLIKELY(!CheckPosixMemalignAlignment(alignment)))
    return errno_EINVAL;

  CHECK(alignment >= kWordSize);

  uptr addr = (uptr)&alloc_memory_for_dlsym[allocated_for_dlsym];
  uptr aligned_addr = RoundUpTo(addr, alignment);
  uptr aligned_size = RoundUpTo(size_in_bytes, kWordSize);

  uptr *end_mem = (uptr *)(aligned_addr + aligned_size);
  uptr allocated = end_mem - alloc_memory_for_dlsym;
  if (allocated >= kDlsymAllocPoolSize)
    return errno_ENOMEM;

  allocated_for_dlsym = allocated;
  *memptr = (void *)aligned_addr;
  return 0;
}

static inline bool MaybeInDlsym() { return memprof_init_is_running; }

static inline bool UseLocalPool() { return MaybeInDlsym(); }

static void *ReallocFromLocalPool(void *ptr, uptr size) {
  const uptr offset = (uptr)ptr - (uptr)alloc_memory_for_dlsym;
  const uptr copy_size = Min(size, kDlsymAllocPoolSize - offset);
  void *new_ptr;
  if (UNLIKELY(UseLocalPool())) {
    new_ptr = AllocateFromLocalPool(size);
  } else {
    ENSURE_MEMPROF_INITED();
    GET_STACK_TRACE_MALLOC;
    new_ptr = memprof_malloc(size, &stack);
  }
  internal_memcpy(new_ptr, ptr, copy_size);
  return new_ptr;
}

INTERCEPTOR(void, free, void *ptr) {
  GET_STACK_TRACE_FREE;
  if (UNLIKELY(IsInDlsymAllocPool(ptr))) {
    DeallocateFromLocalPool(ptr);
    return;
  }
  memprof_free(ptr, &stack, FROM_MALLOC);
}

#if SANITIZER_INTERCEPT_CFREE
INTERCEPTOR(void, cfree, void *ptr) {
  GET_STACK_TRACE_FREE;
  if (UNLIKELY(IsInDlsymAllocPool(ptr)))
    return;
  memprof_free(ptr, &stack, FROM_MALLOC);
}
#endif // SANITIZER_INTERCEPT_CFREE

INTERCEPTOR(void *, malloc, uptr size) {
  if (UNLIKELY(UseLocalPool()))
    // Hack: dlsym calls malloc before REAL(malloc) is retrieved from dlsym.
    return AllocateFromLocalPool(size);
  ENSURE_MEMPROF_INITED();
  GET_STACK_TRACE_MALLOC;
  return memprof_malloc(size, &stack);
}

INTERCEPTOR(void *, calloc, uptr nmemb, uptr size) {
  if (UNLIKELY(UseLocalPool()))
    // Hack: dlsym calls calloc before REAL(calloc) is retrieved from dlsym.
    return AllocateFromLocalPool(nmemb * size);
  ENSURE_MEMPROF_INITED();
  GET_STACK_TRACE_MALLOC;
  return memprof_calloc(nmemb, size, &stack);
}

INTERCEPTOR(void *, realloc, void *ptr, uptr size) {
  if (UNLIKELY(IsInDlsymAllocPool(ptr)))
    return ReallocFromLocalPool(ptr, size);
  if (UNLIKELY(UseLocalPool()))
    return AllocateFromLocalPool(size);
  ENSURE_MEMPROF_INITED();
  GET_STACK_TRACE_MALLOC;
  return memprof_realloc(ptr, size, &stack);
}

#if SANITIZER_INTERCEPT_REALLOCARRAY
INTERCEPTOR(void *, reallocarray, void *ptr, uptr nmemb, uptr size) {
  ENSURE_MEMPROF_INITED();
  GET_STACK_TRACE_MALLOC;
  return memprof_reallocarray(ptr, nmemb, size, &stack);
}
#endif // SANITIZER_INTERCEPT_REALLOCARRAY

#if SANITIZER_INTERCEPT_MEMALIGN
INTERCEPTOR(void *, memalign, uptr boundary, uptr size) {
  GET_STACK_TRACE_MALLOC;
  return memprof_memalign(boundary, size, &stack, FROM_MALLOC);
}

INTERCEPTOR(void *, __libc_memalign, uptr boundary, uptr size) {
  GET_STACK_TRACE_MALLOC;
  void *res = memprof_memalign(boundary, size, &stack, FROM_MALLOC);
  DTLS_on_libc_memalign(res, size);
  return res;
}
#endif // SANITIZER_INTERCEPT_MEMALIGN

#if SANITIZER_INTERCEPT_ALIGNED_ALLOC
INTERCEPTOR(void *, aligned_alloc, uptr boundary, uptr size) {
  GET_STACK_TRACE_MALLOC;
  return memprof_aligned_alloc(boundary, size, &stack);
}
#endif // SANITIZER_INTERCEPT_ALIGNED_ALLOC

INTERCEPTOR(uptr, malloc_usable_size, void *ptr) {
  GET_CURRENT_PC_BP_SP;
  (void)sp;
  return memprof_malloc_usable_size(ptr, pc, bp);
}

#if SANITIZER_INTERCEPT_MALLOPT_AND_MALLINFO
// We avoid including malloc.h for portability reasons.
// man mallinfo says the fields are "long", but the implementation uses int.
// It doesn't matter much -- we just need to make sure that the libc's mallinfo
// is not called.
struct fake_mallinfo {
  int x[10];
};

INTERCEPTOR(struct fake_mallinfo, mallinfo, void) {
  struct fake_mallinfo res;
  REAL(memset)(&res, 0, sizeof(res));
  return res;
}

INTERCEPTOR(int, mallopt, int cmd, int value) { return 0; }
#endif // SANITIZER_INTERCEPT_MALLOPT_AND_MALLINFO

INTERCEPTOR(int, posix_memalign, void **memptr, uptr alignment, uptr size) {
  if (UNLIKELY(UseLocalPool()))
    return PosixMemalignFromLocalPool(memptr, alignment, size);
  GET_STACK_TRACE_MALLOC;
  return memprof_posix_memalign(memptr, alignment, size, &stack);
}

INTERCEPTOR(void *, valloc, uptr size) {
  GET_STACK_TRACE_MALLOC;
  return memprof_valloc(size, &stack);
}

#if SANITIZER_INTERCEPT_PVALLOC
INTERCEPTOR(void *, pvalloc, uptr size) {
  GET_STACK_TRACE_MALLOC;
  return memprof_pvalloc(size, &stack);
}
#endif // SANITIZER_INTERCEPT_PVALLOC

INTERCEPTOR(void, malloc_stats, void) { __memprof_print_accumulated_stats(); }

namespace __memprof {
void ReplaceSystemMalloc() {}
} // namespace __memprof

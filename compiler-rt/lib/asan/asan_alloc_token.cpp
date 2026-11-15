//===-- asan_alloc_token.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Allocation entry points for the default AllocToken ABI.
//
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_platform.h"

#if SANITIZER_LINUX && !SANITIZER_ANDROID && SANITIZER_WORDSIZE == 64 && \
    (SANITIZER_X64 || defined(__aarch64__))

#  include "asan_allocator.h"
#  include "asan_report.h"
#  include "asan_stack.h"

using namespace __asan;

extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE
void* __alloc_token_malloc(uptr size, uptr token) {
  GET_STACK_TRACE_MALLOC;
  return asan_malloc(size, &stack, &token);
}

SANITIZER_INTERFACE_ATTRIBUTE
void* __alloc_token_calloc(uptr count, uptr size, uptr token) {
  GET_STACK_TRACE_MALLOC;
  return asan_calloc(count, size, &stack, &token);
}

SANITIZER_INTERFACE_ATTRIBUTE
void* __alloc_token_realloc(void* ptr, uptr size, uptr token) {
  GET_STACK_TRACE_MALLOC;
  return asan_realloc(ptr, size, &stack, &token);
}

SANITIZER_INTERFACE_ATTRIBUTE
void* __alloc_token_reallocarray(void* ptr, uptr count, uptr size, uptr token) {
  GET_STACK_TRACE_MALLOC;
  return asan_reallocarray(ptr, count, size, &stack, &token);
}

SANITIZER_INTERFACE_ATTRIBUTE
void* __alloc_token_memalign(uptr alignment, uptr size, uptr token) {
  GET_STACK_TRACE_MALLOC;
  return asan_memalign(alignment, size, &stack, &token);
}

SANITIZER_INTERFACE_ATTRIBUTE
void* __alloc_token_aligned_alloc(uptr alignment, uptr size, uptr token) {
  GET_STACK_TRACE_MALLOC;
  return asan_aligned_alloc(alignment, size, &stack, &token);
}

SANITIZER_INTERFACE_ATTRIBUTE
int __alloc_token_posix_memalign(void** ptr, uptr alignment, uptr size,
                                 uptr token) {
  GET_STACK_TRACE_MALLOC;
  return asan_posix_memalign(ptr, alignment, size, &stack, &token);
}

SANITIZER_INTERFACE_ATTRIBUTE
void* __alloc_token_valloc(uptr size, uptr token) {
  GET_STACK_TRACE_MALLOC;
  return asan_valloc(size, &stack, &token);
}

SANITIZER_INTERFACE_ATTRIBUTE
void* __alloc_token_pvalloc(uptr size, uptr token) {
  GET_STACK_TRACE_MALLOC;
  return asan_pvalloc(size, &stack, &token);
}
}  // extern "C"

// Use ABI-level types for the mangled C++ allocation entry points.
#  define ALLOC_TOKEN_NEW(Name, Args, Allocate, CanReturnNull) \
    extern "C" SANITIZER_INTERFACE_ATTRIBUTE void* Name Args { \
      GET_STACK_TRACE_MALLOC;                                  \
      void* ptr = Allocate;                                    \
      if (!CanReturnNull && UNLIKELY(!ptr))                    \
        ReportOutOfMemory(size, &stack);                       \
      return ptr;                                              \
    }

#  define ALLOC_TOKEN_NEW_VARIANTS(Name, New, AlignedNew)                     \
    ALLOC_TOKEN_NEW(Name, (uptr size, uptr token), New(size, &stack, &token), \
                    false)                                                    \
    ALLOC_TOKEN_NEW(Name##RKSt9nothrow_t,                                     \
                    (uptr size, const void*, uptr token),                     \
                    New(size, &stack, &token), true)                          \
    ALLOC_TOKEN_NEW(Name##St11align_val_t,                                    \
                    (uptr size, uptr alignment, uptr token),                  \
                    AlignedNew(size, alignment, &stack, &token), false)       \
    ALLOC_TOKEN_NEW(Name##St11align_val_tRKSt9nothrow_t,                      \
                    (uptr size, uptr alignment, const void*, uptr token),     \
                    AlignedNew(size, alignment, &stack, &token), true)

ALLOC_TOKEN_NEW_VARIANTS(__alloc_token__Znwm, asan_new, asan_new_aligned)
ALLOC_TOKEN_NEW_VARIANTS(__alloc_token__Znam, asan_new_array,
                         asan_new_array_aligned)

// Hot/cold hints do not affect ASan's allocation policy.
#  define ALLOC_TOKEN_HOT_COLD_NEW_VARIANTS(Name, New, AlignedNew)            \
    ALLOC_TOKEN_NEW(Name##12__hot_cold_t, (uptr size, u8, uptr token),        \
                    New(size, &stack, &token), false)                         \
    ALLOC_TOKEN_NEW(Name##RKSt9nothrow_t12__hot_cold_t,                       \
                    (uptr size, const void*, u8, uptr token),                 \
                    New(size, &stack, &token), true)                          \
    ALLOC_TOKEN_NEW(Name##St11align_val_t12__hot_cold_t,                      \
                    (uptr size, uptr alignment, u8, uptr token),              \
                    AlignedNew(size, alignment, &stack, &token), false)       \
    ALLOC_TOKEN_NEW(Name##St11align_val_tRKSt9nothrow_t12__hot_cold_t,        \
                    (uptr size, uptr alignment, const void*, u8, uptr token), \
                    AlignedNew(size, alignment, &stack, &token), true)

ALLOC_TOKEN_HOT_COLD_NEW_VARIANTS(__alloc_token__Znwm, asan_new,
                                  asan_new_aligned)
ALLOC_TOKEN_HOT_COLD_NEW_VARIANTS(__alloc_token__Znam, asan_new_array,
                                  asan_new_array_aligned)
#  undef ALLOC_TOKEN_HOT_COLD_NEW_VARIANTS

#  undef ALLOC_TOKEN_NEW_VARIANTS
#  undef ALLOC_TOKEN_NEW

struct SizeReturningNewResult {
  void* ptr;
  uptr size;
};

extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE
SizeReturningNewResult __alloc_token___size_returning_new(uptr size,
                                                          uptr token) {
  GET_STACK_TRACE_MALLOC;
  void* ptr = asan_new(size, &stack, &token);
  if (UNLIKELY(!ptr))
    ReportOutOfMemory(size, &stack);
  return {ptr, size};
}

SANITIZER_INTERFACE_ATTRIBUTE
SizeReturningNewResult __alloc_token___size_returning_new_aligned(
    uptr size, uptr alignment, uptr token) {
  GET_STACK_TRACE_MALLOC;
  void* ptr = asan_new_aligned(size, alignment, &stack, &token);
  if (UNLIKELY(!ptr))
    ReportOutOfMemory(size, &stack);
  return {ptr, size};
}

SANITIZER_INTERFACE_ATTRIBUTE
SizeReturningNewResult __alloc_token___size_returning_new_hot_cold(uptr size,
                                                                   u8,
                                                                   uptr token) {
  return __alloc_token___size_returning_new(size, token);
}

SANITIZER_INTERFACE_ATTRIBUTE
SizeReturningNewResult __alloc_token___size_returning_new_aligned_hot_cold(
    uptr size, uptr alignment, u8, uptr token) {
  return __alloc_token___size_returning_new_aligned(size, alignment, token);
}
}  // extern "C"

#endif  // Supported Linux targets.

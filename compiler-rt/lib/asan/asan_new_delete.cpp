//===-- asan_interceptors.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// Interceptors for operators new and delete.
//===----------------------------------------------------------------------===//

#include <stddef.h>

#include "asan_allocator.h"
#include "asan_internal.h"
#include "asan_report.h"
#include "asan_stack.h"
#include "interception/interception.h"

// C++ operators can't have dllexport attributes on Windows. We export them
// anyway by passing extra -export flags to the linker, which is exactly that
// dllexport would normally do. We need to export them in order to make the
// VS2015 dynamic CRT (MD) work.
#if SANITIZER_WINDOWS && defined(_MSC_VER)
#define CXX_OPERATOR_ATTRIBUTE
#define COMMENT_EXPORT(sym) __pragma(comment(linker, "/export:" sym))
#ifdef _WIN64
COMMENT_EXPORT("??2@YAPEAX_K@Z")                     // operator new
COMMENT_EXPORT("??2@YAPEAX_KAEBUnothrow_t@std@@@Z")  // operator new nothrow
COMMENT_EXPORT("??3@YAXPEAX@Z")                      // operator delete
COMMENT_EXPORT("??3@YAXPEAX_K@Z")                    // sized operator delete
COMMENT_EXPORT("??_U@YAPEAX_K@Z")                    // operator new[]
COMMENT_EXPORT("??_V@YAXPEAX@Z")                     // operator delete[]
#else
COMMENT_EXPORT("??2@YAPAXI@Z")                    // operator new
COMMENT_EXPORT("??2@YAPAXIABUnothrow_t@std@@@Z")  // operator new nothrow
COMMENT_EXPORT("??3@YAXPAX@Z")                    // operator delete
COMMENT_EXPORT("??3@YAXPAXI@Z")                   // sized operator delete
COMMENT_EXPORT("??_U@YAPAXI@Z")                   // operator new[]
COMMENT_EXPORT("??_V@YAXPAX@Z")                   // operator delete[]
#endif
#undef COMMENT_EXPORT
#else
#define CXX_OPERATOR_ATTRIBUTE INTERCEPTOR_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE
#endif

using namespace __asan;

// This code has issues on OSX.
// See https://github.com/google/sanitizers/issues/131.

// Fake std::nothrow_t and std::align_val_t to avoid including <new>.
namespace std {
struct nothrow_t {};
enum class align_val_t: size_t {};
}  // namespace std

// TODO(alekseyshl): throw std::bad_alloc instead of dying on OOM.
// For local pool allocation, align to SHADOW_GRANULARITY to match asan
// allocator behavior.
#define OPERATOR_NEW_BODY                               \
  GET_STACK_TRACE_MALLOC;                               \
  void *res = asan_memalign(0, size, &stack, FROM_NEW); \
  if (UNLIKELY(!res))                                   \
    ReportOutOfMemory(size, &stack);                    \
  return res
#define OPERATOR_NEW_BODY_NOTHROW \
  GET_STACK_TRACE_MALLOC;         \
  return asan_memalign(0, size, &stack, FROM_NEW)
#define OPERATOR_NEW_BODY_ARRAY                            \
  GET_STACK_TRACE_MALLOC;                                  \
  void *res = asan_memalign(0, size, &stack, FROM_NEW_BR); \
  if (UNLIKELY(!res))                                      \
    ReportOutOfMemory(size, &stack);                       \
  return res
#define OPERATOR_NEW_BODY_ARRAY_NOTHROW \
  GET_STACK_TRACE_MALLOC;               \
  return asan_memalign(0, size, &stack, FROM_NEW_BR)
#define OPERATOR_NEW_BODY_ALIGN                                   \
  GET_STACK_TRACE_MALLOC;                                         \
  void *res = asan_memalign((uptr)align, size, &stack, FROM_NEW); \
  if (UNLIKELY(!res))                                             \
    ReportOutOfMemory(size, &stack);                              \
  return res
#define OPERATOR_NEW_BODY_ALIGN_NOTHROW \
  GET_STACK_TRACE_MALLOC;               \
  return asan_memalign((uptr)align, size, &stack, FROM_NEW)
#define OPERATOR_NEW_BODY_ALIGN_ARRAY                                \
  GET_STACK_TRACE_MALLOC;                                            \
  void *res = asan_memalign((uptr)align, size, &stack, FROM_NEW_BR); \
  if (UNLIKELY(!res))                                                \
    ReportOutOfMemory(size, &stack);                                 \
  return res
#define OPERATOR_NEW_BODY_ALIGN_ARRAY_NOTHROW \
  GET_STACK_TRACE_MALLOC;                     \
  return asan_memalign((uptr)align, size, &stack, FROM_NEW_BR)

// On OS X it's not enough to just provide our own 'operator new' and
// 'operator delete' implementations, because they're going to be in the
// runtime dylib, and the main executable will depend on both the runtime
// dylib and libstdc++, each of those'll have its implementation of new and
// delete.
// To make sure that C++ allocation/deallocation operators are overridden on
// OS X we need to intercept them using their mangled names.
#if !SANITIZER_APPLE
CXX_OPERATOR_ATTRIBUTE
void *operator new(size_t size) { OPERATOR_NEW_BODY; }
CXX_OPERATOR_ATTRIBUTE
void *operator new[](size_t size) { OPERATOR_NEW_BODY_ARRAY; }
CXX_OPERATOR_ATTRIBUTE
void *operator new(size_t size, std::nothrow_t const &) {
  OPERATOR_NEW_BODY_NOTHROW;
}
CXX_OPERATOR_ATTRIBUTE
void *operator new[](size_t size, std::nothrow_t const &) {
  OPERATOR_NEW_BODY_ARRAY_NOTHROW;
}
CXX_OPERATOR_ATTRIBUTE
void *operator new(size_t size, std::align_val_t align) {
  OPERATOR_NEW_BODY_ALIGN;
}
CXX_OPERATOR_ATTRIBUTE
void *operator new[](size_t size, std::align_val_t align) {
  OPERATOR_NEW_BODY_ALIGN_ARRAY;
}
CXX_OPERATOR_ATTRIBUTE
void *operator new(size_t size, std::align_val_t align,
                   std::nothrow_t const &) {
  OPERATOR_NEW_BODY_ALIGN_NOTHROW;
}
CXX_OPERATOR_ATTRIBUTE
void *operator new[](size_t size, std::align_val_t align,
                     std::nothrow_t const &) {
  OPERATOR_NEW_BODY_ALIGN_ARRAY_NOTHROW;
}

#else  // SANITIZER_APPLE
INTERCEPTOR(void *, _Znwm, size_t size) { OPERATOR_NEW_BODY; }
INTERCEPTOR(void *, _Znam, size_t size) { OPERATOR_NEW_BODY_ARRAY; }
INTERCEPTOR(void *, _ZnwmRKSt9nothrow_t, size_t size, std::nothrow_t const&) {
  OPERATOR_NEW_BODY_NOTHROW;
}
INTERCEPTOR(void *, _ZnamRKSt9nothrow_t, size_t size, std::nothrow_t const&) {
  OPERATOR_NEW_BODY_ARRAY_NOTHROW;
}
#endif  // !SANITIZER_APPLE

#define OPERATOR_DELETE_BODY \
  GET_STACK_TRACE_FREE;      \
  asan_delete(ptr, 0, 0, &stack, FROM_NEW)
#define OPERATOR_DELETE_BODY_ARRAY \
  GET_STACK_TRACE_FREE;            \
  asan_delete(ptr, 0, 0, &stack, FROM_NEW_BR)
#define OPERATOR_DELETE_BODY_ALIGN \
  GET_STACK_TRACE_FREE;            \
  asan_delete(ptr, 0, static_cast<uptr>(align), &stack, FROM_NEW)
#define OPERATOR_DELETE_BODY_ALIGN_ARRAY \
  GET_STACK_TRACE_FREE;                  \
  asan_delete(ptr, 0, static_cast<uptr>(align), &stack, FROM_NEW_BR)
#define OPERATOR_DELETE_BODY_SIZE \
  GET_STACK_TRACE_FREE;           \
  asan_delete(ptr, size, 0, &stack, FROM_NEW)
#define OPERATOR_DELETE_BODY_SIZE_ARRAY \
  GET_STACK_TRACE_FREE;                 \
  asan_delete(ptr, size, 0, &stack, FROM_NEW_BR)
#define OPERATOR_DELETE_BODY_SIZE_ALIGN \
  GET_STACK_TRACE_FREE;                 \
  asan_delete(ptr, size, static_cast<uptr>(align), &stack, FROM_NEW)
#define OPERATOR_DELETE_BODY_SIZE_ALIGN_ARRAY \
  GET_STACK_TRACE_FREE;                       \
  asan_delete(ptr, size, static_cast<uptr>(align), &stack, FROM_NEW_BR)

#if !SANITIZER_APPLE
CXX_OPERATOR_ATTRIBUTE
void operator delete(void *ptr) NOEXCEPT { OPERATOR_DELETE_BODY; }
CXX_OPERATOR_ATTRIBUTE
void operator delete[](void *ptr) NOEXCEPT { OPERATOR_DELETE_BODY_ARRAY; }
CXX_OPERATOR_ATTRIBUTE
void operator delete(void *ptr, std::nothrow_t const &) {
  OPERATOR_DELETE_BODY;
}
CXX_OPERATOR_ATTRIBUTE
void operator delete[](void *ptr, std::nothrow_t const &) {
  OPERATOR_DELETE_BODY_ARRAY;
}
CXX_OPERATOR_ATTRIBUTE
void operator delete(void *ptr, size_t size) NOEXCEPT {
  OPERATOR_DELETE_BODY_SIZE;
}
CXX_OPERATOR_ATTRIBUTE
void operator delete[](void *ptr, size_t size) NOEXCEPT {
  OPERATOR_DELETE_BODY_SIZE_ARRAY;
}
CXX_OPERATOR_ATTRIBUTE
void operator delete(void *ptr, std::align_val_t align) NOEXCEPT {
  OPERATOR_DELETE_BODY_ALIGN;
}
CXX_OPERATOR_ATTRIBUTE
void operator delete[](void *ptr, std::align_val_t align) NOEXCEPT {
  OPERATOR_DELETE_BODY_ALIGN_ARRAY;
}
CXX_OPERATOR_ATTRIBUTE
void operator delete(void *ptr, std::align_val_t align,
                     std::nothrow_t const &) {
  OPERATOR_DELETE_BODY_ALIGN;
}
CXX_OPERATOR_ATTRIBUTE
void operator delete[](void *ptr, std::align_val_t align,
                       std::nothrow_t const &) {
  OPERATOR_DELETE_BODY_ALIGN_ARRAY;
}
CXX_OPERATOR_ATTRIBUTE
void operator delete(void *ptr, size_t size, std::align_val_t align) NOEXCEPT {
  OPERATOR_DELETE_BODY_SIZE_ALIGN;
}
CXX_OPERATOR_ATTRIBUTE
void operator delete[](void *ptr, size_t size,
                       std::align_val_t align) NOEXCEPT {
  OPERATOR_DELETE_BODY_SIZE_ALIGN_ARRAY;
}

#else  // SANITIZER_APPLE
INTERCEPTOR(void, _ZdlPv, void *ptr) { OPERATOR_DELETE_BODY; }
INTERCEPTOR(void, _ZdaPv, void *ptr) { OPERATOR_DELETE_BODY_ARRAY; }
INTERCEPTOR(void, _ZdlPvRKSt9nothrow_t, void *ptr, std::nothrow_t const &) {
  OPERATOR_DELETE_BODY;
}
INTERCEPTOR(void, _ZdaPvRKSt9nothrow_t, void *ptr, std::nothrow_t const &) {
  OPERATOR_DELETE_BODY_ARRAY;
}
#endif  // !SANITIZER_APPLE

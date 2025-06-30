//===-- memprof_interceptors.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of MemProfiler, a memory profiler.
//
// Interceptors for operators new and delete.
//===----------------------------------------------------------------------===//

#include "memprof_allocator.h"
#include "memprof_internal.h"
#include "memprof_stack.h"
#include "sanitizer_common/sanitizer_allocator_report.h"

#include "interception/interception.h"

#include <stddef.h>

#define CXX_OPERATOR_ATTRIBUTE INTERCEPTOR_ATTRIBUTE

using namespace __memprof;

// Fake std::nothrow_t and std::align_val_t to avoid including <new>.
namespace std {
struct nothrow_t {};
enum class align_val_t : size_t {};
} // namespace std

#define OPERATOR_NEW_BODY(type, nothrow)                                       \
  GET_STACK_TRACE_MALLOC;                                                      \
  void *res = memprof_memalign(0, size, &stack, type);                         \
  if (!nothrow && UNLIKELY(!res))                                              \
    ReportOutOfMemory(size, &stack);                                           \
  return res;
#define OPERATOR_NEW_BODY_ALIGN(type, nothrow)                                 \
  GET_STACK_TRACE_MALLOC;                                                      \
  void *res = memprof_memalign((uptr)align, size, &stack, type);               \
  if (!nothrow && UNLIKELY(!res))                                              \
    ReportOutOfMemory(size, &stack);                                           \
  return res;

// On OS X it's not enough to just provide our own 'operator new' and
// 'operator delete' implementations, because they're going to be in the
// runtime dylib, and the main executable will depend on both the runtime
// dylib and libstdc++, each of those'll have its implementation of new and
// delete.
// To make sure that C++ allocation/deallocation operators are overridden on
// OS X we need to intercept them using their mangled names.
#if !SANITIZER_APPLE
CXX_OPERATOR_ATTRIBUTE
void *operator new(size_t size) {
  OPERATOR_NEW_BODY(FROM_NEW, false /*nothrow*/);
}
CXX_OPERATOR_ATTRIBUTE
void *operator new[](size_t size) {
  OPERATOR_NEW_BODY(FROM_NEW_BR, false /*nothrow*/);
}
CXX_OPERATOR_ATTRIBUTE
void *operator new(size_t size, std::nothrow_t const &) {
  OPERATOR_NEW_BODY(FROM_NEW, true /*nothrow*/);
}
CXX_OPERATOR_ATTRIBUTE
void *operator new[](size_t size, std::nothrow_t const &) {
  OPERATOR_NEW_BODY(FROM_NEW_BR, true /*nothrow*/);
}
CXX_OPERATOR_ATTRIBUTE
void *operator new(size_t size, std::align_val_t align) {
  OPERATOR_NEW_BODY_ALIGN(FROM_NEW, false /*nothrow*/);
}
CXX_OPERATOR_ATTRIBUTE
void *operator new[](size_t size, std::align_val_t align) {
  OPERATOR_NEW_BODY_ALIGN(FROM_NEW_BR, false /*nothrow*/);
}
CXX_OPERATOR_ATTRIBUTE
void *operator new(size_t size, std::align_val_t align,
                   std::nothrow_t const &) {
  OPERATOR_NEW_BODY_ALIGN(FROM_NEW, true /*nothrow*/);
}
CXX_OPERATOR_ATTRIBUTE
void *operator new[](size_t size, std::align_val_t align,
                     std::nothrow_t const &) {
  OPERATOR_NEW_BODY_ALIGN(FROM_NEW_BR, true /*nothrow*/);
}
#else  // SANITIZER_APPLE
INTERCEPTOR(void *, _Znwm, size_t size) {
  OPERATOR_NEW_BODY(FROM_NEW, false /*nothrow*/);
}
INTERCEPTOR(void *, _Znam, size_t size) {
  OPERATOR_NEW_BODY(FROM_NEW_BR, false /*nothrow*/);
}
INTERCEPTOR(void *, _ZnwmRKSt9nothrow_t, size_t size, std::nothrow_t const &) {
  OPERATOR_NEW_BODY(FROM_NEW, true /*nothrow*/);
}
INTERCEPTOR(void *, _ZnamRKSt9nothrow_t, size_t size, std::nothrow_t const &) {
  OPERATOR_NEW_BODY(FROM_NEW_BR, true /*nothrow*/);
}
#endif // !SANITIZER_APPLE

#define OPERATOR_DELETE_BODY(type)                                             \
  GET_STACK_TRACE_FREE;                                                        \
  memprof_delete(ptr, 0, 0, &stack, type);

#define OPERATOR_DELETE_BODY_SIZE(type)                                        \
  GET_STACK_TRACE_FREE;                                                        \
  memprof_delete(ptr, size, 0, &stack, type);

#define OPERATOR_DELETE_BODY_ALIGN(type)                                       \
  GET_STACK_TRACE_FREE;                                                        \
  memprof_delete(ptr, 0, static_cast<uptr>(align), &stack, type);

#define OPERATOR_DELETE_BODY_SIZE_ALIGN(type)                                  \
  GET_STACK_TRACE_FREE;                                                        \
  memprof_delete(ptr, size, static_cast<uptr>(align), &stack, type);

#if !SANITIZER_APPLE
CXX_OPERATOR_ATTRIBUTE
void operator delete(void *ptr) NOEXCEPT { OPERATOR_DELETE_BODY(FROM_NEW); }
CXX_OPERATOR_ATTRIBUTE
void operator delete[](void *ptr) NOEXCEPT {
  OPERATOR_DELETE_BODY(FROM_NEW_BR);
}
CXX_OPERATOR_ATTRIBUTE
void operator delete(void *ptr, std::nothrow_t const &) {
  OPERATOR_DELETE_BODY(FROM_NEW);
}
CXX_OPERATOR_ATTRIBUTE
void operator delete[](void *ptr, std::nothrow_t const &) {
  OPERATOR_DELETE_BODY(FROM_NEW_BR);
}
CXX_OPERATOR_ATTRIBUTE
void operator delete(void *ptr, size_t size) NOEXCEPT {
  OPERATOR_DELETE_BODY_SIZE(FROM_NEW);
}
CXX_OPERATOR_ATTRIBUTE
void operator delete[](void *ptr, size_t size) NOEXCEPT {
  OPERATOR_DELETE_BODY_SIZE(FROM_NEW_BR);
}
CXX_OPERATOR_ATTRIBUTE
void operator delete(void *ptr, std::align_val_t align) NOEXCEPT {
  OPERATOR_DELETE_BODY_ALIGN(FROM_NEW);
}
CXX_OPERATOR_ATTRIBUTE
void operator delete[](void *ptr, std::align_val_t align) NOEXCEPT {
  OPERATOR_DELETE_BODY_ALIGN(FROM_NEW_BR);
}
CXX_OPERATOR_ATTRIBUTE
void operator delete(void *ptr, std::align_val_t align,
                     std::nothrow_t const &) {
  OPERATOR_DELETE_BODY_ALIGN(FROM_NEW);
}
CXX_OPERATOR_ATTRIBUTE
void operator delete[](void *ptr, std::align_val_t align,
                       std::nothrow_t const &) {
  OPERATOR_DELETE_BODY_ALIGN(FROM_NEW_BR);
}
CXX_OPERATOR_ATTRIBUTE
void operator delete(void *ptr, size_t size, std::align_val_t align) NOEXCEPT {
  OPERATOR_DELETE_BODY_SIZE_ALIGN(FROM_NEW);
}
CXX_OPERATOR_ATTRIBUTE
void operator delete[](void *ptr, size_t size,
                       std::align_val_t align) NOEXCEPT {
  OPERATOR_DELETE_BODY_SIZE_ALIGN(FROM_NEW_BR);
}
#else  // SANITIZER_APPLE
INTERCEPTOR(void, _ZdlPv, void *ptr) { OPERATOR_DELETE_BODY(FROM_NEW); }
INTERCEPTOR(void, _ZdaPv, void *ptr) { OPERATOR_DELETE_BODY(FROM_NEW_BR); }
INTERCEPTOR(void, _ZdlPvRKSt9nothrow_t, void *ptr, std::nothrow_t const &) {
  OPERATOR_DELETE_BODY(FROM_NEW);
}
INTERCEPTOR(void, _ZdaPvRKSt9nothrow_t, void *ptr, std::nothrow_t const &) {
  OPERATOR_DELETE_BODY(FROM_NEW_BR);
}
#endif // !SANITIZER_APPLE

//===-- nsan_new_delete.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of MemorySanitizer.
//
// Interceptors for operators new and delete.
//===----------------------------------------------------------------------===//

#include "interception/interception.h"
#include "nsan.h"
#include "nsan_allocator.h"
#include "sanitizer_common/sanitizer_allocator.h"
#include "sanitizer_common/sanitizer_allocator_report.h"

#include <stddef.h>

using namespace __nsan;

// Fake std::nothrow_t and std::align_val_t to avoid including <new>.
namespace std {
struct nothrow_t {};
enum class align_val_t : size_t {};
} // namespace std

#define OPERATOR_NEW_BODY(nothrow)                                             \
  void *res = nsan_malloc(size);                                               \
  if (!nothrow && UNLIKELY(!res)) {                                            \
    BufferedStackTrace stack;                                                  \
    GET_FATAL_STACK_TRACE_IF_EMPTY(&stack);                                    \
    ReportOutOfMemory(size, &stack);                                           \
  }                                                                            \
  return res
#define OPERATOR_NEW_BODY_ALIGN(nothrow)                                       \
  void *res = nsan_memalign((uptr)align, size);                                \
  if (!nothrow && UNLIKELY(!res)) {                                            \
    BufferedStackTrace stack;                                                  \
    GET_FATAL_STACK_TRACE_IF_EMPTY(&stack);                                    \
    ReportOutOfMemory(size, &stack);                                           \
  }                                                                            \
  return res;

INTERCEPTOR_ATTRIBUTE
void *operator new(size_t size) { OPERATOR_NEW_BODY(/*nothrow=*/false); }
INTERCEPTOR_ATTRIBUTE
void *operator new[](size_t size) { OPERATOR_NEW_BODY(/*nothrow=*/false); }
INTERCEPTOR_ATTRIBUTE
void *operator new(size_t size, std::nothrow_t const &) {
  OPERATOR_NEW_BODY(/*nothrow=*/true);
}
INTERCEPTOR_ATTRIBUTE
void *operator new[](size_t size, std::nothrow_t const &) {
  OPERATOR_NEW_BODY(/*nothrow=*/true);
}
INTERCEPTOR_ATTRIBUTE
void *operator new(size_t size, std::align_val_t align) {
  OPERATOR_NEW_BODY_ALIGN(/*nothrow=*/false);
}
INTERCEPTOR_ATTRIBUTE
void *operator new[](size_t size, std::align_val_t align) {
  OPERATOR_NEW_BODY_ALIGN(/*nothrow=*/false);
}
INTERCEPTOR_ATTRIBUTE
void *operator new(size_t size, std::align_val_t align,
                   std::nothrow_t const &) {
  OPERATOR_NEW_BODY_ALIGN(/*nothrow=*/true);
}
INTERCEPTOR_ATTRIBUTE
void *operator new[](size_t size, std::align_val_t align,
                     std::nothrow_t const &) {
  OPERATOR_NEW_BODY_ALIGN(/*nothrow=*/true);
}

#define OPERATOR_DELETE_BODY                                                   \
  if (ptr)                                                                     \
  NsanDeallocate(ptr)

INTERCEPTOR_ATTRIBUTE
void operator delete(void *ptr) NOEXCEPT { OPERATOR_DELETE_BODY; }
INTERCEPTOR_ATTRIBUTE
void operator delete[](void *ptr) NOEXCEPT { OPERATOR_DELETE_BODY; }
INTERCEPTOR_ATTRIBUTE
void operator delete(void *ptr, std::nothrow_t const &) {
  OPERATOR_DELETE_BODY;
}
INTERCEPTOR_ATTRIBUTE
void operator delete[](void *ptr, std::nothrow_t const &) {
  OPERATOR_DELETE_BODY;
}
INTERCEPTOR_ATTRIBUTE
void operator delete(void *ptr, size_t size) NOEXCEPT { OPERATOR_DELETE_BODY; }
INTERCEPTOR_ATTRIBUTE
void operator delete[](void *ptr, size_t size) NOEXCEPT {
  OPERATOR_DELETE_BODY;
}
INTERCEPTOR_ATTRIBUTE
void operator delete(void *ptr, std::align_val_t align) NOEXCEPT {
  OPERATOR_DELETE_BODY;
}
INTERCEPTOR_ATTRIBUTE
void operator delete[](void *ptr, std::align_val_t align) NOEXCEPT {
  OPERATOR_DELETE_BODY;
}
INTERCEPTOR_ATTRIBUTE
void operator delete(void *ptr, std::align_val_t align,
                     std::nothrow_t const &) {
  OPERATOR_DELETE_BODY;
}
INTERCEPTOR_ATTRIBUTE
void operator delete[](void *ptr, std::align_val_t align,
                       std::nothrow_t const &) {
  OPERATOR_DELETE_BODY;
}
INTERCEPTOR_ATTRIBUTE
void operator delete(void *ptr, size_t size, std::align_val_t align) NOEXCEPT {
  OPERATOR_DELETE_BODY;
}
INTERCEPTOR_ATTRIBUTE
void operator delete[](void *ptr, size_t size,
                       std::align_val_t align) NOEXCEPT {
  OPERATOR_DELETE_BODY;
}

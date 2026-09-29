//===-- hwasan_new_delete.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of HWAddressSanitizer.
//
// Interceptors for operators new and delete.
//===----------------------------------------------------------------------===//

#include "hwasan.h"
#include "interception/interception.h"
#include "sanitizer_common/sanitizer_allocator.h"
#include "sanitizer_common/sanitizer_allocator_report.h"

#include <stddef.h>
#include <stdlib.h>

#if HWASAN_REPLACE_OPERATORS_NEW_AND_DELETE

// TODO(alekseys): throw std::bad_alloc instead of dying on OOM.
#  define OPERATOR_NEW_BODY                  \
    GET_MALLOC_STACK_TRACE;                  \
    void *res = hwasan_malloc(size, &stack); \
    if (UNLIKELY(!res))                      \
      ReportOutOfMemory(size, &stack);       \
    return res
#  define OPERATOR_NEW_BODY_NOTHROW \
    GET_MALLOC_STACK_TRACE;         \
    return hwasan_malloc(size, &stack)
#  define OPERATOR_NEW_BODY_ARRAY            \
    GET_MALLOC_STACK_TRACE;                  \
    void *res = hwasan_malloc(size, &stack); \
    if (UNLIKELY(!res))                      \
      ReportOutOfMemory(size, &stack);       \
    return res
#  define OPERATOR_NEW_BODY_ARRAY_NOTHROW \
    GET_MALLOC_STACK_TRACE;               \
    return hwasan_malloc(size, &stack)
#  define OPERATOR_NEW_BODY_ALIGN                                        \
    GET_MALLOC_STACK_TRACE;                                              \
    void *res = hwasan_memalign(static_cast<uptr>(align), size, &stack); \
    if (UNLIKELY(!res))                                                  \
      ReportOutOfMemory(size, &stack);                                   \
    return res
#  define OPERATOR_NEW_BODY_ALIGN_NOTHROW \
    GET_MALLOC_STACK_TRACE;               \
    return hwasan_memalign(static_cast<uptr>(align), size, &stack)
#  define OPERATOR_NEW_BODY_ALIGN_ARRAY                                  \
    GET_MALLOC_STACK_TRACE;                                              \
    void *res = hwasan_memalign(static_cast<uptr>(align), size, &stack); \
    if (UNLIKELY(!res))                                                  \
      ReportOutOfMemory(size, &stack);                                   \
    return res
#  define OPERATOR_NEW_BODY_ALIGN_ARRAY_NOTHROW \
    GET_MALLOC_STACK_TRACE;                     \
    return hwasan_memalign(static_cast<uptr>(align), size, &stack)

#  define OPERATOR_DELETE_BODY \
    GET_MALLOC_STACK_TRACE;    \
    if (ptr)                   \
    hwasan_free(ptr, &stack)
#  define OPERATOR_DELETE_BODY_ARRAY \
    GET_MALLOC_STACK_TRACE;          \
    if (ptr)                         \
    hwasan_free(ptr, &stack)
#  define OPERATOR_DELETE_BODY_ALIGN \
    GET_MALLOC_STACK_TRACE;          \
    if (ptr)                         \
    hwasan_free(ptr, &stack)
#  define OPERATOR_DELETE_BODY_ALIGN_ARRAY \
    GET_MALLOC_STACK_TRACE;                \
    if (ptr)                               \
    hwasan_free(ptr, &stack)
#  define OPERATOR_DELETE_BODY_SIZE \
    GET_MALLOC_STACK_TRACE;         \
    if (ptr)                        \
    hwasan_free(ptr, &stack)
#  define OPERATOR_DELETE_BODY_SIZE_ARRAY \
    GET_MALLOC_STACK_TRACE;               \
    if (ptr)                              \
    hwasan_free(ptr, &stack)
#  define OPERATOR_DELETE_BODY_SIZE_ALIGN \
    GET_MALLOC_STACK_TRACE;               \
    if (ptr)                              \
    hwasan_free(ptr, &stack)
#  define OPERATOR_DELETE_BODY_SIZE_ALIGN_ARRAY \
    GET_MALLOC_STACK_TRACE;                     \
    if (ptr)                                    \
    hwasan_free(ptr, &stack)

#elif defined(__ANDROID__)

// We don't actually want to intercept operator new and delete on Android, but
// since we previously released a runtime that intercepted these functions,
// removing the interceptors would break ABI. Therefore we simply forward to
// malloc and free.
#  define OPERATOR_NEW_BODY return malloc(size)
#  define OPERATOR_NEW_BODY_NOTHROW return malloc(size)
#  define OPERATOR_NEW_BODY_ARRAY return malloc(size)
#  define OPERATOR_NEW_BODY_ARRAY_NOTHROW return malloc(size)
#  define OPERATOR_DELETE_BODY free(ptr)
#  define OPERATOR_DELETE_BODY_ARRAY free(ptr)
#  define OPERATOR_DELETE_BODY_SIZE free(ptr)
#  define OPERATOR_DELETE_BODY_SIZE_ARRAY free(ptr)

#endif

#ifdef OPERATOR_NEW_BODY

using namespace __hwasan;

// Fake std::nothrow_t to avoid including <new>.
namespace std {
struct nothrow_t {};
}  // namespace std

INTERCEPTOR_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE void *operator new(size_t size) {
  OPERATOR_NEW_BODY;
}
INTERCEPTOR_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE void *operator new[](
    size_t size) {
  OPERATOR_NEW_BODY_ARRAY;
}
INTERCEPTOR_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE void *operator new(
    size_t size, std::nothrow_t const &) {
  OPERATOR_NEW_BODY_NOTHROW;
}
INTERCEPTOR_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE void *operator new[](
    size_t size, std::nothrow_t const &) {
  OPERATOR_NEW_BODY_ARRAY_NOTHROW;
}

INTERCEPTOR_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE void operator delete(
    void *ptr) NOEXCEPT {
  OPERATOR_DELETE_BODY;
}
INTERCEPTOR_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE void operator delete[](
    void *ptr) NOEXCEPT {
  OPERATOR_DELETE_BODY_ARRAY;
}
INTERCEPTOR_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE void operator delete(
    void *ptr, std::nothrow_t const &) {
  OPERATOR_DELETE_BODY;
}
INTERCEPTOR_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE void operator delete[](
    void *ptr, std::nothrow_t const &) {
  OPERATOR_DELETE_BODY_ARRAY;
}
INTERCEPTOR_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE void operator delete(
    void *ptr, size_t) NOEXCEPT {
  OPERATOR_DELETE_BODY_SIZE;
}
INTERCEPTOR_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE void operator delete[](
    void *ptr, size_t) NOEXCEPT {
  OPERATOR_DELETE_BODY_SIZE_ARRAY;
}

#endif  // OPERATOR_NEW_BODY

#ifdef OPERATOR_NEW_BODY_ALIGN

namespace std {
enum class align_val_t : size_t {};
}  // namespace std

INTERCEPTOR_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE void *operator new(
    size_t size, std::align_val_t align) {
  OPERATOR_NEW_BODY_ALIGN;
}
INTERCEPTOR_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE void *operator new[](
    size_t size, std::align_val_t align) {
  OPERATOR_NEW_BODY_ALIGN_ARRAY;
}
INTERCEPTOR_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE void *operator new(
    size_t size, std::align_val_t align, std::nothrow_t const &) {
  OPERATOR_NEW_BODY_ALIGN_NOTHROW;
}
INTERCEPTOR_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE void *operator new[](
    size_t size, std::align_val_t align, std::nothrow_t const &) {
  OPERATOR_NEW_BODY_ALIGN_ARRAY_NOTHROW;
}

INTERCEPTOR_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE void operator delete(
    void *ptr, std::align_val_t align) NOEXCEPT {
  OPERATOR_DELETE_BODY_ALIGN;
}
INTERCEPTOR_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE void operator delete[](
    void *ptr, std::align_val_t) NOEXCEPT {
  OPERATOR_DELETE_BODY_ALIGN_ARRAY;
}
INTERCEPTOR_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE void operator delete(
    void *ptr, std::align_val_t, std::nothrow_t const &) NOEXCEPT {
  OPERATOR_DELETE_BODY_ALIGN;
}
INTERCEPTOR_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE void operator delete[](
    void *ptr, std::align_val_t, std::nothrow_t const &) NOEXCEPT {
  OPERATOR_DELETE_BODY_ALIGN_ARRAY;
}
INTERCEPTOR_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE void operator delete(
    void *ptr, size_t, std::align_val_t) NOEXCEPT {
  OPERATOR_DELETE_BODY_SIZE_ALIGN;
}
INTERCEPTOR_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE void operator delete[](
    void *ptr, size_t, std::align_val_t) NOEXCEPT {
  OPERATOR_DELETE_BODY_SIZE_ALIGN_ARRAY;
}
INTERCEPTOR_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE void operator delete(
    void *ptr, size_t, std::align_val_t, std::nothrow_t const &) NOEXCEPT {
  OPERATOR_DELETE_BODY_SIZE_ALIGN;
}
INTERCEPTOR_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE void operator delete[](
    void *ptr, size_t, std::align_val_t, std::nothrow_t const &) NOEXCEPT {
  OPERATOR_DELETE_BODY_SIZE_ALIGN_ARRAY;
}

#endif  // OPERATOR_NEW_BODY_ALIGN

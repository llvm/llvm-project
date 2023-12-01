//===-- asan_win_new_delete.cc --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// Windows-specific user-provided new/delete operator detection and fallback.
//===----------------------------------------------------------------------===//
#include <stddef.h>

#include "asan_allocator.h"
#include "asan_internal.h"
#include "asan_report.h"
#include "asan_stack.h"

// Fake std::align_val_t to avoid including <new>.
namespace std {
enum class align_val_t : size_t {};
}

using namespace __asan;

#define OPERATOR_NEW_BODY(type, nothrow)            \
  GET_STACK_TRACE_MALLOC                            \
  void *res = asan_memalign(0, size, &stack, type); \
  if (!nothrow && UNLIKELY(!res))                   \
    ReportOutOfMemory(size, &stack);                \
  return res;

#define OPERATOR_NEW_BODY_ALIGN(type, nothrow)                \
  GET_STACK_TRACE_MALLOC                                      \
  void *res = asan_memalign((uptr)align, size, &stack, type); \
  if (!nothrow && UNLIKELY(!res))                             \
    ReportOutOfMemory(size, &stack);                          \
  return res;

#define OPERATOR_DELETE_BODY(type) \
  GET_STACK_TRACE_FREE             \
  asan_delete(ptr, 0, 0, &stack, type);

#define OPERATOR_DELETE_BODY_SIZE(type) \
  GET_STACK_TRACE_FREE                  \
  asan_delete(ptr, size, 0, &stack, type);

#define OPERATOR_DELETE_BODY_ALIGN(type) \
  GET_STACK_TRACE_FREE                   \
  asan_delete(ptr, 0, static_cast<uptr>(align), &stack, type);

#define OPERATOR_DELETE_BODY_SIZE_ALIGN(type) \
  GET_STACK_TRACE_FREE                        \
  asan_delete(ptr, size, static_cast<uptr>(align), &stack, type);

extern "C" {
__declspec(dllexport) void *__cdecl __asan_new(size_t const size) {
  OPERATOR_NEW_BODY(FROM_NEW, false /*nothrow*/);
}

__declspec(dllexport) void *__cdecl __asan_new_array(size_t const size) {
  OPERATOR_NEW_BODY(FROM_NEW_BR, false /*nothrow*/);
}

__declspec(dllexport) void *__cdecl __asan_new_nothrow(size_t const size) {
  OPERATOR_NEW_BODY(FROM_NEW, true /*nothrow*/);
}

__declspec(dllexport) void *__cdecl __asan_new_array_nothrow(
    size_t const size) {
  OPERATOR_NEW_BODY(FROM_NEW_BR, true /*nothrow*/);
}

__declspec(dllexport) void *__cdecl __asan_new_align(
    size_t const size, std::align_val_t const align) {
  OPERATOR_NEW_BODY_ALIGN(FROM_NEW, false /*nothrow*/);
}

__declspec(dllexport) void *__cdecl __asan_new_array_align(
    size_t const size, std::align_val_t const align) {
  OPERATOR_NEW_BODY_ALIGN(FROM_NEW_BR, false /*nothrow*/);
}

__declspec(dllexport) void *__cdecl __asan_new_align_nothrow(
    size_t const size, std::align_val_t const align) {
  OPERATOR_NEW_BODY_ALIGN(FROM_NEW, true /*nothrow*/);
}

__declspec(dllexport) void *__cdecl __asan_new_array_align_nothrow(
    size_t const size, std::align_val_t const align) {
  OPERATOR_NEW_BODY_ALIGN(FROM_NEW_BR, true /*nothrow*/);
}

__declspec(dllexport) void __cdecl __asan_delete(void *ptr) {
  OPERATOR_DELETE_BODY(FROM_NEW);
}

__declspec(dllexport) void __cdecl __asan_delete_array(void *ptr) {
  OPERATOR_DELETE_BODY(FROM_NEW_BR);
}

__declspec(dllexport) void __cdecl __asan_delete_size(void *ptr,
                                                      size_t const size) {
  OPERATOR_DELETE_BODY_SIZE(FROM_NEW);
}

__declspec(dllexport) void __cdecl __asan_delete_array_size(void *ptr,
                                                            size_t const size) {
  OPERATOR_DELETE_BODY_SIZE(FROM_NEW_BR);
}

__declspec(dllexport) void __cdecl __asan_delete_align(
    void *ptr, std::align_val_t const align) {
  OPERATOR_DELETE_BODY_ALIGN(FROM_NEW);
}

__declspec(dllexport) void __cdecl __asan_delete_array_align(
    void *ptr, std::align_val_t const align) {
  OPERATOR_DELETE_BODY_ALIGN(FROM_NEW_BR);
}

__declspec(dllexport) void __cdecl __asan_delete_size_align(
    void *ptr, size_t const size, std::align_val_t const align) {
  OPERATOR_DELETE_BODY_SIZE_ALIGN(FROM_NEW);
}

__declspec(dllexport) void __cdecl __asan_delete_array_size_align(
    void *ptr, size_t const size, std::align_val_t const align) {
  OPERATOR_DELETE_BODY_SIZE_ALIGN(FROM_NEW_BR);
}
}  // extern "C"

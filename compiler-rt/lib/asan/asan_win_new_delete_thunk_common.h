//===-- asan_win_new_delete_thunk_common.h ----------------------*- C++ -*-===//
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
//
// In order to provide correct fallback behavior for operator new and delete, we
// need to honor partially hooked new and delete operators. For example, if
// plain operator new is provided by the user, then array operator new should
// fallback to use that operator new. This is slighly complicated in that ASAN
// must know which operator new/delete was used to correctly track allocations.
// The solution here is to only pass the allocation/deallocation request
// directly to ASAN with full metadata when we know all fallbacks for the given
// overload are provided by ASAN. This requires us to detect which overloads are
// provided by ASAN. We can accomplish this by seperating the definitions into
// multiple TUs so each can be selected individually, and adding a dynamic
// initializer to those TUs to mark whether that overload is included.
//===----------------------------------------------------------------------===//
#ifndef ASAN_WIN_NEW_DELETE_THUNK_COMMON_H
#define ASAN_WIN_NEW_DELETE_THUNK_COMMON_H

#include "sanitizer_common/sanitizer_internal_defs.h"
// Fake std::nothrow_t and std::align_val_t to avoid including <new>.
namespace std {
struct nothrow_t {};
enum class align_val_t : size_t {};
}  // namespace std

void* operator new(size_t, std::align_val_t);
void* operator new[](size_t, std::align_val_t);
void operator delete(void* ptr, std::align_val_t align) noexcept;
void operator delete[](void* ptr, std::align_val_t align) noexcept;

////////////////////////////////////
// clang-format off
// Fallback Ordering for new/delete
// change this if the code called by a given operator in the case where the
// "less specific" operators are not provided by asan changes.
//
// +----------+                                                     +----------------+
// |new_scalar<---------------+                                     |new_scalar_align<--------------+
// +----^-----+               |                                     +----^-----------+              |
//      |                     |                                          |                          |
// +----+-------------+  +----+----+                                +----+-------------------+  +---+-----------+
// |new_scalar_nothrow|  |new_array|                                |new_scalar_align_nothrow|  |new_array_align|
// +------------------+  +----^----+                                +------------------------+  +---^-----------+
//                            |                                                                     |
//               +------------+----+                                                    +-----------+-----------+
//               |new_array_nothrow|                                                    |new_array_align_nothrow|
//               +-----------------+                                                    +-----------------------+
//
// +-------------+                                                  +-------------------+
// |delete_scalar<----+-----------------------+                     |delete_scalar_align<----+---------------------------+
// +--^----------+    |                       |                     +--^----------------+    |                           |
//    |               |                       |                        |                     |                           |
// +--+---------+  +--+---------------+  +----+----------------+    +--+---------------+  +--+---------------------+  +--+------------------------+
// |delete_array|  |delete_scalar_size|  |delete_scalar_nothrow|    |delete_array_align|  |delete_scalar_size_align|  |delete_scalar_align_nothrow|
// +--^----^----+  +------------------+  +---------------------+    +--^-----^---------+  +------------------------+  +---------------------------+
//    |    |                                                           |     |
//    |    +-------------------+                                       |     +------------------------+
//    |                        |                                       |                              |
// +--+--------------+  +------+-------------+                      +--+--------------------+  +------+-------------------+
// |delete_array_size|  |delete_array_nothrow|                      |delete_array_size_align|  |delete_array_align_nothrow|
// +-----------------+  +--------------------+                      +-----------------------+  +--------------------------+
// clang-format on

// Only need definition detection for overloads with children.
enum defined_ops {
  op_new_scalar,
  op_new_array,

  op_new_scalar_align,
  op_new_array_align,

  op_delete_scalar,
  op_delete_array,

  op_delete_scalar_align,
  op_delete_array_align
};

// Define a global of this type in each overload's translation unit
// so that the dynamic initializer will set defined to 1 when
// that TU is included.
// We can then use __asan_InitDefine<op>::defined to check whether that TU is
// included.
template <defined_ops Id>
struct __asan_InitDefine {
  __asan_InitDefine() { defined = 1; }

  static int defined;
};

template <defined_ops Id>
int __asan_InitDefine<Id>::defined = 0;

extern "C" void __cdecl __asan_delete_array_align(void* ptr,
                                                  std::align_val_t align);

extern "C" void __cdecl __asan_delete_array_size_align(void* ptr, size_t size,
                                                       std::align_val_t align);

extern "C" void __cdecl __asan_delete_array_size(void* ptr, size_t size);

extern "C" void __cdecl __asan_delete_array(void* ptr);

extern "C" void __cdecl __asan_delete_align(void* ptr, std::align_val_t align);

extern "C" void __cdecl __asan_delete_size_align(
    void* ptr, size_t size, std::align_val_t align) noexcept;

extern "C" void __cdecl __asan_delete_size(void* ptr, size_t size);

extern "C" void __cdecl __asan_delete(void* ptr);

extern "C" void* __cdecl __asan_new_array_align_nothrow(size_t size,
                                                        std::align_val_t align);

extern "C" void* __cdecl __asan_new_array_align(size_t size,
                                                std::align_val_t align);

extern "C" void* __cdecl __asan_new_array_nothrow(size_t size);

extern "C" void* __cdecl __asan_new_array(size_t size);

extern "C" void* __cdecl __asan_new_align_nothrow(size_t size,
                                                  std::align_val_t align);

extern "C" void* __cdecl __asan_new_align(size_t size, std::align_val_t align);

extern "C" void* __cdecl __asan_new_nothrow(size_t size);

extern "C" void* __cdecl __asan_new(size_t size);

#endif  // ASAN_WIN_NEW_DELETE_THUNK_COMMON_H

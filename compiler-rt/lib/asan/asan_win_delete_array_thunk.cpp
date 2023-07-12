//===-- asan_win_delete_array_thunk.cc ------------------------------------===//
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
#include "asan_win_new_delete_thunk_common.h"

////////////////////////////////////////////////////////////////
// clang-format off
// delete() Fallback Ordering
//
// +-------------+
// |delete_scalar<----+-----------------------+
// +--^----------+    |                       |
//    |               |                       |
// +--+---------+  +--+---------------+  +----+----------------+
// |DELETE_ARRAY|  |delete_scalar_size|  |delete_scalar_nothrow|
// +--^----^----+  +------------------+  +---------------------+
//    |    |
//    |    +-------------------+
//    |                        |
// +--+--------------+  +------+-------------+
// |delete_array_size|  |delete_array_nothrow|
// +-----------------+  +--------------------+
// clang-format on

__asan_InitDefine<op_delete_array> init_delete_array;

extern "C" void __cdecl __asan_delete_array(__asan_win_new_delete_data* data,
                                            void* ptr);

// Avoid tailcall optimization to preserve stack frame.
#pragma optimize("", off)
void operator delete[](void* ptr) noexcept {
  if (__asan_InitDefine<op_delete_scalar>::defined) {
    __asan_win_new_delete_data data{};
    __asan_delete_array(&data, ptr);
  } else {
    operator delete(ptr);
  }
}

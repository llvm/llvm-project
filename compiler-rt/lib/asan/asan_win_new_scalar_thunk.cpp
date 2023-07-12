//===-- asan_win_new_scalar_thunk.cc --------------------------------------===//
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

////////////////////////////////////
// clang-format off
// new() Fallback Ordering
//
// +----------+
// |NEW_SCALAR|<---------------+
// +----^-----+               |
//      |                     |
// +----+-------------+  +----+----+
// |new_scalar_nothrow|  |new_array|
// +------------------+  +----^----+
//                            |
//               +------------+----+
//               |new_array_nothrow|
//               +-----------------+
// clang-format on

__asan_InitDefine<op_new_scalar> init_new_scalar;

extern "C" void* __cdecl __asan_new(__asan_win_new_delete_data* data,
                                    size_t size);

// Avoid tailcall optimization to preserve stack frame.
#pragma optimize("", off)
void* operator new(size_t size) {
  __asan_win_new_delete_data data{};
  return __asan_new(&data, size);
}

//===-- asan_win_new_array_nothrow_thunk.cc -------------------------------===//
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
// |new_scalar<---------------+
// +----^-----+               |
//      |                     |
// +----+-------------+  +----+----+
// |new_scalar_nothrow|  |new_array|
// +------------------+  +----^----+
//                            |
//               +------------+----+
//               |NEW_ARRAY_NOTHROW|
//               +-----------------+
// clang-format on

extern "C" void* __cdecl __asan_new_array_nothrow(
    __asan_win_new_delete_data* data, size_t size);

// Avoid tailcall optimization to preserve stack frame.
#pragma optimize("", off)
void* operator new[](size_t size, std::nothrow_t const&) noexcept {
  if (__asan_InitDefine<op_new_scalar>::defined &&
      __asan_InitDefine<op_new_array>::defined) {
    __asan_win_new_delete_data data{};
    return __asan_new_array_nothrow(&data, size);
  }

  try {
    return operator new[](size);
  } catch (...) {
    return nullptr;
  }
}

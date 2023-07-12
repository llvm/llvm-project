//===-- asan_win_new_array_align_thunk.cc ---------------------------------===//
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

////////////////////////////////////////////////
// clang-format off
// Aligned new() Fallback Ordering
//
// +----------------+
// |new_scalar_align<--------------+
// +----^-----------+              |
//      |                          |
// +----+-------------------+  +---+-----------+
// |new_scalar_align_nothrow|  |NEW_ARRAY_ALIGN|
// +------------------------+  +---^-----------+
//                                 |
//                     +-----------+-----------+
//                     |new_array_align_nothrow|
//                     +-----------------------+
// clang-format on

__asan_InitDefine<op_new_array_align> init_new_array_align;

extern "C" void* __cdecl __asan_new_array_align(
    __asan_win_new_delete_data* data, size_t size, std::align_val_t align);

// Avoid tailcall optimization to preserve stack frame.
#pragma optimize("", off)
void* operator new[](size_t size, std::align_val_t align) {
  if (__asan_InitDefine<op_new_scalar_align>::defined) {
    __asan_win_new_delete_data data{};
    return __asan_new_array_align(&data, size, align);
  }

  return operator new(size, align);
}

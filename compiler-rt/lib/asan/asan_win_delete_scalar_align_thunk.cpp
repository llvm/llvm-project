//===-- asan_win_delete_scalar_align_thunk.cc -----------------------------===//
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

//////////////////////////////////////////////////////////////////////////////////
// clang-format off
// Aligned delete() Fallback Ordering
//
// +-------------------+
// |DELETE_SCALAR_ALIGN<----+---------------------------+
// +--^----------------+    |                           |
//    |                     |                           |
// +--+---------------+  +--+---------------------+  +--+------------------------+
// |delete_array_align|  |delete_scalar_size_align|  |delete_scalar_align_nothrow|
// +--^-----^---------+  +------------------------+  +---------------------------+
//    |     |
//    |     +------------------------+
//    |                              |
// +--+--------------------+  +------+-------------------+
// |delete_array_size_align|  |delete_array_align_nothrow|
// +-----------------------+  +--------------------------+
// clang-format on

__asan_InitDefine<op_delete_scalar_align> init_delete_scalar_align;

extern "C" void __cdecl __asan_delete_align(__asan_win_new_delete_data* data,
                                            void* ptr, std::align_val_t align);

// Avoid tailcall optimization to preserve stack frame.
#pragma optimize("", off)
void operator delete(void* ptr, std::align_val_t align) noexcept {
  __asan_win_new_delete_data data{};
  __asan_delete_align(&data, ptr, align);
}

//===-- asan_win_delete_scalar_align_nothrow_thunk.cc ---------------------===//
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
// |delete_scalar_align<----+---------------------------+
// +--^----------------+    |                           |
//    |                     |                           |
// +--+---------------+  +--+---------------------+  +--+------------------------+
// |delete_array_align|  |delete_scalar_size_align|  |DELETE_SCALAR_ALIGN_NOTHROW|
// +--^-----^---------+  +------------------------+  +---------------------------+
//    |     |
//    |     +------------------------+
//    |                              |
// +--+--------------------+  +------+-------------------+
// |delete_array_size_align|  |delete_array_align_nothrow|
// +-----------------------+  +--------------------------+
// clang-format on

// Avoid tailcall optimization to preserve stack frame.
#pragma optimize("", off)
void operator delete(void* ptr, std::align_val_t align,
                     std::nothrow_t const&) noexcept {
  // nothrow version is identical to throwing version
  operator delete(ptr, align);
}

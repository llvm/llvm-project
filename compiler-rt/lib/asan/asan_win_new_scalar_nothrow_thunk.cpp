//===-- asan_win_new_scalar_nothrow_thunk.cc ------------------------------===//
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

// see diagram in asan_win_new_delete_thunk_common.h for the ordering of the
// new/delete fallbacks.

// Avoid tailcall optimization to preserve stack frame.
#pragma optimize("", off)
void* operator new(size_t size, std::nothrow_t const&) noexcept {
  if (__asan_InitDefine<op_new_scalar>::defined) {
    return __asan_new_nothrow(size);
  }

  try {
    return operator new(size);
  } catch (...) {
    return nullptr;
  }
}

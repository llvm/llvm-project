//===-- asan_win_thunk_common.h  --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// Windows-specific common utilities to communicate between static and DLL
// portions of the ASAN runtime.
//
// This file must not include any core components of the ASAN runtime as it must
// be able to be included in the portions statically linked with the user
// program.
//
//===----------------------------------------------------------------------===//

#pragma once
#include <sanitizer_common/sanitizer_internal_defs.h>
namespace __sanitizer {

__declspec(noinline) inline __sanitizer::uptr __asan_GetCurrentPc() {
  return GET_CALLER_PC();
}

struct __asan_win_stack_data {
  // Put calls to get pc, bp, and caller_pc in ctor arguments so they occur in
  // the calling frame and don't rely on inlining to get correct stack
  // information.
  __forceinline explicit __asan_win_stack_data(
      __sanitizer::uptr pc_arg = __asan_GetCurrentPc(),
      __sanitizer::uptr bp_arg = GET_CURRENT_FRAME(),
      __sanitizer::uptr caller_pc_arg = GET_CALLER_PC())
      : size(sizeof(__asan_win_stack_data)),
        extra_context(2),  // TODO: Should only need one - investigate why we
                           // need an extra frame.
        pc(pc_arg),
        bp(bp_arg),
        caller_pc(caller_pc_arg) {}

  size_t size;        // Size of this struct (it travels over the DLL boundary).
  int extra_context;  // Number of extra frames we need to collect in the
                      // backtrace.
  __sanitizer::uptr pc;
  __sanitizer::uptr bp;
  __sanitizer::uptr caller_pc;
};

}  // namespace __sanitizer
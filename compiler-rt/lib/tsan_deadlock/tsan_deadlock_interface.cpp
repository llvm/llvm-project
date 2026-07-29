//===-- tsan_deadlock_interface.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of DeadlockSanitizer.
//
//===----------------------------------------------------------------------===//

#include "tsan_deadlock_rtl.h"

using namespace __tsan_deadlock;

extern "C" {

SANITIZER_INTERFACE_ATTRIBUTE void __tsan_init() { Initialize(); }

SANITIZER_INTERFACE_ATTRIBUTE void __tsan_func_entry(void *pc) {
  if (UNLIKELY(!thr_tls.is_inited))
    return;
  DCHECK_LT(thr_tls.shadow_stack_pos, thr_tls.shadow_stack_end);
  thr_tls.shadow_stack_pos[0] = (uptr)pc;
  thr_tls.shadow_stack_pos++;
}

SANITIZER_INTERFACE_ATTRIBUTE void __tsan_func_exit() {
  if (UNLIKELY(!thr_tls.is_inited))
    return;
  if (LIKELY(thr_tls.shadow_stack_pos > thr_tls.shadow_stack))
    thr_tls.shadow_stack_pos--;
}

} // extern "C"

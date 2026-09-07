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
  Thread *thr = cur_thread();
  if (UNLIKELY(!thr->is_inited))
    return;
  DCHECK_LT(thr->shadow_stack_pos, thr->shadow_stack_end);
  thr->shadow_stack_pos[0] = (uptr)pc;
  thr->shadow_stack_pos++;
}

SANITIZER_INTERFACE_ATTRIBUTE void __tsan_func_exit() {
  Thread *thr = cur_thread();
  if (UNLIKELY(!thr->is_inited))
    return;
  if (LIKELY(thr->shadow_stack_pos > thr->shadow_stack))
    thr->shadow_stack_pos--;
}

} // extern "C"

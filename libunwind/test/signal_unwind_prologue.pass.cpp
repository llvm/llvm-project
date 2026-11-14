// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Ensure that unwinding through a signal frame applies CFI instructions emitted
// immediately before the faulting instruction.
// REQUIRES: target={{(aarch64|x86_64)-.+}}
// UNSUPPORTED: target={{.*-windows.*}}, target={{.*-apple.*}}

// TODO: Figure out why this fails with Memory Sanitizer.
// XFAIL: msan

// Note: this test fails on musl because:
//
//  (a) musl disables emission of unwind information for its build, and
//  (b) musl's signal trampolines don't include unwind information
//
// XFAIL: target={{.*}}-musl

#undef NDEBUG
#include "support/func_bounds.h"
#include <assert.h>
#include <signal.h>
#include <stdint.h>
#include <stdlib.h>
#include <unwind.h>

FUNC_BOUNDS_DECL(main_func);

static _Unwind_Reason_Code frame_handler(struct _Unwind_Context *ctx,
                                         void *arg) {
  (void)arg;
  uintptr_t ip = _Unwind_GetIP(ctx);
  if (ip >= (uintptr_t)FUNC_START(main_func) &&
      ip < (uintptr_t)FUNC_END(main_func)) {
    _Exit(0);
  }
  return _URC_NO_REASON;
}

static void signal_handler(int) {
  _Unwind_Backtrace(frame_handler, nullptr);
  _Exit(-1);
}

__attribute__((naked)) static void fault_in_prologue(volatile int *) {
#if defined(__aarch64__)
  __asm__("hint #34\n" // bti c
          "hint #25\n" // paciasp
          ".cfi_window_save\n"
          "stp x29, x30, [sp, #-16]!\n"
          ".cfi_def_cfa_offset 16\n"
          ".cfi_offset x29, -16\n"
          ".cfi_offset x30, -8\n"
          "stp xzr, xzr, [sp, #-16]!\n"
          ".cfi_def_cfa_offset 32\n"
          "ldr w0, [x0]\n"
          "add sp, sp, #16\n"
          ".cfi_def_cfa_offset 16\n"
          "ldp x29, x30, [sp], #16\n"
          ".cfi_def_cfa_offset 0\n"
          ".cfi_restore x30\n"
          ".cfi_restore x29\n"
          "hint #29\n" // autiasp
          ".cfi_window_save\n"
          "ret\n");
#elif defined(__x86_64__)
  __asm__("pushq $0\n"
          ".cfi_def_cfa_offset 16\n"
          "pushq $0\n"
          ".cfi_def_cfa_offset 24\n"
          "movl (%rdi), %eax\n"
          "addq $16, %rsp\n"
          ".cfi_def_cfa_offset 8\n"
          "retq\n");
#endif
}

FUNC_ATTR(main_func) int main(int, char **) {
  struct sigaction act = {};
  act.sa_handler = signal_handler;
  sigemptyset(&act.sa_mask);
  assert(sigaction(SIGSEGV, &act, nullptr) == 0);
  volatile int *null_ptr = nullptr;
  fault_in_prologue(null_ptr);
  return -2;
}

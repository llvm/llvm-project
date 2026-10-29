// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Ensure that stepping out of a bottom-of-stack frame whose return-address
// register is marked DW_CFA_undefined stops immediately with UNW_STEP_END
// without evaluating CFA or dereferencing saved registers (matching libgcc).
// REQUIRES: target={{(aarch64|x86_64)-.+}}
// UNSUPPORTED: target={{.*-windows.*}}
// UNSUPPORTED: target={{.*-apple.*}}

// GCC doesn't support __attribute__((naked)) on AArch64.
// UNSUPPORTED: gcc

// Inline assembly is incompatible with MSAN.
// UNSUPPORTED: msan

#undef NDEBUG
#include <assert.h>
#include <libunwind.h>
#include <unwind.h>

static _Unwind_Reason_Code trace_callback(struct _Unwind_Context *, void *arg) {
  ++*static_cast<int *>(arg);
  return _URC_NO_REASON;
}

extern "C" void stepper() {
  unw_cursor_t cursor;
  unw_context_t uc;
  unw_getcontext(&uc);
  unw_init_local(&cursor, &uc);
  // Stepping from stepper() into entry_frame() should succeed.
  assert(unw_step(&cursor) > 0);
  // Stepping past entry_frame() (DW_CFA_undefined on return-address register)
  // must return UNW_STEP_END without dereferencing CFA / saved registers.
  assert(unw_step(&cursor) == 0);

  int frames = 0;
  assert(_Unwind_Backtrace(trace_callback, &frames) == _URC_END_OF_STACK);
  assert(frames == 2);
}

__attribute__((naked)) void entry_frame() {
#if defined(__aarch64__)
  __asm__("stp     x29, x30, [sp, #-16]!\n"
          "mov     x29, #0\n"
          ".cfi_def_cfa x29, 64\n"
          ".cfi_offset x19, -8\n"
          ".cfi_undefined x30\n"
          "bl      stepper\n"
          "ldp     x29, x30, [sp], #16\n"
          "ret\n");
#elif defined(__x86_64__)
  __asm__("pushq   %rbp\n"
          "xorl    %ebp, %ebp\n"
          ".cfi_def_cfa %rbp, 64\n"
          ".cfi_offset %rbx, -8\n"
          ".cfi_undefined %rip\n"
          "callq   stepper\n"
          "popq    %rbp\n"
          "ret\n");
#else
#error This test is only supported on aarch64 or x86-64
#endif
}

int main(int, char **) {
  entry_frame();
  return 0;
}

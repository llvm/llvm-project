// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: linux && target={{riscv64-.+}}

#undef NDEBUG
#include <assert.h>
#include <libunwind.h>

// Check correct unwinding of frame with VLENB-sized objects (vector registers):
// 1. Save return address (ra) in temporary register.
// 2. Load VLENB (vector length in bytes) and substract it from current stack
//    pointer (sp) - equivalent to one vector register on stack frame.
// 3. Set DWARF cannonical frame address (CFA) to "sp + vlenb" expresssion so it
//    can be correctly unwinded.
// 4. Call stepper() function and check that 2 unwind steps are successful -
//    from stepper() into foo() and from foo() into main().
// 5. Restore stack pointer and return address.
__attribute__((naked)) static void foo() {
  __asm__(".cfi_startproc\n"
          "mv s0, ra\n"
          "csrr  s1, vlenb\n"
          "sub sp, sp, s1\n"
          "# .cfi_def_cfa_expression sp + vlenb\n"
          ".cfi_escape 0x0f, 0x07, 0x72, 0x00, 0x92, 0xa2, 0x38, 0x00, 0x22\n"
          "call stepper\n"
          "add sp, sp, s1\n"
          "mv ra, s0\n"
          "ret\n"
          ".cfi_endproc\n");
}

extern "C" void stepper() {
  unw_cursor_t cursor;
  unw_context_t uc;
  unw_getcontext(&uc);
  unw_init_local(&cursor, &uc);
  // Stepping into foo() should succeed.
  assert(unw_step(&cursor) > 0);
  // Stepping past foo() should succeed, too.
  assert(unw_step(&cursor) > 0);
}

int main() { foo(); }

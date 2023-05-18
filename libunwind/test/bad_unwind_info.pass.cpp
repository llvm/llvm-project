// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Ensure that libunwind doesn't crash on invalid info; the Linux aarch64
// sigreturn frame check would previously attempt to access invalid memory in
// this scenario.
// REQUIRES: target={{(aarch64|s390x|x86_64)-.+linux.*}}

// GCC doesn't support __attribute__((naked)) on AArch64.
// UNSUPPORTED: gcc

// Inline assembly is incompatible with MSAN.
// UNSUPPORTED: msan

#undef NDEBUG
#include <assert.h>
#include <libunwind.h>
#include <stdio.h>

__attribute__((naked)) void bad_unwind_info() {
#if defined(__aarch64__)
  __asm__("// not using 0 because unwinder was already resilient to that\n"
          "mov     x8, #4\n"
          "stp     x30, x8, [sp, #-16]!\n"
          ".cfi_def_cfa_offset 16\n"
          "// purposely use incorrect offset for x30\n"
          ".cfi_offset x30, -8\n"
          "bl      stepper\n"
          "ldr     x30, [sp], #16\n"
          ".cfi_def_cfa_offset 0\n"
          ".cfi_restore x30\n"
          "ret\n");
#elif defined(__s390x__)
  __asm__("stmg    %r14,%r15,112(%r15)\n"
	  "mvghi   104(%r15),4\n"
          "# purposely use incorrect offset for %r14\n"
          ".cfi_offset 14, -56\n"
          ".cfi_offset 15, -40\n"
          "lay     %r15,-160(%r15)\n"
          ".cfi_def_cfa_offset 320\n"
          "brasl   %r14,stepper\n"
          "lmg     %r14,%r15,272(%r15)\n"
          ".cfi_restore 15\n"
          ".cfi_restore 14\n"
          ".cfi_def_cfa_offset 160\n"
          "br      %r14\n");
#elif defined(__x86_64__)
  __asm__("pushq   %rbx\n"
          ".cfi_def_cfa_offset 16\n"
          "movq    8(%rsp), %rbx\n"
          "# purposely corrupt return value on stack\n"
          "movq    $4, 8(%rsp)\n"
          "callq   stepper\n"
          "movq    %rbx, 8(%rsp)\n"
          "popq    %rbx\n"
          ".cfi_def_cfa_offset 8\n"
          "ret\n");
#else
#error This test is only supported on aarch64, s390x, or x86-64
#endif
}

extern "C" void stepper() {
  unw_cursor_t cursor;
  unw_context_t uc;
  unw_getcontext(&uc);
  unw_init_local(&cursor, &uc);
  // stepping to bad_unwind_info should succeed
  assert(unw_step(&cursor) > 0);
  // stepping past bad_unwind_info should fail but not crash
  assert(unw_step(&cursor) <= 0);
}

int main() { bad_unwind_info(); }

// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Ensure that evaluateExpression bounds checks its internal operand stack and
// aborts on malformed DWARF expressions rather than overflowing the stack.
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
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

extern "C" void stepper() {
  unw_cursor_t cursor;
  unw_context_t uc;
  unw_getcontext(&uc);
  unw_init_local(&cursor, &uc);
  // Step to bad_cfa_expression (frame 1).
  unw_step(&cursor);
  // Step past bad_cfa_expression (frame 2). This evaluates the CFA expression
  // in bad_cfa_expression and triggers _LIBUNWIND_ABORT due to stack overflow.
  unw_step(&cursor);
}

__attribute__((naked)) void bad_cfa_expression() {
#if defined(__aarch64__)
  __asm__(
      "stp     x29, x30, [sp, #-16]!\n"
      "mov     x29, sp\n"
      // DW_CFA_def_cfa_expression (0x0f), length 4, expression: DW_OP_dup (0x12), DW_OP_skip (0x2f) -4 (0xfc, 0xff)
      ".cfi_escape 0x0f, 0x04, 0x12, 0x2f, 0xfc, 0xff\n"
      "bl      stepper\n"
      "ldp     x29, x30, [sp], #16\n"
      "ret\n");
#elif defined(__x86_64__)
  __asm__(
      "pushq   %rbp\n"
      "movq    %rsp, %rbp\n"
      // DW_CFA_def_cfa_expression (0x0f), length 4, expression: DW_OP_dup (0x12), DW_OP_skip (0x2f) -4 (0xfc, 0xff)
      ".cfi_escape 0x0f, 0x04, 0x12, 0x2f, 0xfc, 0xff\n"
      "callq   stepper\n"
      "popq    %rbp\n"
      "ret\n");
#else
#error This test is only supported on aarch64 or x86-64
#endif
}

int main(int, char **) {
  pid_t pid = fork();
  assert(pid >= 0);
  if (pid == 0) {
    bad_cfa_expression();
    exit(0);
  }

  int status = 0;
  waitpid(pid, &status, 0);
  // The child process should abort due to operand stack bounds violation.
  assert(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT);
  return 0;
}

// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Ensure that a frame whose CFA is defined relative to DWARF register 0
// (%rax on x86-64) unwinds correctly. Register number 0 must not be mistaken
// for "no CFA register", which would make the unwinder evaluate a null CFA
// expression instead.

// REQUIRES: target={{x86_64-.+}}
// UNSUPPORTED: target={{.*-windows.*}}
// UNSUPPORTED: target={{.*-apple.*}}

// Inline assembly is incompatible with MSAN.
// UNSUPPORTED: msan

#undef NDEBUG
#include <assert.h>
#include <libunwind.h>
#include <stdint.h>

// Captured inside cfa_in_reg0, at a point where %rax holds the CFA.
extern "C" unw_context_t reg0_context;
unw_context_t reg0_context;

extern "C" void check_step() {
  uint64_t cfa = reinterpret_cast<uint64_t *>(&reg0_context)[0]; // saved %rax

  unw_cursor_t cursor;
  assert(unw_init_local(&cursor, &reg0_context) == UNW_ESUCCESS);
  assert(unw_step(&cursor) > 0);

  unw_word_t sp, ip;
  assert(unw_get_reg(&cursor, UNW_REG_SP, &sp) == UNW_ESUCCESS);
  assert(unw_get_reg(&cursor, UNW_REG_IP, &ip) == UNW_ESUCCESS);
  assert(sp == cfa);
  assert(ip == *reinterpret_cast<uint64_t *>(cfa - 8));
}

__attribute__((naked)) void cfa_in_reg0() {
  __asm__("subq    $8, %rsp\n"
          ".cfi_def_cfa_offset 16\n"
          "leaq    16(%rsp), %rax\n"
          ".cfi_def_cfa %rax, 0\n"
          "leaq    reg0_context(%rip), %rdi\n"
          "callq   unw_getcontext@PLT\n"
          "callq   check_step\n"
          "addq    $8, %rsp\n"
          ".cfi_def_cfa %rsp, 8\n"
          "retq\n");
}

int main(int, char **) {
  cfa_in_reg0();
  return 0;
}

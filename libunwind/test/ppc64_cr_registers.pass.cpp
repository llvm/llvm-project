//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: target={{powerpc64(le)?-.+}}

// Check that stepping out of a frame that saved the condition register
// restores the caller's cr2 field. The ELFv2 ABI saves CR as a 32-bit word at
// offset 8 of the caller's stack frame (DW_CFA_offset cr2 at CFA+8); reading
// that slot as a 64-bit register returned the saved word in the upper half on
// big-endian targets, so cr2-cr4 of the caller were restored from stale stack
// bytes.

#include <libunwind.h>
#include <stdint.h>
#include <stdlib.h>

#define CR2_MASK 0x00F00000u

static unw_word_t expected_cr2;

__attribute__((noinline)) static void check_caller_cr2() {
  unw_context_t context;
  unw_cursor_t cursor;
  unw_word_t cr2;

  unw_getcontext(&context);
  unw_init_local(&cursor, &context);

  // Step out of this frame into middle(), then out of middle() into outer().
  // The second step restores cr2 from the slot middle() saved it in.
  if (unw_step(&cursor) <= 0)
    abort();
  if (unw_step(&cursor) <= 0)
    abort();
  if (unw_get_reg(&cursor, UNW_PPC64_CR2, &cr2) != UNW_ESUCCESS)
    abort();
  if ((cr2 & CR2_MASK) != expected_cr2)
    abort();
}

__attribute__((noinline)) static void middle(long a, long b) {
  // Clobbering the callee-saved cr2 forces this frame to save CR in its
  // prologue and describe the save slot in its CFI.
  __asm__ volatile("cmpd cr2, %0, %1" : : "r"(a), "r"(b) : "cr2");
  check_caller_cr2();
}

__attribute__((noinline)) static void outer(long a, long b) {
  uint32_t cr;
  // Put a known value into cr2 right before the call; the ABI requires
  // middle() to preserve it.
  __asm__ volatile("cmpd cr2, %1, %2\n\tmfcr %0"
                   : "=r"(cr)
                   : "r"(a), "r"(b)
                   : "cr2");
  expected_cr2 = cr & CR2_MASK;
  middle(b, a);
}

int main(int, char **) {
  outer(1, 2); // cr2 = LT
  outer(2, 1); // cr2 = GT
  outer(3, 3); // cr2 = EQ
  return 0;
}

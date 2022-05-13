// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: linux && target={{aarch64-.+}}

// This test ensures the .cfi_negate_ra_state the RA_SIGN_STATE pseudo register
// could be set directly set by a DWARF expression and the unwinder handles it
// correctly. The two directives can't be mixed in one CIE/FDE sqeuence.

#include <stdlib.h>

__attribute__((noinline, target("branch-protection=pac-ret+leaf")))
void bar() {
  // ".cfi_negate_ra_state" is emitted by the compiler.
  throw 1;
}

__attribute__((noinline, target("branch-protection=none")))
void foo() {
  // Here a DWARF expression sets RA_SIGN_STATE.
  // The LR is signed manually and stored on the stack.
  asm volatile(
      ".cfi_escape 0x16,"    // DW_CFA_val_expression
                    "34,"    // REG_34(RA_SIGN_STATE)
                     "1,"    // expression_length(1)
                    "0x31\n" // DW_OP_lit1
      "add sp, sp, 16\n"     // Restore SP's value before the stack frame is
                             // created.
      "paciasp\n"            // Sign the LR.
      "str lr, [sp, -0x8]\n" // Overwrite LR on the stack.
      "sub sp, sp, 16\n"     // Restore SP's value.
  );
  bar();
  _Exit(-1);
}

__attribute__((noinline, target("branch-protection=pac-ret")))
void bazz() {
  // ".cfi_negate_ra_state" is emitted by the compiler.
  try {
    foo();
  } catch (int i) {
    if (i == 1)
      throw i;
    throw 2;
  }
}

int main() {
  try {
    bazz();
  } catch (int i) {
    if (i == 1)
      _Exit(0);
  }
  return -1;
}

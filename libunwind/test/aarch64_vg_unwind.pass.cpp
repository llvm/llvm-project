//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: linux && target={{aarch64-.+}}

#include <libunwind.h>
#include <stdlib.h>
#include <string.h>

// Basic test of VG (Vector Granule) unwinding. This is meant to mimic SVE/SME
// unwind info without requiring those features for this test.

#define VG_REGNUM 46

__attribute__((noinline)) void baz() {
  // The previous value of VG is 2
  asm(".cfi_escape 0x16, 0x2e, 0x01, 0x32");

  unw_context_t context;
  unw_cursor_t cursor;
  unw_getcontext(&context);
  unw_init_local(&cursor, &context);

  // Note: At this point VG is not defined (until we unw_step).

  uint16_t expected_vgs[]{/*qux*/ 2, /*bar*/ 2, /*foo*/ 8, /*main*/ 2};
  for (uint16_t expected_vg : expected_vgs) {
    unw_step(&cursor);
    unw_word_t vg;
    unw_get_reg(&cursor, VG_REGNUM, &vg);
    if (vg != expected_vg)
      exit(1);
  }
  exit(0);
}

__attribute__((noinline)) void qux() { baz(); }

__attribute__((noinline)) void bar() {
  // The previous value of VG is 8
  asm(".cfi_escape 0x16, 0x2e, 0x01, 0x38");
  // The previous value of W21 is VG (used to force an evaluation of VG).
  asm(".cfi_escape 0x16, 0x15, 0x03, 0x92, 0x2e, 0x00");

  // smstop sm
  qux();
  // smstart sm
}
__attribute__((noinline)) void foo() {
  // The previous value of VG is 2
  asm(".cfi_escape 0x16, 0x2e, 0x01, 0x32");
  // The previous value of W21 is VG (used to force an evaluation of VG).
  asm(".cfi_escape 0x16, 0x15, 0x03, 0x92, 0x2e, 0x00");

  // smstart sm
  bar();
  // smstop sm
}

int main(int, char **) {
  foo();
  return 0;
}

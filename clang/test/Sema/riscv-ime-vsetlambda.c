// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv32 -target-feature +v \
// RUN:   -target-feature +experimental-zvvmm -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple riscv64 -target-feature +v \
// RUN:   -target-feature +experimental-zvvmm -fsyntax-only -verify %s

#include <stddef.h>
#include <riscv_vector.h>

void ok(void) {
  __riscv_vsetlambda(0);
  __riscv_vsetlambda(1);
  __riscv_vsetlambda(2);
  __riscv_vsetlambda(4);
  __riscv_vsetlambda(8);
  __riscv_vsetlambda(16);
  __riscv_vsetlambda(32);
  __riscv_vsetlambda(64);
}

void bad_value(void) {
  __riscv_vsetlambda(3);   // expected-error {{constant argument to RISC-V IME vsetlambda builtin must be 0 or a power of two in the range [1, 64]}}
  __riscv_vsetlambda(128); // expected-error {{constant argument to RISC-V IME vsetlambda builtin must be 0 or a power of two in the range [1, 64]}}
  __riscv_vsetlambda(-1);  // expected-error {{constant argument to RISC-V IME vsetlambda builtin must be 0 or a power of two in the range [1, 64]}}
}

void ok_runtime(size_t x) {
  __riscv_vsetlambda(x);
  __riscv_vsetlambda(x++);
}

void bad_wrap(void) {
  __riscv_vsetlambda(0x100000004ULL);       // expected-error {{constant argument to RISC-V IME vsetlambda builtin must be 0 or a power of two in the range [1, 64]}}
  __riscv_vsetlambda(-4294967292LL);        // expected-error {{constant argument to RISC-V IME vsetlambda builtin must be 0 or a power of two in the range [1, 64]}}
#if __SIZEOF_POINTER__ == 8
  __riscv_vsetlambda(((__int128)1) << 70);  // expected-error {{constant argument to RISC-V IME vsetlambda builtin must be 0 or a power of two in the range [1, 64]}}
#endif
}

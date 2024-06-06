// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +zalrsc -S -verify %s -o -
// RUN: %clang_cc1 -triple riscv64 -target-feature +zalrsc -S -verify %s -o -

#include <riscv_atomics.h>

int zalrsc_lr_w(int* ptr) {
  return __riscv_lr_w(ptr, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
}

int zalrsc_sc_w(int v, int* ptr) {
  return __riscv_sc_w(v, ptr, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
}

// REQUIRES: x86-registered-target && aarch64-registered-target
// RUN: %clang --target=x86_64-unknown-elf -c -O2 -emit-llvm %S/external_library.c -o %t.rlib.bc
// RUN: %clang --target=aarch64-linux-gnu -O2 -fenable-ripple -emit-llvm -S -o - -fripple-lib %t.rlib.bc %s 2> %t; FileCheck %s --input-file %t

#include "external_library.h"
#include <stddef.h>
#include <ripple.h>

#define VEC 0

// CHECK: the Ripple external library datalayout mismatches with the current target datalayout; ignoring

void test(float *f) {
  ripple_block_t BS = ripple_set_block_shape(VEC, 64);
  size_t BlockX = ripple_id(BS, 0);
  size_t BlockSizeX = ripple_get_block_size(BS, 0);
  f[BlockX] += 1.f;
}

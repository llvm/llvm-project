// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -O2 -fenable-ripple -emit-llvm -S -o - -fripple-lib %S/wrong-format-lib %s 2> %t; FileCheck %s --input-file %t

#include "external_library.h"
#include <stddef.h>
#include <ripple.h>

#define VEC 0

// CHECK: the Ripple external library '{{.*}}/clang/test/Ripple/external-ripple-lib/wrong-format-lib' is neither a module nor LLVM bitcode. Module load error: Invalid bitcode signature; Bitcode load error: expected top-level entity

void test(float *f) {
  ripple_block_t BS = ripple_set_block_shape(VEC, 64);
  size_t BlockX = ripple_id(BS, 0);
  size_t BlockSizeX = ripple_get_block_size(BS, 0);
  f[BlockX] += 1.f;
}

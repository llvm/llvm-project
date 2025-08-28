// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -g -S -fenable-ripple -O0 -emit-llvm %s -o - 2>&1 | FileCheck %s --implicit-check-not="warning:" --implicit-check-not="error:"
// RUN: %clang -g -S -fenable-ripple -O1 -emit-llvm %s -o - 2>&1 | FileCheck %s --implicit-check-not="warning:" --implicit-check-not="error:"
// RUN: %clang -g -S -fenable-ripple -O2 -emit-llvm %s -o - 2>&1 | FileCheck %s --implicit-check-not="warning:" --implicit-check-not="error:"
// RUN: %clang -g -S -fenable-ripple -O3 -emit-llvm %s -o - 2>&1 | FileCheck %s --implicit-check-not="warning:" --implicit-check-not="error:"
// RUN: %clang -g -S -fenable-ripple -Os -emit-llvm %s -o - 2>&1 | FileCheck %s --implicit-check-not="warning:" --implicit-check-not="error:"
// RUN: %clang -g -S -fenable-ripple -Oz -emit-llvm %s -o - 2>&1 | FileCheck %s --implicit-check-not="warning:" --implicit-check-not="error:"

#include <ripple.h>

void check(float *Input, float *Output) {
  Output[ripple_id(ripple_set_block_shape(0, 32, 2048), 0) +
         ripple_get_block_size(ripple_set_block_shape(0, 32), 0) *
             ripple_id(
                 ripple_set_block_shape(0, /*it's over 9000!*/ 9001, 2, 4096),
                 1)] = Input[ripple_id(ripple_set_block_shape(0, 32), 0)];
}

// CHECK: load <32 x float>
// CHECK: store <64 x float>

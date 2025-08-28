// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -g -S -fenable-ripple -O0 -emit-llvm %s -o - 2>&1
// RUN: %clang -g -S -fenable-ripple -O1 -emit-llvm %s -o - 2>&1
// RUN: %clang -g -S -fenable-ripple -O2 -emit-llvm %s -o - 2>&1
// RUN: %clang -g -S -fenable-ripple -O3 -emit-llvm %s -o - 2>&1
// RUN: %clang -g -S -fenable-ripple -Os -emit-llvm %s -o - 2>&1
// RUN: %clang -g -S -fenable-ripple -Oz -emit-llvm %s -o - 2>&1

#include <ripple.h>

void valid_struct_copy(size_t ArraySize, size_t *Output) {
  ripple_block_t BS = ripple_set_block_shape(0, 4, 8);
  size_t BlockIdx1 = ripple_id(BS, 0);
  size_t BlockIdx2 = ripple_id(BS, 1);

  // True 1D tensors
  size_t a1D = 32 / BlockIdx1;
  size_t b1D = 42 / BlockIdx1;
  // These are linear series since a1D and b1D have the same base shape
  size_t canLSThrough = a1D + b1D;
  // Below canLSThrough is used as base for another LS and instantiated because of the division
  size_t canLSThrough2 = canLSThrough + BlockIdx2;
  size_t cannotLSThrough = BlockIdx1 / canLSThrough;

  *Output = ripple_reduceadd(0b11, canLSThrough2 + cannotLSThrough);

}

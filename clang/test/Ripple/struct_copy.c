// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -g -S -fenable-ripple -O0 -emit-llvm %s -o - 2>&1
// RUN: %clang -g -S -fenable-ripple -O1 -emit-llvm %s -o - 2>&1
// RUN: %clang -g -S -fenable-ripple -O2 -emit-llvm %s -o - 2>&1
// RUN: %clang -g -S -fenable-ripple -O3 -emit-llvm %s -o - 2>&1
// RUN: %clang -g -S -fenable-ripple -Os -emit-llvm %s -o - 2>&1
// RUN: %clang -g -S -fenable-ripple -Oz -emit-llvm %s -o - 2>&1

#include <ripple.h>

struct Ure {
  signed p;
  char q;
  int r;
};

void valid_struct_copy(size_t ArraySize, struct Ure *Output) {
  ripple_block_t BS = ripple_set_block_shape(0, 4);
  unsigned BlockIdx = ripple_id(BS, 0);
  unsigned BlockSize = ripple_get_block_size(BS, 0);
  const struct Ure S = {.p = 32, .q = 'a' , .r = 42};
  for (size_t Idx = 0; Idx < ArraySize; Idx += BlockSize)
    Output[Idx + BlockIdx] = S;
}

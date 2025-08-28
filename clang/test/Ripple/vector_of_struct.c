// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// %clang -g -S -fenable-ripple -O0 -emit-llvm %s -o - 2>&1
// RUN: %clang -g -S -fenable-ripple -O1 -emit-llvm %s
// RUN: %clang -g -S -fenable-ripple -O2 -emit-llvm %s
// RUN: %clang -g -S -fenable-ripple -O3 -emit-llvm %s
// RUN: %clang -g -S -fenable-ripple -Os -emit-llvm %s
// RUN: %clang -g -S -fenable-ripple -Oz -emit-llvm %s

#include <ripple.h>

struct Ure {
  int p;
  int q;
};

struct Ure processStruct(struct Ure s) {
  s.p++;
  s.q--;
  return s;
}

void invalid_struct_vector(size_t ArraySize, struct Ure *Input, struct Ure *Output) {
  ripple_block_t BS = ripple_set_block_shape(0, 4);
  unsigned BlockIdx = ripple_id(BS, 0);
  unsigned BlockSize = ripple_get_block_size(BS, 0);
  for (size_t Idx = 0; Idx < ArraySize; Idx += BlockSize) {
    struct Ure Tmp = Input[Idx + BlockIdx];
    Tmp.p += 4;
    struct Ure Tmp2 = processStruct(Tmp);
    Tmp2.q += 3;
    Output[Idx + BlockIdx] = Tmp2;
  }
}

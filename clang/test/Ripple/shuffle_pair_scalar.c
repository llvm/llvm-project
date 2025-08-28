// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -g -S -fenable-ripple -O2 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang -x c++ -g -S -fenable-ripple -O2 -emit-llvm %s -o - | FileCheck %s

#include <ripple.h>

size_t indexingFunLHS(size_t BlockIndex, size_t BlockSize) { return 0; }

size_t indexingFunRHS(size_t BlockIndex, size_t BlockSize) { return 1; }

void testScalarShufflepair(int *OutputLHS, int *OutputRHS, int *Dummy) {
  ripple_block_t BS = ripple_set_block_shape(0, 4);
  *Dummy = ripple_reduceadd(0x1, ripple_id(BS, 0));

  int LHSExpect = 42;
  int RHSExpect = 52;

  // CHECK: testScalarShufflepair
  // CHECK: store i{{[0-9]+}} 42, ptr %OutputLHS
  // CHECK: store i{{[0-9]+}} 52, ptr %OutputRHS
  *OutputLHS = ripple_shuffle_pair(LHSExpect, RHSExpect, indexingFunLHS);
  *OutputRHS = ripple_shuffle_pair(LHSExpect, RHSExpect, indexingFunRHS);
}

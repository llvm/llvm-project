// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -Wall -Wextra -Wripple -Xclang -disable-llvm-passes -S -emit-llvm -fenable-ripple %s -o - | FileCheck %s --implicit-check-not="warning:"

#include <ripple.h>

// CHECK: @test1
// CHECK: for.body:
// This load is for the parallel_i -> i rename
// CHECK: load i{{[0-9]+}}, ptr %ripple.par.iv
// CHECK: %[[ParallelIdx:[0-9A-Za-z_]+]] = load i{{[0-9]+}}, ptr %ripple.par.iv
// CHECK: store i{{[0-9]+}} %[[ParallelIdx]], ptr %parallel_idx
// CHECK: ripple.par.for.remainder.body:
// The 'i' update is part of the condition, computed in the previous BB
// CHECK: %[[ParallelIdx:[0-9A-Za-z_]+]] = load i{{[0-9]+}}, ptr %ripple.par.iv
// CHECK: store i{{[0-9]+}} %[[ParallelIdx]], ptr %parallel_idx
void test1(int N, int start, int end, float x[restrict N], float y[restrict N],
           float xpy[restrict N]) {
  ripple_block_t BS = ripple_set_block_shape(0, 32);
  ripple_parallel(BS, 0);
  for (int i = start; i < end; ++i) {
    int parallel_idx = ripple_parallel_idx(BS, 0);
    xpy[i] = (x[i] + y[i]) * (float)parallel_idx;
  }
}

// CHECK: @test2
// Inside the two parallel for loops
// CHECK: ripple.par.for.begin:
// CHECK: ripple.par.for.begin{{[0-9]*}}:
// CHECK: for.body{{[0-9]*}}:
// This load is for the parallel_j -> j rename
// CHECK: load i{{[0-9]+}}, ptr %ripple.par.iv
// CHECK: %[[ParallelIdx:[0-9A-Za-z_]+]] = load i{{[0-9]+}}, ptr %ripple.par.iv
// CHECK: store i{{[0-9]+}} %[[ParallelIdx]], ptr %parallel_idx_i
// CHECK: ripple.par.for.remainder.body:
// The 'i' update is part of the condition, computed in the previous BB
// CHECK: %[[ParallelIdx:[0-9A-Za-z_]+]] = load i{{[0-9]+}}, ptr %ripple.par.iv{{[0-9]+}}
// CHECK: store i{{[0-9]+}} %[[ParallelIdx]], ptr %parallel_idx_j
void test2(int N, int start, int end, float x[restrict N], float y[restrict N],
           float xpy[restrict N]) {
  ripple_block_t BS = ripple_set_block_shape(0, 4, 8);
  ripple_parallel(BS, 0);
  for (int i = start; i < end; ++i) {
    ripple_parallel(BS, 1);
    for (int j = start; j < end; ++j) {
      int parallel_idx_i = ripple_parallel_idx(BS, 0);
      int parallel_idx_j = ripple_parallel_idx(BS, 1);
      xpy[i] = (x[i + j] + y[i + j]) * parallel_idx_i * parallel_idx_j;
    }
  }
}

// CHECK: @test3
// CHECK: for.body:
// This load is for the parallel_i -> i rename
// CHECK: load i{{[0-9]+}}, ptr %ripple.par.iv
// CHECK: %[[ParallelIdx:[0-9A-Za-z_]+]] = load i{{[0-9]+}}, ptr %ripple.par.iv
// CHECK: store i{{[0-9]+}} %[[ParallelIdx]], ptr %parallel_idx
// CHECK: ripple.par.for.remainder.body:
// The 'i' update is part of the condition, computed in the previous BB
// CHECK: %[[ParallelIdx:[0-9A-Za-z_]+]] = load i{{[0-9]+}}, ptr %ripple.par.iv
// CHECK: store i{{[0-9]+}} %[[ParallelIdx]], ptr %parallel_idx
void test3(int N, int start, int end, float x[restrict N], float y[restrict N],
           float xpy[restrict N]) {
  ripple_block_t BS = ripple_set_block_shape(0, 4, 8);
  ripple_parallel(BS, 0, 1);
  for (unsigned i = start; i < end; ++i) {
    int parallel_idx_i = ripple_parallel_idx(BS, 0, 1);
    xpy[i] = (x[i] + y[i]) * parallel_idx_i;
  }
}

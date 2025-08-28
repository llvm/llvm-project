// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -g -Wall -Wextra -Wripple -Xclang -disable-llvm-passes -S -emit-llvm -fenable-ripple %s 2> %t; FileCheck %s --input-file %t

#include <ripple.h>

void test1(int start, int end) {
  ripple_block_t BS = ripple_set_block_shape(0, 4, 8);
  ripple_parallel_full(BS, 0);
  for (int i = start; i < end; ++i) {
    int parallel_idx_i = ripple_parallel_idx(BS, 0);

    // CHECK: parallel-idx-invalid.c:15{{.*}}not within the scope of a ripple_parallel with matching block shape (Block) and dimensions (Dims)


    int err0 = ripple_parallel_idx(BS, 1);
    // CHECK: parallel-idx-invalid.c:19{{.*}}not within the scope of a ripple_parallel with matching block shape (Block) and dimensions (Dims)


    int err1 = ripple_parallel_idx(BS, -1);
    ripple_parallel_full(BS, 1);
    for (int j = start; j < end; ++j) {
      int parallel_idx_i_snd = ripple_parallel_idx(BS, 0);
      int parallel_idx_j = ripple_parallel_idx(BS, 1);
      // CHECK: parallel-idx-invalid.c:27{{.*}}not within the scope of a ripple_parallel with matching block shape (Block) and dimensions (Dims)


      int err2 = ripple_parallel_idx(BS, 2);
    }
  }
  ripple_parallel_full(BS, 0, 1);
  for (int i = start; i < end; ++i) {
    // CHECK: parallel-idx-invalid.c:35:{{.*}}not within the scope of a ripple_parallel with matching block shape (Block) and dimensions (Dims)


    int err1 = ripple_parallel_idx(BS, 1, 0);
  }
}

void test2(int N, int start, int end, float x[restrict N],
           float y[restrict N], float xpy[restrict N]) {
  ripple_block_t BS = ripple_set_block_shape(0, 32);
  ripple_parallel(BS, 0);
  for (int i = start; i < end; ++i) {

    // CHECK: parallel-idx-invalid.c:48{{.*}}3rd argument must be a scalar integer type (was 'double')


    int err0 = ripple_parallel_idx(BS, 1, 32.f);
    // CHECK: parallel-idx-invalid.c:52{{.*}}passing 'float' to parameter of incompatible type 'void *'


    int err1 = ripple_parallel_idx(0.f, 1);
    // CHECK: parallel-idx-invalid.c:56{{.*}}too few arguments to function call, expected at least 2, have 1


    int err2 = ripple_parallel_idx(BS);

    // CHECK: parallel-idx-invalid.c:61{{.*}}argument to '__builtin_ripple_parallel_idx' must be a constant integer


    int err3 = ripple_parallel_idx(BS, N);
  }
}

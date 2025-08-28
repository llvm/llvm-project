// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -S -fenable-ripple -O2 -emit-llvm %s
// RUN: %clang -x c++ -S -fenable-ripple -O2 -emit-llvm %s

#include <ripple.h>
#define N 1024
#define VECTOR_LANE 0

// Extracts the 2nd column of a 32 x 4 block
void extract_2nd_row_from_32x4_blocks(uint16_t input[N], uint16_t output[N/4]) {
  ripple_block_t BS = ripple_set_block_shape(VECTOR_LANE, 32, 4); // set up a 32 x 4 block shape
  size_t v0 = ripple_id(BS, 0);
  size_t v1 = ripple_id(BS, 1);
  size_t block_size0 = ripple_get_block_size(BS, 0);
  size_t block_size1 = ripple_get_block_size(BS, 1);
  size_t block_size = block_size1 * block_size0;

  size_t n_blocks = N / block_size;
  for (size_t block_idx = 0; block_idx < n_blocks; ++block_idx) {
    // coalesced load into a 2d block
    uint16_t block = input[block_size * block_idx + block_size0 * v1 + v0];
    uint16_t extracted = ripple_slice(block, 2, -1);
    // CHECK:   %{{[0-9a-zA-Z.]+}} = shufflevector <128 x i16> %{{[0-9a-zA-Z.]+}}, <128 x i16> poison, <4 x i32> <i32 2, i32 34, i32 66, i32 98>
    // coalesced store of the 1-d block
    output[block_size1 * block_idx + v1] = extracted;
  }
}

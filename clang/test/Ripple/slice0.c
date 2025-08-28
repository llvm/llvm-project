// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -S -fenable-ripple -O2 -emit-llvm %s
// RUN: %clang -x c++ -S -fenable-ripple -O2 -emit-llvm %s

#include <ripple.h>
#define N 1024
#define VECTOR_LANE 0

// We're testing he following use of ripple_extract should go smoothly through Ripple,
// in particular it should work dimension-wise
void extract_2nd_row_from_32x4_blocks(float input[N], float output[N/4]) {
  ripple_block_t BS = ripple_set_block_shape(VECTOR_LANE, 32, 4); // set up a 32 x 4 block shape
  size_t v0 = ripple_id(BS, 0);
  size_t v1 = ripple_id(BS, 1);
  size_t block_size0 = ripple_get_block_size(BS, 0);
  size_t block_size = block_size0 * ripple_get_block_size(BS, 1);

  size_t n_blocks = N / block_size;
  for (size_t block_idx = 0; block_idx < n_blocks; ++block_idx) {
    // coalesced load into a 2d block
    float block = input[block_size * block_idx + block_size0 * v1 + v0];
    float extracted = ripple_slice(block, -1, 2);
    // CHECK:   %ripple.slice = shufflevector <128 x float> %{{[0-9]+}}, <128 x float> poison, <32 x i32> <i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79, i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86, i32 87, i32 88, i32 89, i32 90, i32 91, i32 92, i32 93, i32 94, i32 95>
    // coalesced store of the 1-d block
    output[block_size0 * block_idx + v0] = extracted;
  }
}

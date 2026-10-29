// REQUIRES: __opencl_c_work_group_collective_functions
// RUN: %libclc-compile-and-run --kernel test --threads-x 64 %t

#include "conformance.h"

__kernel void test(void) {
  uint lid = get_local_id(0);
  uint n = get_local_size(0);
  uint sum = work_group_reduce_add(lid);
  if (lid == 0)
    CHECK_EQ(sum, n * (n - 1) / 2);
}

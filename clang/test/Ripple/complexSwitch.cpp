// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -g -S -fenable-ripple -O2 -emit-llvm %s -o - 2>&1

#include <ripple.h>

#define VectorPE 0

extern "C" void ripple_function(size_t size, float*A, float* B, float*C) {
    auto BS = ripple_set_block_shape(VectorPE, 42);
    size_t VecIdx = ripple_id(BS, 0);
    size_t VecSize = ripple_get_block_size(BS, 0);
    size_t i = 0;
    switch (VecIdx)
    {
    case 1:
      A[0] += 2;
      if (size > 1000) {
        for (; i < size; ++i) {
          A[i]++;
          if (i + size > 32)
            break;
          continue;
    case 3:
          i = 3000;
    case 8:
          i += 42;
          A[0] += 4000;
        }
      }
    [[fallthrough]];
    default:
      for (i = 0; i < size; ++i)
        B[i]++;
    case 4:
      if (size < 500)
        C[size-1] = 0.f;
      C[0] = 42.f;
      break;
    case 5:
      for (i = 0; i < size; ++i)
        C[i]++;
    break;
    }
}

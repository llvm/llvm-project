// RUN: %clang_csan %s -o %t && %run --threads 64 --blocks 64 %t 2>&1 | count 0

#include <gpuintrin.h>

[[clang::loader_uninitialized]] static __gpu_local int shared[64];

int main(void) {
  unsigned slot = __gpu_thread_id(__GPU_X_DIM) % 64;
  for (int i = 0; i < 1024; ++i)
    shared[slot] += i;
  return 0;
}

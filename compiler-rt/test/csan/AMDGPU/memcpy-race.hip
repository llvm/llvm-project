// RUN: %clang_hip_csan %s -o %t %hip_libs
// RUN: %run %t 2>&1 | FileCheck %s

#include "race.h"

__device__ volatile unsigned Len = 32;
__device__ int Dst[64];
__device__ int Src[64];

__global__ void kernel() {
  RACE_UNTIL_FOUND(i)
    __builtin_memcpy((void *)Dst, (const void *)Src, Len);
}

int main() {
  kernel<<<64, 64>>>();
  CHECK_HIP(hipDeviceSynchronize());
  return 0;
}

// CHECK: WARNING: ConcurrencySanitizer: {{.*}}race
// CHECK: #0 {{.*}}memcpy-race.hip:{{[0-9]+}}

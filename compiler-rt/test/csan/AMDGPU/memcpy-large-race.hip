// RUN: %clang_hip_csan %s -o %t %hip_libs
// RUN: %run %t 2>&1 | FileCheck %s

#include "race.h"

#define N 512

__device__ int Dst[N];
__device__ int Src[N];

__global__ void kernel() {
  unsigned Id = CSAN_TID();
  RACE_UNTIL_FOUND(i) {
    if (Id == 0)
      __builtin_memcpy((void *)Dst, (const void *)Src, sizeof(Dst));
    else
      Dst[N - 1] = Id;
  }
}

int main() {
  kernel<<<64, 64>>>();
  CHECK_HIP(hipDeviceSynchronize());
  return 0;
}

// CHECK: WARNING: ConcurrencySanitizer: {{.*}}race
// CHECK: #0 {{.*}}memcpy-large-race.hip:{{[0-9]+}}

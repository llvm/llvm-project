// RUN: %clang_hip_csan -O2 %s -o %t %hip_libs
// RUN: %run %t 2>&1 | FileCheck %s

#include "race.h"

__device__ int Global;

__global__ void kernel() {
  unsigned Id = CSAN_TID();
  RACE_UNTIL_FOUND(i) {
    if (Id & 1)
      __atomic_fetch_add(&Global, 1, __ATOMIC_RELAXED);
    else
      Global = i;
  }
}

int main() {
  kernel<<<64, 1>>>();
  CHECK_HIP(hipDeviceSynchronize());
  return 0;
}

// CHECK: WARNING: ConcurrencySanitizer: {{.*}}race
// CHECK: #0 {{.*}}atomic-nonatomic.hip:{{[0-9]+}}
// CHECK: Location is global '{{.*}}Global{{.*}}' of size {{[0-9]+}} at 0x{{.*}}

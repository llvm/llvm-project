// RUN: %clang_hip_csan -O2 %s -o %t %hip_libs
// RUN: %run %t 2>&1 | FileCheck %s

#include "race.h"

__device__ volatile int Global;

__global__ void kernel() {
  unsigned Id = CSAN_TID();
  int Sink = 0;
  RACE_UNTIL_FOUND(i) {
    if (Id & 1) {
      Global = 1;
      Global = 0;
    } else {
      Sink += Global;
    }
  }
  if (Sink == -1)
    __builtin_trap();
}

int main() {
  kernel<<<64, 1>>>();
  CHECK_HIP(hipDeviceSynchronize());
  return 0;
}

// CHECK: WARNING: ConcurrencySanitizer: data race
// CHECK: {{Read|Write}} of size 4
// CHECK: Previous {{read|write}} of size 4
// CHECK: #0 {{.*}}aba-race.hip:{{[0-9]+}}
// CHECK: Location is global '{{.*}}Global{{.*}}' of size {{[0-9]+}} at 0x{{.*}}

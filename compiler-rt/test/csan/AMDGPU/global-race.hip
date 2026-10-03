// RUN: %clang_hip_csan %s -o %t %hip_libs
// RUN: %run %t 2>&1 | FileCheck %s
// RUN: %clang_hip_csan -O2 %s -o %t %hip_libs
// RUN: %run %t 2>&1 | FileCheck %s
// RUN: %clang_hip_csan -shared-libsan %s -o %t %hip_libs
// RUN: %run %t 2>&1 | FileCheck %s

// The host has no references to the shared runtime, it must still be loaded.
// RUN: %clang_hip -Xarch_device -fsanitize=concurrency -shared-libsan %s \
// RUN:   -o %t %hip_libs
// RUN: llvm-readelf -d %t | FileCheck %s --check-prefix=NEEDED
// RUN: %run %t 2>&1 | FileCheck %s
// NEEDED: libclang_rt.csan.so

#include "race.h"

__device__ volatile int global;

__global__ void kernel() {
  RACE_UNTIL_FOUND(i)
    global++;
}

int main() {
  kernel<<<64, 64>>>();
  CHECK_HIP(hipDeviceSynchronize());
  return 0;
}

// CHECK: WARNING: ConcurrencySanitizer: {{.*}}race
// CHECK: #0 {{.*}}global-race.hip:{{[0-9]+}}
// CHECK: Location is global '{{.*}}global{{.*}}' of size {{[0-9]+}} at 0x{{.*}}

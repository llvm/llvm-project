// RUN: %clang_hip_csan %s -o %t %hip_libs
// RUN: %run %t 2>&1 | FileCheck %s --allow-empty

__device__ int Global;

__global__ void kernel() {
  for (int I = 0; I < 1024; ++I)
    Global++;
}

int main() {
  kernel<<<1, 1>>>();
  CHECK_HIP(hipDeviceSynchronize());
  return 0;
}

// CHECK-NOT: WARNING: ConcurrencySanitizer

// RUN: %clang_hip_csan %s -o %t %hip_libs
// RUN: %run %t 2>&1 | FileCheck %s --allow-empty

__device__ int global;

__global__ void kernel() {
  for (int I = 0; I < 1024; ++I)
    __atomic_fetch_add(&global, 1, __ATOMIC_RELAXED);
}

int main() {
  kernel<<<64, 64>>>();
  CHECK_HIP(hipDeviceSynchronize());
  return 0;
}

// CHECK-NOT: WARNING: ConcurrencySanitizer

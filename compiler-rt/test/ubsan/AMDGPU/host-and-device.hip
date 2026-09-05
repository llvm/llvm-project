// REQUIRES: ubsan-hip
// RUN: %clang_ubsan_hip %s -o %t %hip_libs
// RUN: %run %t 2>&1 | FileCheck %s

__global__ void gpu_shift(int *Out) {
  int A = 1;
  *Out = A << 32;
}

int main() {
  int *P = nullptr;
  CHECK_HIP(hipMalloc((void **)&P, sizeof(int)));
  gpu_shift<<<1, 1>>>(P);
  volatile int H = 0x7fffffff;
  H = H + 1;
  CHECK_HIP(hipDeviceSynchronize());
  hipFree(P);
  return 0;
}

// CHECK-DAG: runtime error: signed integer overflow
// CHECK-DAG: runtime error: shift exponent

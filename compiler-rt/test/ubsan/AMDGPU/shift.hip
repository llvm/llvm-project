// REQUIRES: ubsan-hip
// RUN: %clang_ubsan_hip %s -o %t %hip_libs
// RUN: %run %t 2>&1 | FileCheck %s

__global__ void bad_shift(int *Out) {
  int A = 1;
  *Out = A << 32;
}

int main() {
  int *P = nullptr;
  CHECK_HIP(hipMalloc((void **)&P, sizeof(int)));
  bad_shift<<<1, 1>>>(P);
  hipDeviceSynchronize();
  hipFree(P);
  return 0;
}

// CHECK: runtime error: shift exponent
// CHECK: SUMMARY: UndefinedBehaviorSanitizer:

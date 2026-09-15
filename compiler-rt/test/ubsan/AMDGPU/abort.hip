// REQUIRES: ubsan-hip
// RUN: %clang_ubsan_hip %s -o %t %hip_libs -fno-sanitize-recover=undefined
// RUN: not %run %t 2>&1 | FileCheck %s

__global__ void gone(int *Out) {
  *Out = 1;
  __builtin_unreachable();
}

int main() {
  int *P = nullptr;
  CHECK_HIP(hipMalloc((void **)&P, sizeof(int)));
  gone<<<1, 1>>>(P);
  hipDeviceSynchronize();
  hipFree(P);
  return 3;
}

// CHECK: runtime error: execution reached an unreachable program point
// CHECK: SUMMARY: UndefinedBehaviorSanitizer:

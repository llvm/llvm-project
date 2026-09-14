// REQUIRES: ubsan-hip
// RUN: %clang_ubsan_hip %s -o %t %hip_libs
// RUN: %run %t 2>&1 | FileCheck %s
// RUN: %env_ubsan_opts=print_stacktrace=1 %run %t 2>&1 | FileCheck %s --check-prefix=STACK

__global__ void overflow(int *Out) {
  int A = 0x7fffffff;
  *Out = A + 1;
}

int main() {
  int *P = nullptr;
  CHECK_HIP(hipMalloc((void **)&P, sizeof(int)));
  overflow<<<1, 1>>>(P);
  hipDeviceSynchronize();
  hipFree(P);
  return 0;
}

// CHECK: runtime error: signed integer overflow
// CHECK: SUMMARY: UndefinedBehaviorSanitizer:

// STACK: runtime error: signed integer overflow
// STACK: #0 {{.*}}overflow{{.*}}overflow.hip
// STACK: SUMMARY: UndefinedBehaviorSanitizer:

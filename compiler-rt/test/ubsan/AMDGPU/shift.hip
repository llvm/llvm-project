// REQUIRES: ubsan-hip
// RUN: %clang_hip -fsanitize=undefined %s -o %t %hip_libs
// RUN: %run %t 2>&1 | FileCheck %s

// The host has no references to the shared runtime, it must still be loaded.
// RUN: %clang_hip -Xarch_device -fsanitize=undefined -shared-libsan %s -o %t \
// RUN:   %hip_libs
// RUN: llvm-readelf -d %t | FileCheck %s --check-prefix=NEEDED
// RUN: %run %t 2>&1 | FileCheck %s
// NEEDED: libclang_rt.ubsan_standalone.so

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

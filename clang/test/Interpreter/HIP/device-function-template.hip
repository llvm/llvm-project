// Tests device function templates
// RUN: cat %s | clang-repl --hip | FileCheck %s

#include <hip/hip_runtime.h>

extern "C" int printf(const char*, ...);

template <typename T> __device__ inline T sum(T a, T b) { return a + b; }
__global__ void test_kernel(int* value) { *value = sum(40, 2); }

int var;
int* devptr = nullptr;
printf("hipMalloc: %d\n", hipMalloc((void **) &devptr, sizeof(int)));
// CHECK: hipMalloc: 0

test_kernel<<<1,1>>>(devptr);
printf("HIP Error: %d\n", hipGetLastError());
// CHECK-NEXT: HIP Error: 0

printf("hipMemcpy: %d\n", hipMemcpy(&var, devptr, sizeof(int), hipMemcpyDeviceToHost));
// CHECK-NEXT: hipMemcpy: 0

printf("Value: %d\n", var);
// CHECK-NEXT: Value: 42

%quit

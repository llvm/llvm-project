// Tests hipMemcpy and writes from kernel
// RUN: cat %s | clang-repl --hip | FileCheck %s

#include <hip/hip_runtime.h>

extern "C" int printf(const char*, ...);

__global__ void test_func(int* value) { *value = 42; }

int var;
int* devptr = nullptr;
printf("hipMalloc: %d\n", hipMalloc((void **) &devptr, sizeof(int)));
// CHECK: hipMalloc: 0

test_func<<<1,1>>>(devptr);
printf("HIP Error: %d\n", hipGetLastError());
// CHECK-NEXT: HIP Error: 0

printf("hipMemcpy: %d\n", hipMemcpy(&var, devptr, sizeof(int), hipMemcpyDeviceToHost));
// CHECK-NEXT: hipMemcpy: 0

printf("Value: %d\n", var);
// CHECK-NEXT: Value: 42

%quit

// Checks that a function is available in both __host__ and __device__
// RUN: cat %s | clang-repl --hip | FileCheck %s

#include <hip/hip_runtime.h>

extern "C" int printf(const char*, ...);

__host__ __device__ inline int sum(int a, int b){ return a + b; }
__global__ void kernel(int * output){ *output = sum(40,2); }

printf("Host sum: %d\n", sum(41,1));
// CHECK: Host sum: 42

int var = 0;
int * deviceVar;
printf("hipMalloc: %d\n", hipMalloc((void **) &deviceVar, sizeof(int)));
// CHECK-NEXT: hipMalloc: 0

kernel<<<1,1>>>(deviceVar);
printf("HIP Error: %d\n", hipGetLastError());
// CHECK-NEXT: HIP Error: 0

printf("hipMemcpy: %d\n", hipMemcpy(&var, deviceVar, sizeof(int), hipMemcpyDeviceToHost));
// CHECK-NEXT: hipMemcpy: 0

printf("var: %d\n", var);
// CHECK-NEXT: var: 42

%quit

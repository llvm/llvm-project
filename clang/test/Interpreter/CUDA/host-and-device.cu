// Checks that a function is available in both __host__ and __device__
// RUN: rm -rf %t
// RUN: mkdir -p %t/lib
// RUN: %clang --target=%host-jit-triple -nostdlib -fPIC -shared \
// RUN:   %S/Inputs/cudart.c -o %t/lib/libcudart.so
// RUN: cat %s | clang-repl --cuda --cuda-path=%t -Xcc -nocudainc \
// RUN:   -Xcc -nocudalib -Xcc -include -Xcc %S/Inputs/cuda.h | FileCheck %s

extern "C" int printf(const char*, ...);

__host__ __device__ inline int sum(int a, int b){ return a + b; }
__global__ void kernel(int * output){ *output = sum(40,2); }
// CHECK: cudart: register _Z6kernelPi

printf("Host sum: %d\n", sum(41,1));
// CHECK-NEXT: Host sum: 42

int var = 0;
int * deviceVar;
printf("cudaMalloc: %d\n", cudaMalloc((void **) &deviceVar, sizeof(int)));
// CHECK-NEXT: cudaMalloc: 0

kernel<<<1,1>>>(deviceVar);
printf("CUDA Error: %d\n", cudaGetLastError());
// CHECK-NEXT: cudart: launch _Z6kernelPi grid (1,1,1) block (1,1,1)
// CHECK-NEXT: CUDA Error: 0

printf("cudaMemcpy: %d\n", cudaMemcpy(&var, deviceVar, sizeof(int), cudaMemcpyDeviceToHost));
// CHECK-NEXT: cudaMemcpy: 0

%quit

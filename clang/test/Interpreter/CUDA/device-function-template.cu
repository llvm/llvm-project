// Tests device function templates
// RUN: rm -rf %t
// RUN: mkdir -p %t/lib
// RUN: %clang --target=%host-jit-triple -nostdlib -fPIC -shared \
// RUN:   %S/Inputs/cudart.c -o %t/lib/libcudart.so
// RUN: cat %s | clang-repl --cuda --cuda-path=%t -Xcc -nocudainc \
// RUN:   -Xcc -nocudalib -Xcc -include -Xcc %S/Inputs/cuda.h | FileCheck %s

extern "C" int printf(const char*, ...);

template <typename T> __device__ inline T sum(T a, T b) { return a + b; }
__global__ void test_kernel(int* value) { *value = sum(40, 2); }
// CHECK: cudart: register _Z11test_kernelPi

int var;
int* devptr = nullptr;
printf("cudaMalloc: %d\n", cudaMalloc((void **) &devptr, sizeof(int)));
// CHECK-NEXT: cudaMalloc: 0

test_kernel<<<1,1>>>(devptr);
printf("CUDA Error: %d\n", cudaGetLastError());
// CHECK-NEXT: cudart: launch _Z11test_kernelPi grid (1,1,1) block (1,1,1)
// CHECK-NEXT: CUDA Error: 0

printf("cudaMemcpy: %d\n", cudaMemcpy(&var, devptr, sizeof(int), cudaMemcpyDeviceToHost));
// CHECK-NEXT: cudaMemcpy: 0

%quit

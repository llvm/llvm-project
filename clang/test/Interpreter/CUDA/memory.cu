// Tests cudaMalloc, cudaMemcpy and a kernel that takes a device pointer
// RUN: rm -rf %t
// RUN: mkdir -p %t/lib
// RUN: %clang --target=%host-jit-triple -nostdlib -fPIC -shared \
// RUN:   %S/Inputs/cudart.c -o %t/lib/libcudart.so
// RUN: cat %s | clang-repl --cuda --cuda-path=%t -Xcc -nocudainc \
// RUN:   -Xcc -nocudalib -Xcc -include -Xcc %S/Inputs/cuda.h | FileCheck %s

extern "C" int printf(const char*, ...);

__global__ void test_func(int* value) { *value = 42; }
// CHECK: cudart: register _Z9test_funcPi

int var;
int* devptr = nullptr;
printf("cudaMalloc: %d\n", cudaMalloc((void **) &devptr, sizeof(int)));
// CHECK-NEXT: cudaMalloc: 0

test_func<<<1,1>>>(devptr);
printf("CUDA Error: %d\n", cudaGetLastError());
// CHECK-NEXT: cudart: launch _Z9test_funcPi grid (1,1,1) block (1,1,1)
// CHECK-NEXT: CUDA Error: 0

printf("cudaMemcpy: %d\n", cudaMemcpy(&var, devptr, sizeof(int), cudaMemcpyDeviceToHost));
// CHECK-NEXT: cudaMemcpy: 0

%quit

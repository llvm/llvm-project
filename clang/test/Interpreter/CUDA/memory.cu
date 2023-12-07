// Tests cudaMemcpy and writes from kernel
// RUN: cat %s | clang-repl --cuda | FileCheck %s

extern "C" int printf(const char*, ...);

__global__ void test_func(int* value) { *value = 42; }

int var;
int* devptr = nullptr;
printf("cudaMalloc: %d\n", cudaMalloc((void **) &devptr, sizeof(int)));
// CHECK: cudaMalloc: 0

test_func<<<1,1>>>(devptr);
printf("CUDA Error: %d\n", cudaGetLastError());
// CHECK-NEXT: CUDA Error: 0

printf("cudaMemcpy: %d\n", cudaMemcpy(&var, devptr, sizeof(int), cudaMemcpyDeviceToHost));
// CHECK-NEXT: cudaMemcpy: 0

printf("Value: %d\n", var);
// CHECK-NEXT: Value: 42

%quit

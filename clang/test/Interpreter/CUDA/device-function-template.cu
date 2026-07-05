// Tests device function templates
// RUN: cat %s | clang-repl --cuda | FileCheck %s

extern "C" int printf(const char*, ...);

template <typename T> __device__ inline T sum(T a, T b) { return a + b; }
__global__ void test_kernel(int* value) { *value = sum(40, 2); }

int var;
int* devptr = nullptr;
printf("cudaMalloc: %d\n", cudaMalloc((void **) &devptr, sizeof(int)));
// CHECK: cudaMalloc: 0

test_kernel<<<1,1>>>(devptr);
printf("CUDA Error: %d\n", cudaGetLastError());
// CHECK-NEXT: CUDA Error: 0

printf("cudaMemcpy: %d\n", cudaMemcpy(&var, devptr, sizeof(int), cudaMemcpyDeviceToHost));
// CHECK-NEXT: cudaMemcpy: 0

printf("Value: %d\n", var);
// CHECK-NEXT: Value: 42

%quit

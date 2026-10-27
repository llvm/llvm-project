// Checks that a function is available in both __host__ and __device__
// RUN: cat %s | clang-repl --cuda | FileCheck %s

extern "C" int printf(const char*, ...);

__host__ __device__ inline int sum(int a, int b){ return a + b; }
__global__ void kernel(int * output){ *output = sum(40,2); }

printf("Host sum: %d\n", sum(41,1));
// CHECK: Host sum: 42

int var = 0;
int * deviceVar;
printf("cudaMalloc: %d\n", cudaMalloc((void **) &deviceVar, sizeof(int)));
// CHECK-NEXT: cudaMalloc: 0

kernel<<<1,1>>>(deviceVar);
printf("CUDA Error: %d\n", cudaGetLastError());
// CHECK-NEXT: CUDA Error: 0

printf("cudaMemcpy: %d\n", cudaMemcpy(&var, deviceVar, sizeof(int), cudaMemcpyDeviceToHost));
// CHECK-NEXT: cudaMemcpy: 0

printf("var: %d\n", var);
// CHECK-NEXT: var: 42

%quit

// Tests device math from libdevice
// RUN: cat %s | clang-repl --cuda | FileCheck %s

extern "C" int printf(const char*, ...);

__global__ void test_kernel(float* value) { *value = sinf(0.0f) + expf(0.0f); }

float var;
float* devptr = nullptr;
printf("cudaMalloc: %d\n", cudaMalloc((void **) &devptr, sizeof(float)));
// CHECK: cudaMalloc: 0

test_kernel<<<1,1>>>(devptr);
printf("CUDA Error: %d\n", cudaGetLastError());
// CHECK-NEXT: CUDA Error: 0

printf("cudaMemcpy: %d\n", cudaMemcpy(&var, devptr, sizeof(float), cudaMemcpyDeviceToHost));
// CHECK-NEXT: cudaMemcpy: 0

printf("Value: %.1f\n", var);
// CHECK-NEXT: Value: 1.0

%quit

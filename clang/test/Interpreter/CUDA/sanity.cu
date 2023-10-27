// RUN: cat %s | clang-repl --cuda | FileCheck %s

extern "C" int printf(const char*, ...);

__global__ void test_func() {}

test_func<<<1,1>>>();
printf("CUDA Error: %d", cudaGetLastError());
// CHECK: CUDA Error: 0

%quit

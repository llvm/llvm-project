// RUN: cat %s | clang-repl --hip | FileCheck %s

#include <hip/hip_runtime.h>

extern "C" int printf(const char*, ...);

__global__ void test_func() {}

test_func<<<1,1>>>();
printf("HIP Error: %d", hipGetLastError());
// CHECK: HIP Error: 0

%quit

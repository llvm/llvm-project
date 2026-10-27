// RUN: rm -rf %t
// RUN: mkdir -p %t/lib
// RUN: %clang --target=%host-jit-triple -nostdlib -fPIC -shared \
// RUN:   %S/Inputs/cudart.c -o %t/lib/libcudart.so
// RUN: cat %s | clang-repl --cuda --cuda-path=%t -Xcc -nocudainc \
// RUN:   -Xcc -nocudalib -Xcc -include -Xcc %S/Inputs/cuda.h | FileCheck %s

extern "C" int printf(const char*, ...);

__global__ void test_func() {}
// CHECK: cudart: fatbin .version {{[0-9]+\.[0-9]+}} .target sm_{{[0-9]+}}
// CHECK-NEXT: cudart: register _Z9test_funcv

test_func<<<1,1>>>();
printf("CUDA Error: %d", cudaGetLastError());
// CHECK-NEXT: cudart: launch _Z9test_funcv grid (1,1,1) block (1,1,1)
// CHECK-NEXT: CUDA Error: 0

%quit

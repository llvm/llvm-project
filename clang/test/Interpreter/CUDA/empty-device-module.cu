// Tests host-only inputs. They produce an empty device module, and emitting
// PTX for it must not be reported as a failure just because no pass changed
// the module.
// RUN: rm -rf %t
// RUN: mkdir -p %t/lib
// RUN: %clang --target=%host-jit-triple -nostdlib -fPIC -shared \
// RUN:   %S/Inputs/cudart.c -o %t/lib/libcudart.so
// RUN: cat %s | clang-repl --cuda --cuda-path=%t -Xcc -nocudainc \
// RUN:   -Xcc -nocudalib -Xcc -include -Xcc %S/Inputs/cuda.h | FileCheck %s

extern "C" int printf(const char*, ...);

int host_only = 42;
printf("host_only: %d\n", host_only);
// CHECK: host_only: 42

__global__ void kernel() {}
// CHECK-NEXT: cudart: fatbin
// CHECK-NEXT: cudart: register _Z6kernelv

kernel<<<1,1>>>();
printf("CUDA Error: %d\n", cudaGetLastError());
// CHECK-NEXT: cudart: launch _Z6kernelv grid (1,1,1) block (1,1,1)
// CHECK-NEXT: CUDA Error: 0

%quit

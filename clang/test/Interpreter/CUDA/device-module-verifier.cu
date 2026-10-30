// Release builds skip the IR verifier. Turn it on and check that the first
// device module passes it.
// RUN: rm -rf %t
// RUN: mkdir -p %t/lib
// RUN: %clang --target=%host-jit-triple -nostdlib -fPIC -shared \
// RUN:   %S/Inputs/cudart.c -o %t/lib/libcudart.so
// RUN: cat %s | clang-repl --cuda --cuda-path=%t -Xcc -nocudainc \
// RUN:   -Xcc -nocudalib -Xcc -include -Xcc %S/Inputs/cuda.h \
// RUN:   -Xcc -fverify-intermediate-code 2>&1 | FileCheck %s

extern "C" int printf(const char*, ...);

__global__ void kernel() {}
printf("kernel: %d\n", 0);
// CHECK-NOT: module flag identifiers must be unique
// CHECK-NOT: Broken module found
// CHECK: kernel: 0

%quit

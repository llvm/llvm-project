// -emit-llvm prints the device IR of each input before its host IR.
// RUN: rm -rf %t
// RUN: mkdir -p %t/lib
// RUN: %clang --target=%host-jit-triple -nostdlib -fPIC -shared \
// RUN:   %S/Inputs/cudart.c -o %t/lib/libcudart.so
// RUN: cat %s | clang-repl --cuda --cuda-path=%t -Xcc -nocudainc \
// RUN:   -Xcc -nocudalib -Xcc -include -Xcc %S/Inputs/cuda.h \
// RUN:   -Xcc -Xclang -Xcc -emit-llvm | FileCheck %s

extern "C" __global__ void kernel(int *p) { *p = 42; }
// CHECK: target triple = "nvptx64-nvidia-cuda"
// CHECK: define {{.*}}ptx_kernel void @kernel(
// CHECK: define {{.*}}void @__device_stub__kernel(

%quit

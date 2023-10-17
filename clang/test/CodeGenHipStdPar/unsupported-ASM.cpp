// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -aux-triple x86_64-unknown-linux-gnu \
// RUN:   --hipstdpar -x hip -emit-llvm -fcuda-is-device -o - %s | FileCheck %s

#define __global__ __attribute__((global))

__global__ void foo(int i) {
    asm ("addl %2, %1; seto %b0" : "=q" (i), "+g" (i) : "r" (i));
}

// CHECK: declare void @__ASM__hipstdpar_unsupported([{{.*}}])

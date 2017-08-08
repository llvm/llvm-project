// Test that the kernel argument info always refers to SPIR address spaces,
// even if the target has only one address space like x86_64 does.
// RUN: %clang_cc1 %s -cl-std=CL2.0 -emit-llvm -o - -triple x86_64-unknown-unknown -cl-kernel-arg-info | FileCheck %s

kernel void foo(__global int * G, __constant int *C, __local int *L) {
  *G = *C + *L;
}
// CHECK: !kernel_arg_addr_space ![[MD123:[0-9]+]]
// CHECK: ![[MD123]] = !{i32 1, i32 2, i32 3}

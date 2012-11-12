// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s

void normal_function() {
}

__kernel void kernel_function() {
}

// CHECK: !opencl.kernels = !{!0}
// CHECK: !0 = metadata !{void ()* @kernel_function}

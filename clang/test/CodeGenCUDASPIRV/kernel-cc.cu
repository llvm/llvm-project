// RUN: %clang_cc1 -fcuda-is-device -triple spirv32 -o - -emit-llvm -x cuda %s  | FileCheck %s
// RUN: %clang_cc1 -fcuda-is-device -triple spirv64 -o - -emit-llvm -x cuda %s  | FileCheck %s

// Verifies that building CUDA targeting SPIR-V {32,64} generates LLVM IR with
// spir_kernel attributes for kernel functions.

// CHECK: define spir_kernel void @_Z6kernelv()

__attribute__((global)) void kernel() { return; }

// CHECK: !opencl.ocl.version = !{[[OCL:![0-9]+]]}
// CHECK: [[OCL]] = !{i32 2, i32 0}

// RUN: %clang_cc1 -fcuda-is-device -triple spirv32 -o - -emit-llvm -x cuda %s  | FileCheck %s
// RUN: %clang_cc1 -fcuda-is-device -triple spirv64 -o - -emit-llvm -x cuda %s  | FileCheck %s
// RUN: %if cir-enabled %{ %clang_cc1 -fcuda-is-device -triple spirv32 -o - -emit-cir -fclangir -x cuda %s | FileCheck %s --check-prefix=CIR %}
// RUN: %if cir-enabled %{ %clang_cc1 -fcuda-is-device -triple spirv64 -o - -emit-cir -fclangir -x cuda %s | FileCheck %s --check-prefix=CIR %}

// Verifies that building CUDA targeting SPIR-V {32,64} generates LLVM IR with
// spir_kernel attributes for kernel functions.

// CHECK: define spir_kernel void @_Z6kernelv()

__attribute__((global)) void kernel() { return; }

// CHECK: !opencl.ocl.version = !{[[OCL:![0-9]+]]}
// CHECK: [[OCL]] = !{i32 2, i32 0}

// CIR: cir.cl.version = #cir.cl.version<2, 0>

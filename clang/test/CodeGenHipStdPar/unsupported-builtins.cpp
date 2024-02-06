// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -aux-triple x86_64-unknown-linux-gnu \
// RUN:   --hipstdpar -x hip -emit-llvm -fcuda-is-device -o - %s | FileCheck %s

#define __global__ __attribute__((global))

__global__ void foo() { return __builtin_ia32_pause(); }

// CHECK: declare void @__builtin_ia32_pause__hipstdpar_unsupported()

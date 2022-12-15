// Tests CUDA kernel arguments get global address space when targetting SPIR-V.


// RUN: %clang -emit-llvm --cuda-device-only --offload=spirv32 \
// RUN:   -nocudalib -nocudainc %s -o %t.bc -c 2>&1
// RUN: llvm-dis %t.bc -o %t.ll
// RUN: FileCheck %s --input-file=%t.ll

// RUN: %clang -emit-llvm --cuda-device-only --offload=spirv64 \
// RUN:   -nocudalib -nocudainc %s -o %t.bc -c 2>&1
// RUN: llvm-dis %t.bc -o %t.ll
// RUN: FileCheck %s --input-file=%t.ll

// CHECK: define
// CHECK-SAME: spir_kernel void @_Z6kernelPi(ptr addrspace(1) noundef

__attribute__((global)) void kernel(int* output) { *output = 1; }

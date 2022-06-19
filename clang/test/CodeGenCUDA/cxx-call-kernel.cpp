// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-unknown-linux-gnu \
// RUN:   -x hip -emit-llvm-bc %s -o %t.hip.bc
// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-unknown-linux-gnu \
// RUN:   -mlink-bitcode-file %t.hip.bc -DHIP_PLATFORM -emit-llvm \
// RUN:   %s -o - | FileCheck %s

#include "Inputs/cuda.h"

// CHECK: @_Z2g1i = constant void (i32)* @_Z17__device_stub__g1i, align 8
#if __HIP__
__global__ void g1(int x) {}
#else
extern void g1(int x);

// CHECK: call i32 @hipLaunchKernel{{.*}}@_Z2g1i
void test() {
  hipLaunchKernel((void*)g1, 1, 1, nullptr, 0, 0);
}

// CHECK: __hipRegisterFunction{{.*}}@_Z2g1i
#endif

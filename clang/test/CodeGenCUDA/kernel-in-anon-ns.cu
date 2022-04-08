// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -cuid=abc \
// RUN:   -aux-triple x86_64-unknown-linux-gnu -std=c++11 -fgpu-rdc \
// RUN:   -emit-llvm -o - -x hip %s > %t.dev

// RUN: %clang_cc1 -triple x86_64-gnu-linux -cuid=abc \
// RUN:   -aux-triple amdgcn-amd-amdhsa -std=c++11 -fgpu-rdc \
// RUN:   -emit-llvm -o - -x hip %s > %t.host

// RUN: cat %t.dev %t.host | FileCheck %s

#include "Inputs/cuda.h"

// CHECK: define weak_odr {{.*}}void @[[KERN:_ZN12_GLOBAL__N_16kernelEv\.anon\.b04fd23c98500190]](
// CHECK: @[[STR:.*]] = {{.*}} c"[[KERN]]\00"
// CHECK: call i32 @__hipRegisterFunction({{.*}}@[[STR]]

namespace {
__global__ void kernel() {
}
}

void test() {
  kernel<<<1, 1>>>();
}

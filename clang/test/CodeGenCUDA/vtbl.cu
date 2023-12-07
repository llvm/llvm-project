// RUN: %clang_cc1 -fcuda-is-device -triple amdgcn-amd-amdhsa -target-cpu gfx906 \
// RUN:   -emit-llvm -o - %s | FileCheck %s

#include "Inputs/cuda.h"

// CHECK-LABEL: define {{.*}}@_ZN1AC2Ev(ptr noundef nonnull align 8 dereferenceable(8) %this)
// CHECK: store ptr %this, ptr %this.addr.ascast
// CHECK: %this1 = load ptr, ptr %this.addr.ascast
// CHECK: store ptr addrspace(1) {{.*}} @_ZTV1A{{.*}}, ptr %this1
struct A {
  __device__ virtual void vf() {}
};

__global__ void kern() {
  A a;
}

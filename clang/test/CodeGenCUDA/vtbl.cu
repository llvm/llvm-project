// RUN: %clang_cc1 -fcuda-is-device -triple amdgcn-amd-amdhsa -target-cpu gfx906 \
// RUN:   -emit-llvm -o - %s | FileCheck %s

#include "Inputs/cuda.h"

// CHECK-LABEL: define {{.*}}@_ZN1AC2Ev(%struct.A* nonnull align 8 dereferenceable(8) %this)
// CHECK: store %struct.A* %this, %struct.A** %this.addr.ascast
// CHECK: %this1 = load %struct.A*, %struct.A** %this.addr.ascast
// CHECK: %[[VTFIELD:.*]] = bitcast %struct.A* %this1 to i32 (...)* addrspace(1)**
// CHECK: store i32 (...)* addrspace(1)* bitcast{{.*}} @_ZTV1A{{.*}}, i32 (...)* addrspace(1)** %[[VTFIELD]]
struct A {
  __device__ virtual void vf() {}
};

__global__ void kern() {
  A a;
}

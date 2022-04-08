// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -no-opaque-pointers -triple amdgcn-amd-amdhsa -fcuda-is-device -x hip %s \
// RUN:   -emit-llvm -o - | FileCheck -check-prefix=DEV %s
// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-unknown-linux-gnu -x hip %s \
// RUN:   -emit-llvm -o - | FileCheck -check-prefix=HOST %s

// Negative tests.

// RUN: %clang_cc1 -no-opaque-pointers -triple amdgcn-amd-amdhsa -fcuda-is-device -x hip %s \
// RUN:   -emit-llvm -o - | FileCheck -check-prefix=DEV-NEG %s

#include "Inputs/cuda.h"

// Test const var initialized with address of a const var.
// Both are promoted to device side.

// DEV-DAG: @_ZN5Test1L1aE = internal addrspace(4) constant i32 1
// DEV-DAG: @_ZN5Test11B2p1E = addrspace(4) externally_initialized constant i32* addrspacecast (i32 addrspace(4)* @_ZN5Test1L1aE to i32*)
// DEV-DAG: @_ZN5Test11B2p2E = addrspace(4) externally_initialized constant i32* addrspacecast (i32 addrspace(4)* @_ZN5Test1L1aE to i32*)
// DEV-DAG: @_ZN5Test12b2E = addrspace(1) externally_initialized global i32 1
// HOST-DAG: @_ZN5Test1L1aE = internal constant i32 1
// HOST-DAG: @_ZN5Test11B2p1E = constant i32* @_ZN5Test1L1aE
// HOST-DAG: @_ZN5Test11B2p2E = internal constant i32* undef
// HOST-DAG: @_ZN5Test12b1E = global i32 1
// HOST-DAG: @_ZN5Test12b2E = internal global i32 undef
namespace Test1 {
const int a = 1;

struct B {
    static const int *const p1;
    static __device__ const int *const p2;
};
const int *const B::p1 = &a;
__device__ const int *const B::p2 = &a;
int b1 = B::p1 == B::p2;
__device__ int b2 = B::p1 == B::p2;
}

// Test const var initialized with address of a non-cost var.
// Neither is promoted to device side.

// DEV-NEG-NOT: @_ZN5Test2L1aE
// DEV-NEG-NOT: @_ZN5Test21B1pE
// HOST-DAG: @_ZN5Test21aE = global i32 1
// HOST-DAG: @_ZN5Test21B1pE = constant i32* @_ZN5Test21aE

namespace Test2 {
int a = 1;

struct B {
    static int *const p;
};
int *const B::p = &a;
}

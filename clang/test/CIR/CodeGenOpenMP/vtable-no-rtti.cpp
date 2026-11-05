// RUN: %clang_cc1 -std=c++17 -triple amdgpu-amd-amdhsa -fopenmp \
// RUN:   -fopenmp-is-target-device \
// RUN:   -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -triple amdgpu-amd-amdhsa -fopenmp \
// RUN:   -fopenmp-is-target-device \
// RUN:   -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -std=c++17 -triple amdgpu-amd-amdhsa -fopenmp \
// RUN:   -fopenmp-is-target-device \
// RUN:   -emit-llvm %s -o - | FileCheck %s --check-prefix=OGCG

#pragma omp declare target

struct A {
  virtual ~A();
};

A::~A() = default;

#pragma omp end declare target

// CIR-NOT: cir.global {{.*}} @_ZTI1A
// CIR: cir.global {{.*}} @_ZTV1A
// CIR-SAME: #cir.ptr<null> : !cir.ptr<!u8i, target_address_space(1)>

// LLVM-NOT: @_ZTI1A
// LLVM: @_ZTV1A = {{.*}} addrspace(1) constant
// LLVM-SAME: ptr addrspace(1) null

// OGCG-NOT: @_ZTI1A
// OGCG: @_ZTV1A = {{.*}} addrspace(1) constant
// OGCG-SAME: ptr addrspace(1) null

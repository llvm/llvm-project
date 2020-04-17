// RUN: %clang_cc1 -x hip -emit-llvm -std=c++11 %s -o - \
// RUN:   -triple x86_64-linux-gnu | FileCheck -check-prefix=HOST %s
// RUN: %clang_cc1 -x hip -emit-llvm -std=c++11 %s -o - \
// RUN:   -triple amdgcn-amd-amdhsa -fcuda-is-device \
// RUN:   | FileCheck -check-prefix=DEV %s

#include "Inputs/cuda.h"

// Device side kernel name.
// HOST: @[[KERN:[0-9]+]] = private unnamed_addr constant [22 x i8] c"_Z1gIZ4mainEUlvE_EvT_\00"

// Template instantiation for h.
// HOST-LABEL: define internal void @_Z1hIZ4mainEUlvE_EvT_

// HOST-LABEL: define internal void @_Z16__device_stub__gIZ4mainEUlvE_EvT_

// Check kernel is registered with correct device side kernel name.
// HOST: @__hipRegisterFunction(i8** %0, i8* bitcast ({{.*}}@[[KERN]]

// Check lambda is not emitted in host compilation.
// HOST-NOT: define{{.*}}@_ZZ4mainENKUlvE_clEv

// DEV: @a = addrspace(1) externally_initialized global i32 0

// Check kernel is calling lambda function.
// DEV-LABEL: define amdgpu_kernel void @_Z1gIZ4mainEUlvE_EvT_
// DEV: call void @_ZZ4mainENKUlvE_clEv

// Check lambda is emitted in device compilation and accessind device variable.
// DEV-LABEL: define internal void @_ZZ4mainENKUlvE_clEv
// DEV: store i32 1, i32* addrspacecast (i32 addrspace(1)* @a to i32*)
template<class F>
__global__ void g(F f) { f(); }

template<class F>
void h(F f) { g<<<1,1>>>(f); }

__device__ int a;

int main(void) {
  h([&](){ a=1;});
}

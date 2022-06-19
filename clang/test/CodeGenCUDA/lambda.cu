// RUN: %clang_cc1 -no-opaque-pointers -x hip -emit-llvm -std=c++11 %s -o - \
// RUN:   -triple x86_64-linux-gnu \
// RUN:   | FileCheck -check-prefix=HOST %s
// RUN: %clang_cc1 -no-opaque-pointers -x hip -emit-llvm -std=c++11 %s -o - \
// RUN:   -triple amdgcn-amd-amdhsa -fcuda-is-device \
// RUN:   | FileCheck -check-prefix=DEV %s

#include "Inputs/cuda.h"

// Device side kernel name.
// HOST: @[[KERN_CAPTURE:[0-9]+]] = {{.*}} c"_Z1gIZ12test_capturevEUlvE_EvT_\00"
// HOST: @[[KERN_RESOLVE:[0-9]+]] = {{.*}} c"_Z1gIZ12test_resolvevEUlvE_EvT_\00"

// Check functions emitted for test_capture in host compilation.
// Check lambda is not emitted in host compilation.
// HOST-LABEL: define{{.*}} void @_Z12test_capturev
// HOST:  call void @_Z19test_capture_helperIZ12test_capturevEUlvE_EvT_
// HOST-LABEL: define internal void @_Z19test_capture_helperIZ12test_capturevEUlvE_EvT_
// HOST:  call void @_Z16__device_stub__gIZ12test_capturevEUlvE_EvT_
// HOST-NOT: define{{.*}}@_ZZ4mainENKUlvE_clEv

// Check functions emitted for test_resolve in host compilation.
// Check host version of template function 'overloaded' is emitted and called
// by the lambda function.
// HOST-LABEL: define{{.*}} void @_Z12test_resolvev
// HOST:  call void @_Z19test_resolve_helperIZ12test_resolvevEUlvE_EvT_()
// HOST-LABEL: define internal void @_Z19test_resolve_helperIZ12test_resolvevEUlvE_EvT_
// HOST:  call void @_Z16__device_stub__gIZ12test_resolvevEUlvE_EvT_
// HOST:  call void @_ZZ12test_resolvevENKUlvE_clEv
// HOST-LABEL: define internal void @_ZZ12test_resolvevENKUlvE_clEv
// HOST:  call noundef i32 @_Z10overloadedIiET_v
// HOST-LABEL: define linkonce_odr noundef i32 @_Z10overloadedIiET_v
// HOST:  ret i32 2

// Check kernel is registered with correct device side kernel name.
// HOST: @__hipRegisterFunction({{.*}}@[[KERN_CAPTURE]]
// HOST: @__hipRegisterFunction({{.*}}@[[KERN_RESOLVE]]

// DEV: @a ={{.*}} addrspace(1) externally_initialized global i32 0

// Check functions emitted for test_capture in device compilation.
// Check lambda is emitted in device compilation and accessing device variable.
// DEV-LABEL: define{{.*}} amdgpu_kernel void @_Z1gIZ12test_capturevEUlvE_EvT_
// DEV:  call void @_ZZ12test_capturevENKUlvE_clEv
// DEV-LABEL: define internal void @_ZZ12test_capturevENKUlvE_clEv
// DEV:  store i32 1, i32* addrspacecast (i32 addrspace(1)* @a to i32*)

// Check functions emitted for test_resolve in device compilation.
// Check device version of template function 'overloaded' is emitted and called
// by the lambda function.
// DEV-LABEL: define{{.*}} amdgpu_kernel void @_Z1gIZ12test_resolvevEUlvE_EvT_
// DEV:  call void @_ZZ12test_resolvevENKUlvE_clEv
// DEV-LABEL: define internal void @_ZZ12test_resolvevENKUlvE_clEv
// DEV:  call noundef i32 @_Z10overloadedIiET_v
// DEV-LABEL: define linkonce_odr noundef i32 @_Z10overloadedIiET_v
// DEV:  ret i32 1

__device__ int a;

template<class T>
__device__ T overloaded() { return 1; }

template<class T>
__host__ T overloaded() { return 2; }

template<class F>
__global__ void g(F f) { f(); }

template<class F>
void test_capture_helper(F f) { g<<<1,1>>>(f); }

template<class F>
void test_resolve_helper(F f) { g<<<1,1>>>(f); f(); }

// Test capture of device variable in lambda function.
void test_capture(void) {
  test_capture_helper([](){ a = 1;});
}

// Test resolving host/device function in lambda function.
// Callee should resolve to correct host/device function based on where
// the lambda function is called, not where it is defined.
void test_resolve(void) {
  test_resolve_helper([](){ overloaded<int>();});
}

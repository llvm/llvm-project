// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgcn-unknown-unknown -target-cpu tonga -S -emit-llvm -O0 -o - %s | FileCheck %s

// Make sure using numbered address spaces doesn't trigger crashes when a
// builtin has an address space parameter.

// CHECK-LABEL: @test_numbered_as_to_generic(
// CHECK: addrspacecast i32 addrspace(42)* %0 to i32*
void test_numbered_as_to_generic(__attribute__((address_space(42))) int *arbitary_numbered_ptr) {
  generic int* generic_ptr = arbitary_numbered_ptr;
  *generic_ptr = 4;
}

// CHECK-LABEL: @test_numbered_as_to_builtin(
// CHECK: addrspacecast i32 addrspace(42)* %0 to float addrspace(3)*
void test_numbered_as_to_builtin(__attribute__((address_space(42))) int *arbitary_numbered_ptr, float src) {
  volatile float result = __builtin_amdgcn_ds_fmaxf(arbitary_numbered_ptr, src, 0, 0, false);
}

// CHECK-LABEL: @test_generic_as_to_builtin_parameter_explicit_cast(
// CHECK: addrspacecast i32 addrspace(3)* %0 to i32*
void test_generic_as_to_builtin_parameter_explicit_cast(__local int *local_ptr, float src) {
  generic int* generic_ptr = local_ptr;
  volatile float result = __builtin_amdgcn_ds_fmaxf((__local float*) generic_ptr, src, 0, 0, false);
}

// CHECK-LABEL: @test_generic_as_to_builtin_parameter_implicit_cast(
// CHECK: addrspacecast i32* %2 to float addrspace(3)*
void test_generic_as_to_builtin_parameter_implicit_cast(__local int *local_ptr, float src) {
  generic int* generic_ptr = local_ptr;

  volatile float result = __builtin_amdgcn_ds_fmaxf(generic_ptr, src, 0, 0, false);
}


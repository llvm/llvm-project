// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgcn-unknown-unknown -target-cpu tahiti -emit-llvm -disable-llvm-passes -o - %s | FileCheck -enable-var-scope %s

// SI did not actually support flat addressing, but we can codegen the address
// space test builtins. The target specfic part is a load from the implicit
// argument buffer to use for the high pointer bits. It's just that buffer won't
// be initialized to something useful. The proper way to diagnose invalid flat
// usage is to forbid flat pointers on unsupported targets.

// CHECK-LABEL: @test_is_shared_global(
// CHECK: [[CAST:%[0-9]+]] = addrspacecast ptr addrspace(1) %{{[0-9]+}} to ptr
// CHECK: call i1 @llvm.amdgcn.is.shared(ptr [[CAST]]
int test_is_shared_global(const global int* ptr) {
  return __builtin_amdgcn_is_shared(ptr);
}

// CHECK-LABEL: @test_is_private_global(
// CHECK: [[CAST:%[0-9]+]] = addrspacecast ptr addrspace(1) %{{[0-9]+}} to ptr
// CHECK: call i1 @llvm.amdgcn.is.private(ptr [[CAST]]
int test_is_private_global(const global int* ptr) {
  return __builtin_amdgcn_is_private(ptr);
}

// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgpu9.4-amd-amdhsa -DTEST_VALID -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgpu9.0a-amd-amdhsa -DTEST_UNSUPPORTED -emit-llvm -o /dev/null -verify %s

#ifdef TEST_VALID
// CHECK-LABEL: @test_buffer_inv(
// CHECK: call void @llvm.amdgcn.buffer.inv(i32 0)
// CHECK: call void @llvm.amdgcn.buffer.inv(i32 1)
// CHECK: call void @llvm.amdgcn.buffer.inv(i32 16)
// CHECK: call void @llvm.amdgcn.buffer.inv(i32 17)
void test_buffer_inv(void) {
  __builtin_amdgcn_buffer_inv(0);
  __builtin_amdgcn_buffer_inv(1);
  __builtin_amdgcn_buffer_inv(16);
  __builtin_amdgcn_buffer_inv(17);
}
#endif

#ifdef TEST_UNSUPPORTED
void test_unsupported(void) {
  __builtin_amdgcn_buffer_inv(0); // expected-error {{'__builtin_amdgcn_buffer_inv' needs target feature buffer-inv-inst}}
}
#endif

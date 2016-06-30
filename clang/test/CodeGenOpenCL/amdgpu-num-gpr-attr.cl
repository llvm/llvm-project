// RUN: %clang_cc1 -triple amdgcn-- -target-cpu tahiti -O0 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O0 -emit-llvm -verify -o - %s | FileCheck -check-prefix=X86 %s

// Make sure this is silently accepted on other targets.

__attribute__((amdgpu_num_vgpr(64))) // expected-no-diagnostics
kernel void test_num_vgpr64() {
// CHECK: define amdgpu_kernel void @test_num_vgpr64() [[ATTR_VGPR64:#[0-9]+]]
}

__attribute__((amdgpu_num_sgpr(32))) // expected-no-diagnostics
kernel void test_num_sgpr32() {
// CHECK: define amdgpu_kernel void @test_num_sgpr32() [[ATTR_SGPR32:#[0-9]+]]
}

__attribute__((amdgpu_num_vgpr(64), amdgpu_num_sgpr(32))) // expected-no-diagnostics
kernel void test_num_vgpr64_sgpr32() {
// CHECK: define amdgpu_kernel void @test_num_vgpr64_sgpr32() [[ATTR_VGPR64_SGPR32:#[0-9]+]]

}

__attribute__((amdgpu_num_sgpr(20), amdgpu_num_vgpr(40))) // expected-no-diagnostics
kernel void test_num_sgpr20_vgpr40() {
// CHECK: define amdgpu_kernel void @test_num_sgpr20_vgpr40() [[ATTR_SGPR20_VGPR40:#[0-9]+]]
}

__attribute__((amdgpu_num_vgpr(0))) // expected-no-diagnostics
kernel void test_num_vgpr0() {
}

__attribute__((amdgpu_num_sgpr(0))) // expected-no-diagnostics
kernel void test_num_sgpr0() {
}

__attribute__((amdgpu_num_vgpr(0), amdgpu_num_sgpr(0))) // expected-no-diagnostics
kernel void test_num_vgpr0_sgpr0() {
}


// X86-NOT: "amdgpu_num_vgpr"
// X86-NOT: "amdgpu_num_sgpr"

// CHECK-NOT: "amdgpu_num_vgpr"="0"
// CHECK-NOT: "amdgpu_num_sgpr"="0"
// CHECK-DAG: attributes [[ATTR_VGPR64]] = { nounwind "amdgpu_num_vgpr"="64"
// CHECK-DAG: attributes [[ATTR_SGPR32]] = { nounwind "amdgpu_num_sgpr"="32"
// CHECK-DAG: attributes [[ATTR_VGPR64_SGPR32]] = { nounwind "amdgpu_num_sgpr"="32" "amdgpu_num_vgpr"="64"
// CHECK-DAG: attributes [[ATTR_SGPR20_VGPR40]] = { nounwind "amdgpu_num_sgpr"="20" "amdgpu_num_vgpr"="40"

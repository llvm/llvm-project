// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgpu12.50-amd-amdhsa -aux-triple x86_64-unknown-linux-gnu -x hip -emit-llvm -fcuda-is-device -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple amdgpu12.50-amd-amdhsa -aux-triple x86_64-pc-windows-msvc -x hip -emit-llvm -fcuda-is-device -o - %s | FileCheck %s

#define __device__ __attribute__((device))

typedef unsigned int u32x2 __attribute__((ext_vector_type(2)));
typedef unsigned int u32x3 __attribute__((ext_vector_type(3)));
typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));

// CHECK-LABEL: @_Z20test_perm_pk16_b6_u4jyDv2_j
// CHECK: call <3 x i32> @llvm.amdgcn.perm.pk16.b6.u4(i32 %{{.*}}, i64 %{{.*}}, <2 x i32> %{{.*}})
__device__ u32x3 test_perm_pk16_b6_u4(unsigned int a, unsigned long long b, u32x2 c) {
  return __builtin_amdgcn_perm_pk16_b6_u4(a, b, c);
}
// CHECK-LABEL: @_Z20test_perm_pk16_b8_u4yyDv2_j
// CHECK: call <4 x i32> @llvm.amdgcn.perm.pk16.b8.u4(i64 %{{.*}}, i64 %{{.*}}, <2 x i32> %{{.*}})
__device__ u32x4 test_perm_pk16_b8_u4(unsigned long long a, unsigned long long b, u32x2 c) {
  return __builtin_amdgcn_perm_pk16_b8_u4(a, b, c);
}

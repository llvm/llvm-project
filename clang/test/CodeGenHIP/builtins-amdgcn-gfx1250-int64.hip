// RUN: %clang_cc1 -triple amdgpu12.50-amd-amdhsa -aux-triple x86_64-unknown-linux-gnu -x hip -emit-llvm -fcuda-is-device -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple amdgpu12.50-amd-amdhsa -aux-triple x86_64-pc-windows-msvc -x hip -emit-llvm -fcuda-is-device -o - %s | FileCheck %s

#define __device__ __attribute__((device))

typedef unsigned int u32x2 __attribute__((ext_vector_type(2)));
typedef unsigned int u32x3 __attribute__((ext_vector_type(3)));
typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));
typedef int v16i __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef float v16f __attribute__((ext_vector_type(16)));

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

// CHECK-LABEL: @_Z{{[0-9]+}}test_ds_atomic_async_barrier_arrive_b64
// CHECK: call void @llvm.amdgcn.ds.atomic.async.barrier.arrive.b64(ptr addrspace(3) %{{.*}})
__device__ void test_ds_atomic_async_barrier_arrive_b64(__attribute__((address_space(3))) __INT64_TYPE__ *p) {
  __builtin_amdgcn_ds_atomic_async_barrier_arrive_b64(p);
}

// CHECK-LABEL: @_Z{{[0-9]+}}test_ds_atomic_barrier_arrive_rtn_b64
// CHECK: call i64 @llvm.amdgcn.ds.atomic.barrier.arrive.rtn.b64(ptr addrspace(3) %{{.*}}, i64 %{{.*}})
__device__ long long test_ds_atomic_barrier_arrive_rtn_b64(__attribute__((address_space(3))) __INT64_TYPE__ *p, long long v) {
  return __builtin_amdgcn_ds_atomic_barrier_arrive_rtn_b64(p, v);
}

// CHECK-LABEL: @_Z{{[0-9]+}}test_wmma_scale16_f32_16x16x128_f8f6f4
// CHECK: call contract <8 x float> @llvm.amdgcn.wmma.scale16.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %{{.*}}, i32 0, <16 x i32> %{{.*}}, i16 0, <8 x float> %{{.*}}, i32 0, i32 0, i64 %{{.*}}, i32 0, i32 0, i64 %{{.*}}, i1 false, i1 false)
__device__ v8f test_wmma_scale16_f32_16x16x128_f8f6f4(v16i a, v16i b, v8f c, long long s0, long long s1) {
  return __builtin_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4(0, a, 0, b, 0, c, 0, 0, s0, 0, 0, s1, false, false);
}

// CHECK-LABEL: @_Z{{[0-9]+}}test_wmma_scale16_f32_32x16x128_f4
// CHECK: call contract <16 x float> @llvm.amdgcn.wmma.scale16.f32.32x16x128.f4.v16f32.v16i32.v8i32(<16 x i32> %{{.*}}, <8 x i32> %{{.*}}, i16 0, <16 x float> %{{.*}}, i32 0, i32 0, i64 %{{.*}}, i32 0, i32 0, i64 %{{.*}}, i1 false, i1 false)
__device__ v16f test_wmma_scale16_f32_32x16x128_f4(v16i a, v8i b, v16f c, long long s0, long long s1) {
  return __builtin_amdgcn_wmma_scale16_f32_32x16x128_f4(a, b, 0, c, 0, 0, s0, 0, 0, s1, false, false);
}

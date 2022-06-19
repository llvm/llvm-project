// RUN: %clang_cc1 -no-opaque-pointers -O0 -cl-std=CL2.0 -triple amdgcn-amd-amdhsa -target-cpu gfx940 \
// RUN:   %s -S -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -no-opaque-pointers -O0 -cl-std=CL2.0 -triple amdgcn-amd-amdhsa -target-cpu gfx940 \
// RUN:   -S -o - %s | FileCheck -check-prefix=GFX940 %s

// REQUIRES: amdgpu-registered-target

typedef half  __attribute__((ext_vector_type(2))) half2;
typedef short __attribute__((ext_vector_type(2))) short2;

// CHECK-LABEL: test_flat_add_f32
// CHECK: call float @llvm.amdgcn.flat.atomic.fadd.f32.p0f32.f32(float* %{{.*}}, float %{{.*}})
// GFX940-LABEL:  test_flat_add_f32
// GFX940: flat_atomic_add_f32
half2 test_flat_add_f32(__generic float *addr, float x) {
  return __builtin_amdgcn_flat_atomic_fadd_f32(addr, x);
}

// CHECK-LABEL: test_flat_add_2f16
// CHECK: call <2 x half> @llvm.amdgcn.flat.atomic.fadd.v2f16.p0v2f16.v2f16(<2 x half>* %{{.*}}, <2 x half> %{{.*}})
// GFX940-LABEL:  test_flat_add_2f16
// GFX940: flat_atomic_pk_add_f16
half2 test_flat_add_2f16(__generic half2 *addr, half2 x) {
  return __builtin_amdgcn_flat_atomic_fadd_v2f16(addr, x);
}

// CHECK-LABEL: test_flat_add_2bf16
// CHECK: call <2 x i16> @llvm.amdgcn.flat.atomic.fadd.v2bf16.p0v2i16(<2 x i16>* %{{.*}}, <2 x i16> %{{.*}})
// GFX940-LABEL:  test_flat_add_2bf16
// GFX940: flat_atomic_pk_add_bf16
short2 test_flat_add_2bf16(__generic short2 *addr, short2 x) {
  return __builtin_amdgcn_flat_atomic_fadd_v2bf16(addr, x);
}

// CHECK-LABEL: test_global_add_2bf16
// CHECK: call <2 x i16> @llvm.amdgcn.global.atomic.fadd.v2bf16.p1v2i16(<2 x i16> addrspace(1)* %{{.*}}, <2 x i16> %{{.*}})
// GFX940-LABEL:  test_global_add_2bf16
// GFX940: global_atomic_pk_add_bf16
short2 test_global_add_2bf16(__global short2 *addr, short2 x) {
  return __builtin_amdgcn_global_atomic_fadd_v2bf16(addr, x);
}

// CHECK-LABEL: test_local_add_2bf16
// CHECK: call <2 x i16> @llvm.amdgcn.ds.fadd.v2bf16(<2 x i16> addrspace(3)* %{{.*}}, <2 x i16> %
// GFX940-LABEL:  test_local_add_2bf16
// GFX940: ds_pk_add_rtn_bf16
short2 test_local_add_2bf16(__local short2 *addr, short2 x) {
  return __builtin_amdgcn_ds_atomic_fadd_v2bf16(addr, x);
}

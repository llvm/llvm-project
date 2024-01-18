// RUN: %clang_cc1 -O0 -cl-std=CL2.0 -triple amdgcn-amd-amdhsa -target-cpu gfx1200 \
// RUN:   %s -S -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -O0 -cl-std=CL2.0 -triple amdgcn-amd-amdhsa -target-cpu gfx1200 \
// RUN:   -S -o - %s | FileCheck -check-prefix=GFX12 %s

// REQUIRES: amdgpu-registered-target

typedef half  __attribute__((ext_vector_type(2))) half2;
typedef short __attribute__((ext_vector_type(2))) short2;

// CHECK-LABEL: test_local_add_2bf16
// CHECK: call <2 x i16> @llvm.amdgcn.ds.fadd.v2bf16(ptr addrspace(3) %{{.*}}, <2 x i16> %
// GFX12-LABEL:  test_local_add_2bf16
// GFX12: ds_pk_add_rtn_bf16
short2 test_local_add_2bf16(__local short2 *addr, short2 x) {
  return __builtin_amdgcn_ds_atomic_fadd_v2bf16(addr, x);
}

// CHECK-LABEL: test_local_add_2bf16_noret
// CHECK: call <2 x i16> @llvm.amdgcn.ds.fadd.v2bf16(ptr addrspace(3) %{{.*}}, <2 x i16> %
// GFX12-LABEL:  test_local_add_2bf16_noret
// GFX12: ds_pk_add_bf16
void test_local_add_2bf16_noret(__local short2 *addr, short2 x) {
  __builtin_amdgcn_ds_atomic_fadd_v2bf16(addr, x);
}

// CHECK-LABEL: test_local_add_2f16
// CHECK: call <2 x half> @llvm.amdgcn.ds.fadd.v2f16(ptr addrspace(3) %{{.*}}, <2 x half> %
// GFX12-LABEL:  test_local_add_2f16
// GFX12: ds_pk_add_rtn_f16
half2 test_local_add_2f16(__local half2 *addr, half2 x) {
  return __builtin_amdgcn_ds_atomic_fadd_v2f16(addr, x);
}

// CHECK-LABEL: test_local_add_2f16_noret
// CHECK: call <2 x half> @llvm.amdgcn.ds.fadd.v2f16(ptr addrspace(3) %{{.*}}, <2 x half> %
// GFX12-LABEL:  test_local_add_2f16_noret
// GFX12: ds_pk_add_f16
void test_local_add_2f16_noret(__local half2 *addr, half2 x) {
  __builtin_amdgcn_ds_atomic_fadd_v2f16(addr, x);
}

// CHECK-LABEL: test_flat_add_2f16
// CHECK: call <2 x half> @llvm.amdgcn.flat.atomic.fadd.v2f16.p0.v2f16(ptr %{{.*}}, <2 x half> %{{.*}})
// GFX12-LABEL:  test_flat_add_2f16
// GFX12: flat_atomic_pk_add_f16
half2 test_flat_add_2f16(__generic half2 *addr, half2 x) {
  return __builtin_amdgcn_flat_atomic_fadd_v2f16(addr, x);
}

// CHECK-LABEL: test_flat_add_2bf16
// CHECK: call <2 x i16> @llvm.amdgcn.flat.atomic.fadd.v2bf16.p0(ptr %{{.*}}, <2 x i16> %{{.*}})
// GFX12-LABEL:  test_flat_add_2bf16
// GFX12: flat_atomic_pk_add_bf16
short2 test_flat_add_2bf16(__generic short2 *addr, short2 x) {
  return __builtin_amdgcn_flat_atomic_fadd_v2bf16(addr, x);
}

// CHECK-LABEL: test_global_add_half2
// CHECK: call <2 x half> @llvm.amdgcn.global.atomic.fadd.v2f16.p1.v2f16(ptr addrspace(1) %{{.*}}, <2 x half> %{{.*}})
// GFX12-LABEL:  test_global_add_half2
// GFX12:  global_atomic_pk_add_f16 v2, v[0:1], v2, off th:TH_ATOMIC_RETURN
void test_global_add_half2(__global half2 *addr, half2 x) {
  half2 *rtn;
  *rtn = __builtin_amdgcn_global_atomic_fadd_v2f16(addr, x);
}

// CHECK-LABEL: test_global_add_half2_noret
// CHECK: call <2 x half> @llvm.amdgcn.global.atomic.fadd.v2f16.p1.v2f16(ptr addrspace(1) %{{.*}}, <2 x half> %{{.*}})
// GFX12-LABEL:  test_global_add_half2_noret
// GFX12:  global_atomic_pk_add_f16 v[0:1], v2, off
void test_global_add_half2_noret(__global half2 *addr, half2 x) {
  __builtin_amdgcn_global_atomic_fadd_v2f16(addr, x);
}

// CHECK-LABEL: test_global_add_2bf16
// CHECK: call <2 x i16> @llvm.amdgcn.global.atomic.fadd.v2bf16.p1(ptr addrspace(1) %{{.*}}, <2 x i16> %{{.*}})
// GFX12-LABEL:  test_global_add_2bf16
// GFX12: global_atomic_pk_add_bf16 v2, v[0:1], v2, off th:TH_ATOMIC_RETURN
void test_global_add_2bf16(__global short2 *addr, short2 x) {
  short2 *rtn;
  *rtn = __builtin_amdgcn_global_atomic_fadd_v2bf16(addr, x);
}

// CHECK-LABEL: test_global_add_2bf16_noret
// CHECK: call <2 x i16> @llvm.amdgcn.global.atomic.fadd.v2bf16.p1(ptr addrspace(1) %{{.*}}, <2 x i16> %{{.*}})
// GFX12-LABEL:  test_global_add_2bf16_noret
// GFX12: global_atomic_pk_add_bf16 v[0:1], v2, off
void test_global_add_2bf16_noret(__global short2 *addr, short2 x) {
  __builtin_amdgcn_global_atomic_fadd_v2bf16(addr, x);
}

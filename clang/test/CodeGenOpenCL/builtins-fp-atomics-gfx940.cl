// RUN: %clang_cc1 -O0 -cl-std=CL2.0 -triple amdgcn-amd-amdhsa -target-cpu gfx940 \
// RUN:   %s -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -O0 -cl-std=CL2.0 -triple amdgcn-amd-amdhsa -target-cpu gfx940 \
// RUN:   -S -o - %s | FileCheck -check-prefix=GFX940 %s

// REQUIRES: amdgpu-registered-target

typedef half  __attribute__((ext_vector_type(2))) half2;
typedef short __attribute__((ext_vector_type(2))) short2;

// CHECK-LABEL: test_flat_add_f32
// CHECK: [[RMW:%.+]] = atomicrmw fadd ptr %{{.+}}, float %{{.+}} syncscope("agent") monotonic, align 4, !amdgpu.no.fine.grained.memory !{{[0-9]+}}, !amdgpu.ignore.denormal.mode !{{[0-9]+$}}

// GFX940-LABEL:  test_flat_add_f32
// GFX940: flat_atomic_add_f32
half2 test_flat_add_f32(__generic float *addr, float x) {
  return __builtin_amdgcn_flat_atomic_fadd_f32(addr, x);
}

// CHECK-LABEL: test_flat_add_2f16
// CHECK: [[RMW:%.+]] = atomicrmw fadd ptr %{{.+}}, <2 x half> %{{.+}} syncscope("agent") monotonic, align 4, !amdgpu.no.fine.grained.memory !{{[0-9]+$}}

// GFX940-LABEL:  test_flat_add_2f16
// GFX940: flat_atomic_pk_add_f16
half2 test_flat_add_2f16(__generic half2 *addr, half2 x) {
  return __builtin_amdgcn_flat_atomic_fadd_v2f16(addr, x);
}

// CHECK-LABEL: test_flat_add_2bf16
// CHECK: [[BC0:%.+]] = bitcast <2 x i16> {{.+}} to <2 x bfloat>
// CHECK: [[RMW:%.+]] = atomicrmw fadd ptr %{{.+}}, <2 x bfloat> [[BC0]] syncscope("agent") monotonic, align 4, !amdgpu.no.fine.grained.memory !{{[0-9]+$}}
// CHECK-NEXT: bitcast <2 x bfloat> [[RMW]] to <2 x i16>

// GFX940-LABEL:  test_flat_add_2bf16
// GFX940: flat_atomic_pk_add_bf16
short2 test_flat_add_2bf16(__generic short2 *addr, short2 x) {
  return __builtin_amdgcn_flat_atomic_fadd_v2bf16(addr, x);
}

// CHECK-LABEL: test_global_add_2bf16
// CHECK: [[BC0:%.+]] = bitcast <2 x i16> {{.+}} to <2 x bfloat>
// CHECK: [[RMW:%.+]] = atomicrmw fadd ptr addrspace(1) %{{.+}}, <2 x bfloat> [[BC0]] syncscope("agent") monotonic, align 4, !amdgpu.no.fine.grained.memory !{{[0-9]+$}}
// CHECK-NEXT: bitcast <2 x bfloat> [[RMW]] to <2 x i16>

// GFX940-LABEL:  test_global_add_2bf16
// GFX940: global_atomic_pk_add_bf16
short2 test_global_add_2bf16(__global short2 *addr, short2 x) {
  return __builtin_amdgcn_global_atomic_fadd_v2bf16(addr, x);
}

// CHECK-LABEL: test_local_add_2bf16

// CHECK: [[BC0:%.+]] = bitcast <2 x i16> {{.+}} to <2 x bfloat>
// CHECK: [[RMW:%.+]] = atomicrmw fadd ptr addrspace(3) %{{.+}}, <2 x bfloat> [[BC0]] syncscope("agent") monotonic, align 4{{$}}
// CHECK-NEXT: bitcast <2 x bfloat> [[RMW]] to <2 x i16>

// GFX940-LABEL:  test_local_add_2bf16
// GFX940: ds_pk_add_rtn_bf16
short2 test_local_add_2bf16(__local short2 *addr, short2 x) {
  return __builtin_amdgcn_ds_atomic_fadd_v2bf16(addr, x);
}

// CHECK-LABEL: test_local_add_2f16
// CHECK: = atomicrmw fadd ptr addrspace(3) %{{.+}}, <2 x half> %{{.+}} monotonic, align 4
// GFX940-LABEL:  test_local_add_2f16
// GFX940: ds_pk_add_rtn_f16
half2 test_local_add_2f16(__local half2 *addr, half2 x) {
  return __builtin_amdgcn_ds_atomic_fadd_v2f16(addr, x);
}

// CHECK-LABEL: test_local_add_2f16_noret
// CHECK: = atomicrmw fadd ptr addrspace(3) %{{.+}}, <2 x half> %{{.+}} monotonic, align 4
// GFX940-LABEL:  test_local_add_2f16_noret
// GFX940: ds_pk_add_f16
void test_local_add_2f16_noret(__local half2 *addr, half2 x) {
  __builtin_amdgcn_ds_atomic_fadd_v2f16(addr, x);
}

// CHECK-LABEL: @test_global_add_f32
// CHECK: = atomicrmw fadd ptr addrspace(1) %{{.+}}, float %{{.+}} syncscope("agent") monotonic, align 4, !amdgpu.no.fine.grained.memory !{{[0-9]+}}, !amdgpu.ignore.denormal.mode !{{[0-9]+$}}
void test_global_add_f32(float *rtn, global float *addr, float x) {
  *rtn = __builtin_amdgcn_global_atomic_fadd_f32(addr, x);
}

// RUN: %clang_cc1 -no-opaque-pointers -O0 -cl-std=CL2.0 -triple amdgcn-amd-amdhsa -target-cpu gfx810 \
// RUN:   %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -O0 -cl-std=CL2.0 -triple amdgcn-amd-amdhsa -target-cpu gfx810 \
// RUN:   -S -o - %s | FileCheck -check-prefix=GFX8 %s

// REQUIRES: amdgpu-registered-target

// CHECK-LABEL: test_fadd_local
// CHECK: call float @llvm.amdgcn.ds.fadd.f32(float addrspace(3)* %{{.*}}, float %{{.*}}, i32 0, i32 0, i1 false)
// GFX8-LABEL: test_fadd_local$local:
// GFX8: ds_add_rtn_f32 v2, v0, v1
// GFX8: s_endpgm
kernel void test_fadd_local(__local float *ptr, float val){
    float *res;
    *res = __builtin_amdgcn_ds_atomic_fadd_f32(ptr, val);
}

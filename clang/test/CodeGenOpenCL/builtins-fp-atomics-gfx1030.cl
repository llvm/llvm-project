// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -O0 -cl-std=CL2.0 -triple amdgcn-amd-amdhsa -target-cpu gfx1030 \
// RUN:   -S -o - %s
// RUN: %clang_cc1 -O0 -cl-std=CL2.0 -triple amdgcn-amd-amdhsa -target-cpu gfx1030 \
// RUN:   -S -o - %s | FileCheck -check-prefix=GFX1030 %s

// CHECK-LABEL: test_ds_addf_local
// CHECK: call float @llvm.amdgcn.ds.fadd.f32(float addrspace(3)* %{{.*}}, float %{{.*}},
// GFX1030-LABEL:  test_ds_addf_local$local
// GFX1030:  ds_add_rtn_f32
void test_ds_addf_local(__local float *addr, float x){
  float *rtn;
  *rtn = __builtin_amdgcn_ds_atomic_fadd_f32(addr, x);
}

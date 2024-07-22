// RUN: %clang_cc1 -O0 -cl-std=CL2.0 -triple amdgcn-amd-amdhsa -target-cpu gfx810 \
// RUN:   %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -O0 -cl-std=CL2.0 -triple amdgcn-amd-amdhsa -target-cpu gfx810 \
// RUN:   -S -o - %s | FileCheck -check-prefix=GFX8 %s

// REQUIRES: amdgpu-registered-target

// CHECK-LABEL: test_fadd_local
// CHECK: = atomicrmw fadd ptr addrspace(3) %{{.+}}, float %{{.+}} seq_cst, align 4
// GFX8-LABEL: test_fadd_local$local:
// GFX8: ds_add_rtn_f32 v2, v0, v1
// GFX8: s_endpgm
kernel void test_fadd_local(__local float *ptr, float val){
    float *res;
    *res = __builtin_amdgcn_ds_atomic_fadd_f32(ptr, val);
}

// CHECK-LABEL: test_fadd_local_volatile
// CHECK: = atomicrmw volatile fadd ptr addrspace(3) %{{.+}}, float %{{.+}} seq_cst, align 4
kernel void test_fadd_local_volatile(volatile __local float *ptr, float val){
    volatile float *res;
    *res = __builtin_amdgcn_ds_atomic_fadd_f32(ptr, val);
}

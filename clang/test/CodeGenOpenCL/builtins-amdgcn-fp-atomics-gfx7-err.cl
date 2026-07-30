// RUN: %clang_cc1 -O0 -cl-std=CL2.0 -triple amdgpu7.00-amd-amdhsa \
// RUN:   %s -verify -S -o -

kernel void test_fadd_local(__local float *ptr, float val){
    float *res;
    *res = __builtin_amdgcn_ds_atomic_fadd_f32(ptr, val); // expected-error{{'__builtin_amdgcn_ds_atomic_fadd_f32' needs target feature gfx8-insts}}
}

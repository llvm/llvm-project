
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// CHECK-LABEL: {{^}}test_rsqrt_f16:
// CHECK: s_waitcnt
// CHECK-NEXT: v_rsq_f16{{(_e32)?}} v0, v0
// CHECK-NEXT: s_setpc_b64
half test_rsqrt_f16(half x) {
    return rsqrt(x);
}

// CHECK-LABEL: {{^}}test_rsqrt_f32:
// IEEE: v_mul_f32
// IEEE: v_cmp_gt_f32
// IEEE: v_cndmask_b32
// IEEE: v_rsq_f32
// IEEE: v_mul_f32
// IEEE: v_cndmask_b32

// DAZ: s_waitcnt
// DAZ-NEXT: v_rsq_f32{{(_e32)?}} v0, v0
// DAZ-NEXT: s_setpc_b64
float test_rsqrt_f32(float x) {
    return rsqrt(x);
}

// CHECK-LABEL: {{^}}test_rsqrt_f64:
// CHECK: v_rsq_f64
// CHECK: v_mul_f64
// CHECK: v_fma_f64
// CHECK: v_mul_f64
// CHECK: v_fma_f64
// CHECK: v_fma_f64
// CHECK: v_cndmask_b32
// CHECK: v_cndmask_b32
double test_rsqrt_f64(double x) {
    return rsqrt(x);
}

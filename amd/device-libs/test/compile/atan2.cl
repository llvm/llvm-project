
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// GCN: {{^}}test_atan2_f16:
// GFX700: v_cvt_f32_f16
// GFX700: v_mul_f32
// GFX700: v_div_scale_f32
// GFX700: v_div_scale_f32
// GFX700: v_cmp_class_f32
// GFX700: v_cmp_class_f32
// GFX700: v_div_fixup_f32
// GFX700: v_bfi_b32

// GFX803: v_max_f16
// GFX803: v_rcp_f32
// GFX803: v_mul_f32
// GFX803: v_fma_f16
// GFX803: v_cmp_o_f16
// GFX803: v_bfi_b32
kernel void test_atan2_f16(global half* restrict out, global half* restrict in0, global half* restrict in1) {
    int id = get_local_id(0);
    out[id] = atan2(in0[id], in1[id]);
}

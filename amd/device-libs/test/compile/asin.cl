
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// GCN: {{^}}test_asin_f16:
// GFX700: v_cvt_f32_f16{{(_e32)?}} [[CVT:v[0-9]+]]
// GFX700: v_cmp_le_f32{{(_e64)?}} s{{\[[0-9]+:[0-9]+\]}}, |[[CVT]]|, 0.5
// GFX700: v_mul_f32
// GFX700: v_mad_f32
// GFX700: v_sqrt_f32
// GFX700: v_bfi_b32
// GFX700: v_cvt_f16_f32


// GFX803: v_cmp_le_f16{{(_e64)?}} s{{\[[0-9]+:[0-9]+\]}}, |{{v[0-9]+}}|, 0.5
// GFX803: v_mad_f32
// GFX803: v_sqrt_f32
// GFX803: v_bfi_b32
kernel void test_asin_f16(global half* restrict out, global half* restrict in) {
    int id = get_local_id(0);
    out[id] = asin(in[id]);
}


#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// FIXME: OpenCL's native_rsqrt doesn't seem wired up to the ocml
// functions for f16/f64

half __ocml_native_rsqrt_f16(half);

// FIXME: Promoted case using full expansion
// GCN-LABEL: {{^}}test_native_rsqrt_f16:
// GFX600: v_sqrt_f32
// GFX600: v_rcp_f32

// GFX700: v_sqrt_f32
// GFX700: v_rcp_f32

// GFX803: v_rsq_f16
kernel void test_native_rsqrt_f16(global half* restrict out, global half* restrict in) {
    int id = get_local_id(0);
    out[id] = __ocml_native_rsqrt_f16(in[id]);
}

// GCN-LABEL: {{^}}test_native_rsqrt_f32:
// GCN: v_rsq_f32
kernel void test_native_rsqrt_f32(global float* restrict out, global float* restrict in) {
    int id = get_local_id(0);
    out[id] = native_rsqrt(in[id]);
}

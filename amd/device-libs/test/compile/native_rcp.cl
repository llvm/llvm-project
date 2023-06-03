
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// FIXME: OpenCL's native_recip doesn't seem wired up to the ocml
// functions for f16/f64

half __ocml_native_rcp_f16(half);

// GCN: {{^}}test_native_recip_f16:
// GFX600: v_rcp_f32
// GFX700: v_rcp_f32
// GFX803: v_rcp_f16
kernel void test_native_recip_f16(global half* restrict out, global half* restrict in) {
    int id = get_local_id(0);
    out[id] = __ocml_native_rcp_f16(in[id]);
}

// GCN: {{^}}test_native_recip_f32:
// GCN: v_rcp_f32
kernel void test_native_recip_f32(global float* restrict out, global float* restrict in) {
    int id = get_local_id(0);
    out[id] = native_recip(in[id]);
}

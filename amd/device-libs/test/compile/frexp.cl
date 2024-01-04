
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Test that a hardware bug is worked around for gfx6, not applied
// later.

// GCN-LABEL: {{^}}test_frexp_f32:
// GFX600-DAG: s_mov_b32 [[INF:s[0-9]+]], 0x7f80000
// GFX600-DAG: v_frexp_mant_f32{{(_e32)?}} [[MANT:v[0-9]+]], [[SRC:v[0-9]+]]
// GFX600-DAG: v_frexp_exp_i32_f32{{(_e32)?}} [[EXP:v[0-9]+]], [[SRC:v[0-9]+]]

// GFX600-DAG: v_cmp_lt_f32{{(_e64)?}} [[CMP:(vcc|s{{\[[0-9]+:[0-9]+\]}})]], |[[SRC]]|, [[INF]]

// GFX600-DAG: v_cndmask_b32{{(_e32)?|(e64)?}} v{{[0-9]+}}, [[SRC]], [[MANT]], [[CMP]]
// GFX600-DAG: v_cndmask_b32{{(_e32)?|(e64)?}} v{{[0-9]+}}, 0, [[EXP]], [[CMP]]


// GFX700-NOT: v_cmp_class
// GFX700-DAG: v_frexp_mant_f32{{(_e32)?}} [[MANT:v[0-9]+]], [[SRC:v[0-9]+]]
// GFX700-DAG: v_frexp_exp_i32_f32{{(_e32)?}} [[EXP:v[0-9]+]], [[SRC:v[0-9]+]]
// GFX700-NOT: v_cmp_class
kernel void test_frexp_f32(global float* restrict out0,
                           global int* restrict out1,
                           global float* restrict in) {
    int id = get_local_id(0);

    int exponent;
    out0[id] = frexp(in[id], &exponent);
    out1[id] = exponent;
}

// GCN-LABEL: {{^}}test_frexp_f64:
// GFX600: s_mov_b32 s{{[0-9]+}}, 0{{$}}

// GFX600-DAG: s_mov_b32 s[[INF_LO:[0-9]+]], 0{{$}}
// GFX600-DAG: s_mov_b32 s[[INF_HI:[0-9]+]], 0x7ff00000{{$}}
// GFX600-DAG: v_frexp_mant_f64{{(_e32)?}} v{{\[}}[[MANT_LO:[0-9]+]]:[[MANT_HI:[0-9]+]]{{\]}}, [[SRC:v\[[0-9]+:[0-9]+\]]]
// GFX600-DAG: v_frexp_exp_i32_f64{{(_e32)?}} [[EXP:v[0-9]+]], [[SRC:v\[[0-9]+:[0-9]+\]]]

// GFX600-DAG: v_cmp_lt_f64{{(_e64)?}} [[CMP:(vcc|s{{\[[0-9]+:[0-9]+\]}})]], |[[SRC]]|, s{{\[}}[[INF_LO]]:[[INF_HI]]{{\]}}

// GFX600-DAG: v_cndmask_b32{{(_e32)?|(e64)?}} v{{[0-9]+}}, v{{[0-9]+}}, v[[MANT_LO]], [[CMP]]
// GFX600-DAG: v_cndmask_b32{{(_e32)?|(e64)?}} v{{[0-9]+}}, v{{[0-9]+}}, v[[MANT_HI]], [[CMP]]
// GFX600-DAG: v_cndmask_b32{{(_e32)?|(e64)?}} v{{[0-9]+}}, 0, [[EXP]], [[CMP]]


// GFX700-NOT: v_cmp_class
// GFX700-DAG: v_frexp_mant_f64
// GFX700-DAG: v_frexp_exp_i32_f64
// GFX700-NOT: v_cmp_class
kernel void test_frexp_f64(global double* restrict out0,
                           global int* restrict out1,
                           global double* restrict in) {
    int id = get_local_id(0);

    int exponent;
    out0[id] = frexp(in[id], &exponent);
    out1[id] = exponent;
}

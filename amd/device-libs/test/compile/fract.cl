#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// CHECK-LABEL: test_fract_f16
// GFX600: v_cvt_f32_f16
// GFX600-DAG: v_floor_f32
// GFX600-DAG: v_sub_f32
// GFX600-DAG: v_min_f32_e32 v{{[0-9]+}}, 0x3f7fe000,
// GFX600-DAG: v_cmp_u_f32
// GFX600-DAG: v_cmp_neq_f32
// GFX600-DAG: v_cndmask_b32
// GFX600-DAG: v_cvt_f16_f32
// GFX600-DAG: v_cvt_f16_f32


// TODO: Could promote the f16 pattern to f32
// GFX700: flat_load_ushort [[VAL:v[0-9]+]]
// GFX700: v_cvt_f32_f16_e32 [[VAL_F32:v[0-9]+]]
// GFX700-DAG: v_floor_f32_e32 [[FLOOR:v[0-9]+]], [[VAL_F32]]
// GFX700: v_sub_f32_e32 [[SUB:v[0-9]+]], [[VAL_F32]], [[FLOOR]]

// GFX700-DAG: v_min_f32_e32 [[CLAMP:v[0-9]+]], 0x3f7fe000, [[SUB]]
// GFX700-DAG: v_cmp_u_f32
// GFX700-DAG: v_cmp_neq_f32
// GFX700-DAG: v_cndmask_b32
// GFX700-DAG: v_cvt_f16_f32
// GFX700-DAG: v_cvt_f16_f32

// GFX803: flat_load_ushort [[VAL:v[0-9]+]]
// GFX803-DAG: v_floor_f16_e32 [[FLOOR:v[0-9]+]], [[VAL]]
// GFX803-DAG: v_fract_f16_e32 [[FRACT:v[0-9]+]], [[VAL]]
// GFX803-DAG: s_movk_i32 [[INF:s[0-9]+]], 0x7c00
// GFX803: v_cmp_neq_f16_e64 [[FINITE:(vcc)?(s\[[[[0-9]+:[0-9]+\]]])?]], |[[VAL]]|, [[INF]]
// GFX803: v_cndmask_b32_e32 [[SELECT:v[0-9]+]], 0, [[FRACT]]
// GFX803: flat_store_short v{{\[[0-9]+:[0-9]+\]}}, [[FLOOR]]
// GFX803: flat_store_short v{{\[[0-9]+:[0-9]+\]}}, [[SELECT]]
kernel void test_fract_f16(global half* restrict out0,
                           global half* restrict out1,
                           global half* restrict in) {
    int id = get_local_id(0);
    out0[id] = fract(in[id], &out1[id]);
}

// CHECK-LABEL: test_fract_f32
// GFX600-DAG: v_floor_f32
// GFX600-DAG: v_sub_f32
// GFX600-DAG: v_min_f32_e32 v{{[0-9]+}}, 0x3f7fffff,
// GFX600-DAG: v_cmp_u_f32
// GFX600-DAG: v_cndmask_b32
// GFX600-DAG: v_cmp_neq_f32
// GFX600-DAG: v_cndmask_b32


// GFX803: flat_load_dword [[VAL:v[0-9]+]]
// GFX803-DAG: v_floor_f32_e32 [[FLOOR:v[0-9]+]], [[VAL]]
// GFX803-DAG: v_fract_f32_e32 [[FRACT:v[0-9]+]], [[VAL]]
// GFX803-DAG: s_mov_b32 [[INF:s[0-9]+]], 0x7f800000
// GFX803: v_cmp_neq_f32_e64 [[FINITE:(vcc)?(s\[[[[0-9]+:[0-9]+\]]])?]], |[[VAL]]|, [[INF]]
// GFX803: v_cndmask_b32_e32 [[SELECT:v[0-9]+]], 0, [[FRACT]]
// GFX803: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[FLOOR]]
// GFX803: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[SELECT]]
kernel void test_fract_f32(global float* restrict out0,
                           global float* restrict out1,
                           global float* restrict in) {
    int id = get_local_id(0);
    out0[id] = fract(in[id], &out1[id]);
}

// CHECK-LABEL: test_fract_f64

// Fract is used in floor expansion, not directly for fract
// GFX600: v_fract_f64_e32
// GFX600: v_cmp_class_f64_e64
// GFX600: v_min_f64
// GFX600: v_cndmask_b32
// GFX600: v_cndmask_b32
// GFX600: v_add_f64
// GFX600: v_cmp_u_f64
// GFX600: v_add_f64
// GFX600: v_min_f64
// GFX600: v_cmp_neq_f64


// GFX700: flat_load_dwordx2 [[VAL:v[[0-9]+:[0-9]+]]]
// GFX700-DAG: v_floor_f64_e32 [[FLOOR:v\[[0-9]+:[0-9]+\]]], [[VAL]]

// GFX700-DAG: v_fract_f64_e32 v{{\[}}[[FRACT_LO:[0-9]+]]:[[FRACT_HI:[0-9]+]]{{\]}}, [[VAL]]

// GFX700-DAG: s_mov_b32 s[[INF_HI:[0-9]+]], 0x7ff00000
// GFX700-DAG: s_mov_b32 s[[INF_LO:[0-9]+]], 0{{$}}
// GFX700-DAG: v_cmp_neq_f64_e64 [[FINITE:(vcc)?(s\[[[[0-9]+:[0-9]+\]]])?]], |[[VAL]]|, s{{\[}}[[INF_LO]]:[[INF_HI]]{{\]}}

// GFX700-DAG: v_cndmask_b32_e32 v[[SELECT0:[0-9]+]], 0, v[[FRACT_LO]]
// GFX700-DAG: v_cndmask_b32_e32 v[[SELECT1:[0-9]+]], 0, v[[FRACT_HI]]
// GFX700: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[FLOOR]]
// GFX700: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, v{{\[}}[[SELECT0]]:[[SELECT1]]{{\]}}


// GFX803: flat_load_dwordx2 [[VAL:v[[0-9]+:[0-9]+]]]
// GFX803-DAG: v_floor_f64_e32 [[FLOOR:v\[[0-9]+:[0-9]+\]]], [[VAL]]
// GFX803-DAG: v_fract_f64_e32 v{{\[}}[[FRACT_LO:[0-9]+]]:[[FRACT_HI:[0-9]+]]{{\]}}, [[VAL]]

// GFX803-DAG: s_mov_b32 s[[INF_HI:[0-9]+]], 0x7ff00000
// GFX803-DAG: s_mov_b32 s[[INF_LO:[0-9]+]], 0{{$}}
// GFX803-DAG: v_cmp_neq_f64_e64 [[FINITE:(vcc)?(s\[[[[0-9]+:[0-9]+\]]])?]], |[[VAL]]|, s{{\[}}[[INF_LO]]:[[INF_HI]]{{\]}}

// GFX803-DAG: v_cndmask_b32_e32 v[[SELECT0:[0-9]+]], 0, v[[FRACT_LO]]
// GFX803-DAG: v_cndmask_b32_e32 v[[SELECT1:[0-9]+]], 0, v[[FRACT_HI]]
// GFX803: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[FLOOR]]
// GFX803: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, v{{\[}}[[SELECT0]]:[[SELECT1]]{{\]}}
kernel void test_fract_f64(global double* restrict out0,
                           global double* restrict out1,
                           global double* restrict in) {
    int id = get_local_id(0);
    out0[id] = fract(in[id], &out1[id]);
}

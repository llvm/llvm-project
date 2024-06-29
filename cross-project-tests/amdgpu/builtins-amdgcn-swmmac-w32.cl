// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1200 -target-feature +wavefrontsize32 -S -o - %s | FileCheck %s --check-prefix=CHECK-GFX1200

typedef int    v2i   __attribute__((ext_vector_type(2)));
typedef int    v4i   __attribute__((ext_vector_type(4)));
typedef float  v8f   __attribute__((ext_vector_type(8)));
typedef half   v8h   __attribute__((ext_vector_type(8)));
typedef short  v8s   __attribute__((ext_vector_type(8)));
typedef int    v8i   __attribute__((ext_vector_type(8)));
typedef half  v16h   __attribute__((ext_vector_type(16)));
typedef short v16s   __attribute__((ext_vector_type(16)));

// Wave32


// CHECK-GFX1200-LABEL: test_amdgcn_swmmac_f32_16x16x32_f16_w32:
// CHECK-GFX1200: v_swmmac_f32_16x16x32_f16 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}
//
void test_amdgcn_swmmac_f32_16x16x32_f16_w32(global v8f* out, v8h a, v16h b, v8f c, short index)
{
  *out = __builtin_amdgcn_swmmac_f32_16x16x32_f16_w32(a, b, c, index);
}


// CHECK-GFX1200-LABEL: test_amdgcn_swmmac_f32_16x16x32_bf16_w32:
// CHECK-GFX1200: v_swmmac_f32_16x16x32_bf16 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}
//
void test_amdgcn_swmmac_f32_16x16x32_bf16_w32(global v8f* out, v8s a, v16s b, v8f c, short index)
{
  *out = __builtin_amdgcn_swmmac_f32_16x16x32_bf16_w32(a, b, c, index);
}


// CHECK-GFX1200-LABEL: test_amdgcn_swmmac_f16_16x16x32_f16_w32:
// CHECK-GFX1200: v_swmmac_f16_16x16x32_f16 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}
//
void test_amdgcn_swmmac_f16_16x16x32_f16_w32(global v8h* out, v8h a, v16h b, v8h c, short index)
{
  *out = __builtin_amdgcn_swmmac_f16_16x16x32_f16_w32(a, b, c, index);
}


// CHECK-GFX1200-LABEL: test_amdgcn_swmmac_bf16_16x16x32_bf16_w32:
// CHECK-GFX1200: v_swmmac_bf16_16x16x32_bf16 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}
//
void test_amdgcn_swmmac_bf16_16x16x32_bf16_w32(global v8s* out, v8s a, v16s b, v8s c, short index)
{
  *out = __builtin_amdgcn_swmmac_bf16_16x16x32_bf16_w32(a, b, c, index);
}


// CHECK-GFX1200-LABEL: test_amdgcn_swmmac_i32_16x16x32_iu8_w32:
// CHECK-GFX1200: v_swmmac_i32_16x16x32_iu8 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} neg_lo:[1,1,0] clamp
//
void test_amdgcn_swmmac_i32_16x16x32_iu8_w32(global v8i* out, v2i a, v4i b, v8i c, short index)
{
  *out = __builtin_amdgcn_swmmac_i32_16x16x32_iu8_w32(true, a, true, b, c, index, true);
}


// CHECK-GFX1200-LABEL: test_amdgcn_swmmac_i32_16x16x32_iu4_w32:
// CHECK-GFX1200: v_swmmac_i32_16x16x32_iu4 v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} neg_lo:[1,1,0] clamp
//
void test_amdgcn_swmmac_i32_16x16x32_iu4_w32(global v8i* out, int a, v2i b, v8i c, short index)
{
  *out = __builtin_amdgcn_swmmac_i32_16x16x32_iu4_w32(true, a, true, b, c, index, true);
}


// CHECK-GFX1200-LABEL: test_amdgcn_swmmac_i32_16x16x64_iu4_w32:
// CHECK-GFX1200: v_swmmac_i32_16x16x64_iu4 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} neg_lo:[1,1,0] clamp
//
void test_amdgcn_swmmac_i32_16x16x64_iu4_w32(global v8i* out, v2i a, v4i b, v8i c, short index)
{
  *out = __builtin_amdgcn_swmmac_i32_16x16x64_iu4_w32(true, a, true, b, c, index, true);
}


// CHECK-GFX1200-LABEL: test_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32:
// CHECK-GFX1200: v_swmmac_f32_16x16x32_fp8_fp8 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}
//
void test_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32(global v8f* out, v2i a, v4i b, v8f c, short index)
{
  *out = __builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32(a, b, c, index);
}


// CHECK-GFX1200-LABEL: test_amdgcn_swmmac_f32_16x16x32_fp8_bf8_w32:
// CHECK-GFX1200: v_swmmac_f32_16x16x32_fp8_bf8 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}
//
void test_amdgcn_swmmac_f32_16x16x32_fp8_bf8_w32(global v8f* out, v2i a, v4i b, v8f c, short index)
{
  *out = __builtin_amdgcn_swmmac_f32_16x16x32_fp8_bf8_w32(a, b, c, index);
}


// CHECK-GFX1200-LABEL: test_amdgcn_swmmac_f32_16x16x32_bf8_fp8_w32:
// CHECK-GFX1200: v_swmmac_f32_16x16x32_bf8_fp8 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}
//
void test_amdgcn_swmmac_f32_16x16x32_bf8_fp8_w32(global v8f* out, v2i a, v4i b, v8f c, short index)
{
  *out = __builtin_amdgcn_swmmac_f32_16x16x32_bf8_fp8_w32(a, b, c, index);
}

// CHECK-GFX1200-LABEL: test_amdgcn_swmmac_f32_16x16x32_bf8_bf8_w32:
// CHECK-GFX1200: v_swmmac_f32_16x16x32_bf8_bf8 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}
//
void test_amdgcn_swmmac_f32_16x16x32_bf8_bf8_w32(global v8f* out, v2i a, v4i b, v8f c, short index)
{
  *out = __builtin_amdgcn_swmmac_f32_16x16x32_bf8_bf8_w32(a, b, c, index);
}

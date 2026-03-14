// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1200 -target-feature +wavefrontsize64 -S -o - %s | FileCheck %s --check-prefix=CHECK-GFX1200

typedef int    v2i   __attribute__((ext_vector_type(2)));
typedef int    v4i   __attribute__((ext_vector_type(4)));
typedef float  v4f   __attribute__((ext_vector_type(4)));
typedef half   v4h   __attribute__((ext_vector_type(4)));
typedef short  v4s   __attribute__((ext_vector_type(4)));
typedef half   v8h   __attribute__((ext_vector_type(8)));
typedef short  v8s   __attribute__((ext_vector_type(8)));

// Wave64


// CHECK-GFX1200-LABEL: test_amdgcn_swmmac_f32_16x16x32_f16_w64:
// CHECK-GFX1200: v_swmmac_f32_16x16x32_f16 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}
//
void test_amdgcn_swmmac_f32_16x16x32_f16_w64(global v4f* out, v4h a, v8h b, v4f c, short index)
{
  *out = __builtin_amdgcn_swmmac_f32_16x16x32_f16_w64(a, b, c, index);
}


// CHECK-GFX1200-LABEL: test_amdgcn_swmmac_f32_16x16x32_bf16_w64:
// CHECK-GFX1200: v_swmmac_f32_16x16x32_bf16 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}
//
void test_amdgcn_swmmac_f32_16x16x32_bf16_w64(global v4f* out, v4s a, v8s b, v4f c, short index)
{
  *out = __builtin_amdgcn_swmmac_f32_16x16x32_bf16_w64(a, b, c, index);
}


// CHECK-GFX1200-LABEL: test_amdgcn_swmmac_f16_16x16x32_f16_w64:
// CHECK-GFX1200: v_swmmac_f16_16x16x32_f16 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}
//
void test_amdgcn_swmmac_f16_16x16x32_f16_w64(global v4h* out, v4h a, v8h b, v4h c, short index)
{
  *out = __builtin_amdgcn_swmmac_f16_16x16x32_f16_w64(a, b, c, index);
}


// CHECK-GFX1200-LABEL: test_amdgcn_swmmac_bf16_16x16x32_bf16_w64:
// CHECK-GFX1200: v_swmmac_bf16_16x16x32_bf16 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}
//
void test_amdgcn_swmmac_bf16_16x16x32_bf16_w64(global v4s* out, v4s a, v8s b, v4s c, short index)
{
  *out = __builtin_amdgcn_swmmac_bf16_16x16x32_bf16_w64(a, b, c, index);
}


// CHECK-GFX1200-LABEL: test_amdgcn_swmmac_i32_16x16x32_iu8_w64:
// CHECK-GFX1200: v_swmmac_i32_16x16x32_iu8 v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} neg_lo:[1,1,0] clamp
//
void test_amdgcn_swmmac_i32_16x16x32_iu8_w64(global v4i* out, int a, v2i b, v4i c, short index)
{
  *out = __builtin_amdgcn_swmmac_i32_16x16x32_iu8_w64(true, a, true, b, c, index, true);
}


// CHECK-GFX1200-LABEL: test_amdgcn_swmmac_i32_16x16x32_iu4_w64:
// CHECK-GFX1200: v_swmmac_i32_16x16x32_iu4 v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} neg_lo:[1,1,0] clamp
//
void test_amdgcn_swmmac_i32_16x16x32_iu4_w64(global v4i* out, int a, int b, v4i c, short index)
{
  *out = __builtin_amdgcn_swmmac_i32_16x16x32_iu4_w64(true, a, true, b, c, index, true);
}


// CHECK-GFX1200-LABEL: test_amdgcn_swmmac_i32_16x16x64_iu4_w64:
// CHECK-GFX1200: v_swmmac_i32_16x16x64_iu4 v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} neg_lo:[1,1,0] clamp
//
void test_amdgcn_swmmac_i32_16x16x64_iu4_w64(global v4i* out, int a, v2i b, v4i c, short index)
{
  *out = __builtin_amdgcn_swmmac_i32_16x16x64_iu4_w64(true, a, true, b, c, index, true);
}


// CHECK-GFX1200-LABEL: test_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w64:
// CHECK-GFX1200: v_swmmac_f32_16x16x32_fp8_fp8 v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}
//
void test_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w64(global v4f* out, int a, v2i b, v4f c, short index)
{
  *out = __builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w64(a, b, c, index);
}


// CHECK-GFX1200-LABEL: test_amdgcn_swmmac_f32_16x16x32_fp8_bf8_w64:
// CHECK-GFX1200: v_swmmac_f32_16x16x32_fp8_bf8 v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}
//
void test_amdgcn_swmmac_f32_16x16x32_fp8_bf8_w64(global v4f* out, int a, v2i b, v4f c, short index)
{
  *out = __builtin_amdgcn_swmmac_f32_16x16x32_fp8_bf8_w64(a, b, c, index);
}


// CHECK-GFX1200-LABEL: test_amdgcn_swmmac_f32_16x16x32_bf8_fp8_w64:
// CHECK-GFX1200: v_swmmac_f32_16x16x32_bf8_fp8 v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}
//
void test_amdgcn_swmmac_f32_16x16x32_bf8_fp8_w64(global v4f* out, int a, v2i b, v4f c, short index)
{
  *out = __builtin_amdgcn_swmmac_f32_16x16x32_bf8_fp8_w64(a, b, c, index);
}

// CHECK-GFX1200-LABEL: test_amdgcn_swmmac_f32_16x16x32_bf8_bf8_w64:
// CHECK-GFX1200: v_swmmac_f32_16x16x32_bf8_bf8 v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}
//
void test_amdgcn_swmmac_f32_16x16x32_bf8_bf8_w64(global v4f* out, int a, v2i b, v4f c, short index)
{
  *out = __builtin_amdgcn_swmmac_f32_16x16x32_bf8_bf8_w64(a, b, c, index);
}

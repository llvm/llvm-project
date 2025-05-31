// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1200 -target-feature +wavefrontsize64 -S -o - %s | FileCheck %s --check-prefix=CHECK-GFX1200

typedef float  v4f   __attribute__((ext_vector_type(4)));
typedef half   v4h   __attribute__((ext_vector_type(4)));
typedef short  v4s   __attribute__((ext_vector_type(4)));
typedef int    v4i   __attribute__((ext_vector_type(4)));

// Wave64


// CHECK-GFX1200-LABEL: test_amdgcn_wmma_f32_16x16x16_f16_w64:
// CHECK-GFX1200: v_wmma_f32_16x16x16_f16 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
//
void test_amdgcn_wmma_f32_16x16x16_f16_w64(global v4f* out, v4h a, v4h b, v4f c)
{
  *out = __builtin_amdgcn_wmma_f32_16x16x16_f16_w64_gfx12(a, b, c);
}


// CHECK-GFX1200-LABEL: test_amdgcn_wmma_f32_16x16x16_bf16_w64:
// CHECK-GFX1200: v_wmma_f32_16x16x16_bf16 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
//
void test_amdgcn_wmma_f32_16x16x16_bf16_w64(global v4f* out, v4s a, v4s b, v4f c)
{
  *out = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w64_gfx12(a, b, c);
}


// CHECK-GFX1200-LABEL: test_amdgcn_wmma_f16_16x16x16_f16_w64:
// CHECK-GFX1200: v_wmma_f16_16x16x16_f16 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
//
void test_amdgcn_wmma_f16_16x16x16_f16_w64(global v4h* out, v4h a, v4h b, v4h c)
{
  *out = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64_gfx12(a, b, c);
}


// CHECK-GFX1200-LABEL: test_amdgcn_wmma_bf16_16x16x16_bf16_w64:
// CHECK-GFX1200: v_wmma_bf16_16x16x16_bf16 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
//
void test_amdgcn_wmma_bf16_16x16x16_bf16_w64(global v4s* out, v4s a, v4s b, v4s c)
{
  *out = __builtin_amdgcn_wmma_bf16_16x16x16_bf16_w64_gfx12(a, b, c);
}


// CHECK-GFX1200-LABEL: test_amdgcn_wmma_i32_16x16x16_iu8_w64:
// CHECK-GFX1200: v_wmma_i32_16x16x16_iu8 v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}] neg_lo:[1,1,0]
//
void test_amdgcn_wmma_i32_16x16x16_iu8_w64(global v4i* out, int a, int b, v4i c)
{
  *out = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w64_gfx12(true, a, true, b, c, false);
}


// CHECK-GFX1200-LABEL: test_amdgcn_wmma_i32_16x16x16_iu4_w64:
// CHECK-GFX1200: v_wmma_i32_16x16x16_iu4 v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}] neg_lo:[1,1,0]
//
void test_amdgcn_wmma_i32_16x16x16_iu4_w64(global v4i* out, int a, int b, v4i c)
{
  *out = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w64_gfx12(true, a, true, b, c, false);
}


// CHECK-GFX1200-LABEL: test_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32:
// CHECK-GFX1200: v_wmma_f32_16x16x16_fp8_fp8 v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}]
//
void test_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32(global v4f* out, int a, int b, v4f c)
{
  *out = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w64_gfx12(a, b, c);
}


// CHECK-GFX1200-LABEL: test_amdgcn_wmma_f32_16x16x16_fp8_bf8_w32:
// CHECK-GFX1200: v_wmma_f32_16x16x16_fp8_bf8 v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}]
//
void test_amdgcn_wmma_f32_16x16x16_fp8_bf8_w32(global v4f* out, int a, int b, v4f c)
{
  *out = __builtin_amdgcn_wmma_f32_16x16x16_fp8_bf8_w64_gfx12(a, b, c);
}

// CHECK-GFX1200-LABEL: test_amdgcn_wmma_f32_16x16x16_bf8_fp8_w32:
// CHECK-GFX1200: v_wmma_f32_16x16x16_bf8_fp8 v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}]
//
void test_amdgcn_wmma_f32_16x16x16_bf8_fp8_w32(global v4f* out, int a, int b, v4f c)
{
  *out = __builtin_amdgcn_wmma_f32_16x16x16_bf8_fp8_w64_gfx12(a, b, c);
}

// CHECK-GFX1200-LABEL: test_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32:
// CHECK-GFX1200: v_wmma_f32_16x16x16_bf8_bf8 v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}]
//
void test_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32(global v4f* out, int a, int b, v4f c)
{
  *out = __builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w64_gfx12(a, b, c);
}

// CHECK-GFX1200-LABEL: test_amdgcn_wmma_i32_16x16x32_iu4_w32:
// CHECK-GFX1200: v_wmma_i32_16x16x32_iu4 v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}] neg_lo:[1,1,0]
//
void test_amdgcn_wmma_i32_16x16x32_iu4_w32(global v4i* out, int a, int b, v4i c)
{
  *out = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w64_gfx12(true, a, true, b, c, false);
}

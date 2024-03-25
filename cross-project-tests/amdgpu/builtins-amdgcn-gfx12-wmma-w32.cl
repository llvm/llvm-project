// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1200 -target-feature +wavefrontsize32 -S -o - %s | FileCheck %s --check-prefix=CHECK-GFX1200

typedef int    v2i   __attribute__((ext_vector_type(2)));
typedef float  v8f   __attribute__((ext_vector_type(8)));
typedef half   v8h   __attribute__((ext_vector_type(8)));
typedef short  v8s   __attribute__((ext_vector_type(8)));
typedef int    v8i   __attribute__((ext_vector_type(8)));

// Wave32


// CHECK-GFX1200-LABEL: test_amdgcn_wmma_f32_16x16x16_f16_w32:
// CHECK-GFX1200: v_wmma_f32_16x16x16_f16 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
//
void test_amdgcn_wmma_f32_16x16x16_f16_w32(global v8f* out, v8h a, v8h b, v8f c)
{
  *out = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
}


// CHECK-GFX1200-LABEL: test_amdgcn_wmma_f32_16x16x16_bf16_w32:
// CHECK-GFX1200: v_wmma_f32_16x16x16_bf16 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
//
void test_amdgcn_wmma_f32_16x16x16_bf16_w32(global v8f* out, v8s a, v8s b, v8f c)
{
  *out = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}


// CHECK-GFX1200-LABEL: test_amdgcn_wmma_f16_16x16x16_f16_w32:
// CHECK-GFX1200: v_wmma_f16_16x16x16_f16 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
//
void test_amdgcn_wmma_f16_16x16x16_f16_w32(global v8h* out, v8h a, v8h b, v8h c)
{
  *out = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12(a, b, c);
}


// CHECK-GFX1200-LABEL: test_amdgcn_wmma_bf16_16x16x16_bf16_w32:
// CHECK-GFX1200: v_wmma_bf16_16x16x16_bf16 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
void test_amdgcn_wmma_bf16_16x16x16_bf16_w32(global v8s* out, v8s a, v8s b, v8s c)
{
  *out = __builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32_gfx12(a, b, c);
}


// CHECK-GFX1200-LABEL: test_amdgcn_wmma_i32_16x16x16_iu8_w32:
// CHECK-GFX1200: v_wmma_i32_16x16x16_iu8 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}] neg_lo:[1,1,0]
//
void test_amdgcn_wmma_i32_16x16x16_iu8_w32(global v8i* out, v2i a, v2i b, v8i c)
{
  *out = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c, false);
}


// CHECK-GFX1200-LABEL: test_amdgcn_wmma_i32_16x16x16_iu4_w32:
// CHECK-GFX1200: v_wmma_i32_16x16x16_iu4 v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}] neg_lo:[1,1,0]
//
void test_amdgcn_wmma_i32_16x16x16_iu4_w32(global v8i* out, int a, int b, v8i c)
{
  *out = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12(true, a, true, b, c, false);
}


// CHECK-GFX1200-LABEL: test_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32:
// CHECK-GFX1200: v_wmma_f32_16x16x16_fp8_fp8 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
//
void test_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32(global v8f* out, v2i a, v2i b, v8f c)
{
  *out = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c);
}


// CHECK-GFX1200-LABEL: test_amdgcn_wmma_f32_16x16x16_fp8_bf8_w32:
// CHECK-GFX1200: v_wmma_f32_16x16x16_fp8_bf8 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
//
void test_amdgcn_wmma_f32_16x16x16_fp8_bf8_w32(global v8f* out, v2i a, v2i b, v8f c)
{
  *out = __builtin_amdgcn_wmma_f32_16x16x16_fp8_bf8_w32_gfx12(a, b, c);
}


// CHECK-GFX1200-LABEL: test_amdgcn_wmma_f32_16x16x16_bf8_fp8_w32:
// CHECK-GFX1200: v_wmma_f32_16x16x16_bf8_fp8 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
//
void test_amdgcn_wmma_f32_16x16x16_bf8_fp8_w32(global v8f* out, v2i a, v2i b, v8f c)
{
  *out = __builtin_amdgcn_wmma_f32_16x16x16_bf8_fp8_w32_gfx12(a, b, c);
}


// CHECK-GFX1200-LABEL: test_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32:
// CHECK-GFX1200: v_wmma_f32_16x16x16_bf8_bf8 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
//
void test_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32(global v8f* out, v2i a, v2i b, v8f c)
{
  *out = __builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12(a, b, c);
}


// CHECK-GFX1200-LABEL: test_amdgcn_wmma_i32_16x16x32_iu4_w32:
// CHECK-GFX1200: v_wmma_i32_16x16x32_iu4 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}] neg_lo:[1,1,0]
//
void test_amdgcn_wmma_i32_16x16x32_iu4_w32(global v8i* out, v2i a, v2i b, v8i c)
{
  *out = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true, a, true, b, c, false);
}

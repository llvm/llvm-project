// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1100 -target-feature +wavefrontsize32 -DWMMA_GFX1100_TESTS -S -o - %s | FileCheck %s --check-prefix=CHECK-GFX1100

typedef float  v4f   __attribute__((ext_vector_type(4)));
typedef float  v8f   __attribute__((ext_vector_type(8)));
typedef half   v16h  __attribute__((ext_vector_type(16)));
typedef int    v2i   __attribute__((ext_vector_type(2)));
typedef int    v4i   __attribute__((ext_vector_type(4)));
typedef int    v8i   __attribute__((ext_vector_type(8)));
typedef short  v16s  __attribute__((ext_vector_type(16)));

#ifdef WMMA_GFX1100_TESTS

// Wave32


// CHECK-GFX1100-LABEL: test_amdgcn_wmma_f32_16x16x16_f16_w32:
// CHECK-GFX1100: v_wmma_f32_16x16x16_f16 v[{{.*}}], v[{{.*}} v[{{.*}}], v[{{.*}}]
//
void test_amdgcn_wmma_f32_16x16x16_f16_w32(global v8f* out, v16h a, v16h b, v8f c)
{
  *out = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}


// CHECK-GFX1100-LABEL: test_amdgcn_wmma_f32_16x16x16_bf16_w32:
// CHECK-GFX1100: v_wmma_f32_16x16x16_bf16 v[{{.*}}], v[{{.*}} v[{{.*}}], v[{{.*}}]
//
void test_amdgcn_wmma_f32_16x16x16_bf16_w32(global v8f* out, v16s a, v16s b, v8f c)
{
  *out = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
}


// CHECK-GFX1100-LABEL: test_amdgcn_wmma_f16_16x16x16_f16_w32:
// CHECK-GFX1100: v_wmma_f16_16x16x16_f16 v[{{.*}}], v[{{.*}} v[{{.*}}], v[{{.*}}] op_sel:[0,0,1]
//
void test_amdgcn_wmma_f16_16x16x16_f16_w32(global v16h* out, v16h a, v16h b, v16h c)
{
  *out = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a, b, c, true);
}


// CHECK-GFX1100-LABEL: test_amdgcn_wmma_bf16_16x16x16_bf16_w32:
// CHECK-GFX1100: v_wmma_bf16_16x16x16_bf16 v[{{.*}}], v[{{.*}} v[{{.*}}], v[{{.*}}] op_sel:[0,0,1]
//
void test_amdgcn_wmma_bf16_16x16x16_bf16_w32(global v16s* out, v16s a, v16s b, v16s c)
{
  *out = __builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32(a, b, c, true);
}


// CHECK-GFX1100-LABEL: test_amdgcn_wmma_i32_16x16x16_iu8_w32:
// CHECK-GFX1100: v_wmma_i32_16x16x16_iu8 v[{{.*}}], v[{{.*}} v[{{.*}} v[{{.*}}] neg_lo:[1,1,0]
//
void test_amdgcn_wmma_i32_16x16x16_iu8_w32(global v8i* out, v4i a, v4i b, v8i c)
{
  *out = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, false);
}


// CHECK-GFX1100-LABEL: test_amdgcn_wmma_i32_16x16x16_iu4_w32:
// CHECK-GFX1100: v_wmma_i32_16x16x16_iu4 v[{{.*}}, v[{{.*}} v[{{.*}} v[{{.*}} neg_lo:[1,1,0]
void test_amdgcn_wmma_i32_16x16x16_iu4_w32(global v8i* out, v2i a, v2i b, v8i c)
{
  *out = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, c, false);
}

#endif

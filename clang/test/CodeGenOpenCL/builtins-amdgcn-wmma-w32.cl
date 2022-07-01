// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1100 -DWMMA_GFX1100_TESTS -S -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-GFX1100

typedef float  v4f   __attribute__((ext_vector_type(4)));
typedef float  v8f   __attribute__((ext_vector_type(8)));
typedef half   v16h  __attribute__((ext_vector_type(16)));
typedef int    v2i   __attribute__((ext_vector_type(2)));
typedef int    v4i   __attribute__((ext_vector_type(4)));
typedef int    v8i   __attribute__((ext_vector_type(8)));
typedef short  v16s  __attribute__((ext_vector_type(16)));

#ifdef WMMA_GFX1100_TESTS

// Wave32

//
// amdgcn_wmma_f32_16x16x16_f16
//

// CHECK-GFX1100-LABEL: @test_amdgcn_wmma_f32_16x16x16_f16_w32
// CHECK-GFX1100: call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32(<16 x half> %a, <16 x half> %b, <8 x float> %c)
void test_amdgcn_wmma_f32_16x16x16_f16_w32(global v8f* out, v16h a, v16h b, v8f c)
{
  *out = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

//
// amdgcn_wmma_f32_16x16x16_bf16
//

// CHECK-GFX1100-LABEL: @test_amdgcn_wmma_f32_16x16x16_bf16_w32
// CHECK-GFX1100: call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32(<16 x i16> %a, <16 x i16> %b, <8 x float> %c)
void test_amdgcn_wmma_f32_16x16x16_bf16_w32(global v8f* out, v16s a, v16s b, v8f c)
{
  *out = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
}

//
// amdgcn_wmma_f16_16x16x16_f16
//

// CHECK-GFX1100-LABEL: @test_amdgcn_wmma_f16_16x16x16_f16_w32
// CHECK-GFX1100: call <16 x half> @llvm.amdgcn.wmma.f16.16x16x16.f16.v16f16(<16 x half> %a, <16 x half> %b, <16 x half> %c, i1 true)
void test_amdgcn_wmma_f16_16x16x16_f16_w32(global v16h* out, v16h a, v16h b, v16h c)
{
  *out = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a, b, c, true);
}

//
// amdgcn_wmma_bf16_16x16x16_bf16
//

// CHECK-GFX1100-LABEL: @test_amdgcn_wmma_bf16_16x16x16_bf16_w32
// CHECK-GFX1100: call <16 x i16> @llvm.amdgcn.wmma.bf16.16x16x16.bf16.v16i16(<16 x i16> %a, <16 x i16> %b, <16 x i16> %c, i1 true)
void test_amdgcn_wmma_bf16_16x16x16_bf16_w32(global v16s* out, v16s a, v16s b, v16s c)
{
  *out = __builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32(a, b, c, true);
}

//
// amdgcn_wmma_i32_16x16x16_iu8
//

// CHECK-GFX1100-LABEL: @test_amdgcn_wmma_i32_16x16x16_iu8_w32
// CHECK-GFX1100: call <8 x i32> @llvm.amdgcn.wmma.i32.16x16x16.iu8.v8i32(i1 true, <4 x i32> %a, i1 true, <4 x i32> %b, <8 x i32> %c, i1 false)
void test_amdgcn_wmma_i32_16x16x16_iu8_w32(global v8i* out, v4i a, v4i b, v8i c)
{
  *out = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, false);
}

//
// amdgcn_wmma_i32_16x16x16_iu4
//

// CHECK-GFX1100-LABEL: @test_amdgcn_wmma_i32_16x16x16_iu4_w32
// CHECK-GFX1100: call <8 x i32> @llvm.amdgcn.wmma.i32.16x16x16.iu4.v8i32(i1 true, <2 x i32> %a, i1 true, <2 x i32> %b, <8 x i32> %c, i1 false)
void test_amdgcn_wmma_i32_16x16x16_iu4_w32(global v8i* out, v2i a, v2i b, v8i c)
{
  *out = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, c, false);
}

#endif

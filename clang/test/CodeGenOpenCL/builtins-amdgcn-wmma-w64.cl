// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1100 -DWMMA_GFX1100_TESTS -S -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-GFX1100

typedef float  v4f   __attribute__((ext_vector_type(4)));
typedef float  v8f   __attribute__((ext_vector_type(8)));
typedef half   v8h   __attribute__((ext_vector_type(8)));
typedef half   v16h  __attribute__((ext_vector_type(16)));
typedef int    v2i   __attribute__((ext_vector_type(2)));
typedef int    v4i   __attribute__((ext_vector_type(4)));
typedef int    v8i   __attribute__((ext_vector_type(8)));
typedef short  v8s   __attribute__((ext_vector_type(8)));
typedef short  v16s  __attribute__((ext_vector_type(16)));

#ifdef WMMA_GFX1100_TESTS

// Wave64

//
// amdgcn_wmma_f32_16x16x16_f16
//

// CHECK-GFX1100-LABEL: @test_amdgcn_wmma_f32_16x16x16_f16_w64
// CHECK-GFX1100: call <4 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v4f32(<16 x half> %a, <16 x half> %b, <4 x float> %c)
void test_amdgcn_wmma_f32_16x16x16_f16_w64(global v4f* out, v16h a, v16h b, v4f c)
{
  *out = __builtin_amdgcn_wmma_f32_16x16x16_f16_w64(a, b, c);
}

//
// amdgcn_wmma_f32_16x16x16_bf16
//

// CHECK-GFX1100-LABEL: @test_amdgcn_wmma_f32_16x16x16_bf16_w64
// CHECK-GFX1100: call <4 x float> @llvm.amdgcn.wmma.f32.16x16x16.bf16.v4f32(<16 x i16> %a, <16 x i16> %b, <4 x float> %c)
void test_amdgcn_wmma_f32_16x16x16_bf16_w64(global v4f* out, v16s a, v16s b, v4f c)
{
  *out = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w64(a, b, c);
}

//
// amdgcn_wmma_f16_16x16x16_f16
//

// CHECK-GFX1100-LABEL: @test_amdgcn_wmma_f16_16x16x16_f16_w64
// CHECK-GFX1100: call <8 x half> @llvm.amdgcn.wmma.f16.16x16x16.f16.v8f16(<16 x half> %a, <16 x half> %b, <8 x half> %c, i1 true)
void test_amdgcn_wmma_f16_16x16x16_f16_w64(global v8h* out, v16h a, v16h b, v8h c)
{
  *out = __builtin_amdgcn_wmma_f16_16x16x16_f16_w64(a, b, c, true);
}

//
// amdgcn_wmma_bf16_16x16x16_bf16
//

// CHECK-GFX1100-LABEL: @test_amdgcn_wmma_bf16_16x16x16_bf16_w64
// CHECK-GFX1100: call <8 x i16> @llvm.amdgcn.wmma.bf16.16x16x16.bf16.v8i16(<16 x i16> %a, <16 x i16> %b, <8 x i16> %c, i1 true)
void test_amdgcn_wmma_bf16_16x16x16_bf16_w64(global v8s* out, v16s a, v16s b, v8s c)
{
  *out = __builtin_amdgcn_wmma_bf16_16x16x16_bf16_w64(a, b, c, true);
}

//
// amdgcn_wmma_i32_16x16x16_iu8
//

// CHECK-GFX1100-LABEL: @test_amdgcn_wmma_i32_16x16x16_iu8_w64
// CHECK-GFX1100: call <4 x i32> @llvm.amdgcn.wmma.i32.16x16x16.iu8.v4i32(i1 true, <4 x i32> %a, i1 true, <4 x i32> %b, <4 x i32> %c, i1 false)
void test_amdgcn_wmma_i32_16x16x16_iu8_w64(global v4i* out, v4i a, v4i b, v4i c)
{
  *out = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w64(true, a, true, b, c, false);
}

//
// amdgcn_wmma_i32_16x16x16_iu4
//

// CHECK-GFX1100-LABEL: @test_amdgcn_wmma_i32_16x16x16_iu4_w64
// CHECK-GFX1100: call <4 x i32> @llvm.amdgcn.wmma.i32.16x16x16.iu4.v4i32(i1 true, <2 x i32> %a, i1 true, <2 x i32> %b, <4 x i32> %c, i1 false)
void test_amdgcn_wmma_i32_16x16x16_iu4_w64(global v4i* out, v2i a, v2i b, v4i c)
{
  *out = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w64(true, a, true, b, c, false);
}

#endif

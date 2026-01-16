// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512fp16 -target-feature +avx512f -target-feature +avx512vl -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion 
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512fp16 -target-feature +avx512f -target-feature +avx512vl -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512fp16 -target-feature +avx512f -target-feature +avx512vl -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=OGCG --input-file=%t.ll %s

#include <immintrin.h>

__m128 test_vcvtph2ps_mask(__m128i a, __m128 src, __mmask8 k) {
  // CIR-LABEL: cir.func no_inline dso_local @test_vcvtph2ps_mask
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !s16i>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i] : !cir.vector<4 x !s16i>
  // CIR: %{{.*}} = cir.cast floating %{{.*}} : !cir.vector<4 x !cir.f16> -> !cir.vector<4 x !cir.float>
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !cir.int<s, 1>>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i] : !cir.vector<4 x !cir.int<s, 1>>
  // CIR: %{{.*}} = cir.vec.ternary(%{{.*}}, %{{.*}}, %{{.*}}) : !cir.vector<4 x !cir.int<s, 1>>, !cir.vector<4 x !cir.float>

  // LLVM-LABEL: @test_vcvtph2ps_mask
  // LLVM: %{{.*}} = fpext <4 x half> %{{.*}} to <4 x float>
  // LLVM: %{{.*}} = select <4 x i1> {{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}

  // OGCG-LABEL: @test_vcvtph2ps_mask
  // OGCG: %{{.*}} = fpext <4 x half> %{{.*}} to <4 x float>
  // OGCG: %{{.*}} = select <4 x i1> {{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  typedef short __v8hi __attribute__((__vector_size__(16)));
  return __builtin_ia32_vcvtph2ps_mask((__v8hi)a, src, k);
}

__m256 test_vcvtph2ps256_mask(__m128i a, __m256 src, __mmask8 k) {
  // CIR-LABEL: cir.func no_inline dso_local @test_vcvtph2ps256_mask
  // CIR: %{{.*}} = cir.cast floating %{{.*}} : !cir.vector<8 x !cir.f16> -> !cir.vector<8 x !cir.float>
  // CIR: %{{.*}} = cir.vec.ternary(%{{.*}}, %{{.*}}, %{{.*}}) : !cir.vector<8 x !cir.int<s, 1>>, !cir.vector<8 x !cir.float>

  // LLVM-LABEL: @test_vcvtph2ps256_mask
  // LLVM: %{{.*}} = fpext <8 x half> %{{.*}} to <8 x float>
  // LLVM: %{{.*}} = select <8 x i1> {{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}

  // OGCG-LABEL: @test_vcvtph2ps256_mask
  // OGCG: %{{.*}} = fpext <8 x half> %{{.*}} to <8 x float>
  // OGCG: %{{.*}} = select <8 x i1> {{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  typedef short __v8hi __attribute__((__vector_size__(16)));
  return __builtin_ia32_vcvtph2ps256_mask((__v8hi)a, src, k);
}

__m512 test_vcvtph2ps512_mask(__m256i a, __m512 src, __mmask16 k) {
  // CIR-LABEL: cir.func no_inline dso_local @test_vcvtph2ps512_mask
  // CIR: %{{.*}} = cir.cast floating %{{.*}} : !cir.vector<16 x !cir.f16> -> !cir.vector<16 x !cir.float>
  // CIR: %{{.*}} = cir.vec.ternary(%{{.*}}, %{{.*}}, %{{.*}}) : !cir.vector<16 x !cir.int<s, 1>>, !cir.vector<16 x !cir.float>

  // LLVM-LABEL: @test_vcvtph2ps512_mask
  // LLVM: %{{.*}} = fpext <16 x half> %{{.*}} to <16 x float>
  // LLVM: %{{.*}} = select <16 x i1> {{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}

  // OGCG-LABEL: @test_vcvtph2ps512_mask
  // OGCG: %{{.*}} = fpext <16 x half> %{{.*}} to <16 x float>
  // OGCG: %{{.*}} = select <16 x i1> {{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  typedef short __v16hi __attribute__((__vector_size__(32)));
  return __builtin_ia32_vcvtph2ps512_mask((__v16hi)a, src, k, 4);
}

__m128 test_vcvtph2ps_maskz(__m128i a, __mmask8 k) {
  // CIR-LABEL: cir.func no_inline dso_local @test_vcvtph2ps_maskz
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !s16i>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i] : !cir.vector<4 x !s16i>
  // CIR: %{{.*}} = cir.cast floating %{{.*}} : !cir.vector<4 x !cir.f16> -> !cir.vector<4 x !cir.float>
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !cir.int<s, 1>>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i] : !cir.vector<4 x !cir.int<s, 1>>
  // CIR: %{{.*}} = cir.vec.ternary(%{{.*}}, %{{.*}}, %{{.*}}) : !cir.vector<4 x !cir.int<s, 1>>, !cir.vector<4 x !cir.float>

  // LLVM-LABEL: @test_vcvtph2ps_maskz
  // LLVM: %{{.*}} = fpext <4 x half> %{{.*}} to <4 x float>
  // LLVM: %{{.*}} = select <4 x i1> {{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}

  // OGCG-LABEL: @test_vcvtph2ps_maskz
  // OGCG: %{{.*}} = fpext <4 x half> %{{.*}} to <4 x float>
  // OGCG: %{{.*}} = select <4 x i1> {{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  typedef short __v8hi __attribute__((__vector_size__(16)));
  return __builtin_ia32_vcvtph2ps_mask((__v8hi)a, _mm_setzero_ps(), k);
}

__m256 test_vcvtph2ps256_maskz(__m128i a, __mmask8 k) {
  // CIR-LABEL: cir.func no_inline dso_local @test_vcvtph2ps256_maskz
  // CIR: %{{.*}} = cir.cast floating %{{.*}} : !cir.vector<8 x !cir.f16> -> !cir.vector<8 x !cir.float>
  // CIR: %{{.*}} = cir.vec.ternary(%{{.*}}, %{{.*}}, %{{.*}}) : !cir.vector<8 x !cir.int<s, 1>>, !cir.vector<8 x !cir.float>

  // LLVM-LABEL: @test_vcvtph2ps256_maskz
  // LLVM: %{{.*}} = fpext <8 x half> %{{.*}} to <8 x float>
  // LLVM: %{{.*}} = select <8 x i1> {{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}

  // OGCG-LABEL: @test_vcvtph2ps256_maskz
  // OGCG: %{{.*}} = fpext <8 x half> %{{.*}} to <8 x float>
  // OGCG: %{{.*}} = select <8 x i1> {{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  typedef short __v8hi __attribute__((__vector_size__(16)));
  return __builtin_ia32_vcvtph2ps256_mask((__v8hi)a, _mm256_setzero_ps(), k);
}

__m512 test_vcvtph2ps512_maskz(__m256i a, __mmask16 k) {
  // CIR-LABEL: cir.func no_inline dso_local @test_vcvtph2ps512_maskz
  // CIR: %{{.*}} = cir.cast floating %{{.*}} : !cir.vector<16 x !cir.f16> -> !cir.vector<16 x !cir.float>
  // CIR: %{{.*}} = cir.vec.ternary(%{{.*}}, %{{.*}}, %{{.*}}) : !cir.vector<16 x !cir.int<s, 1>>, !cir.vector<16 x !cir.float>

  // LLVM-LABEL: @test_vcvtph2ps512_maskz
  // LLVM: %{{.*}} = fpext <16 x half> %{{.*}} to <16 x float>
  // LLVM: %{{.*}} = select <16 x i1> {{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}

  // OGCG-LABEL: @test_vcvtph2ps512_maskz
  // OGCG: %{{.*}} = fpext <16 x half> %{{.*}} to <16 x float>
  // OGCG: %{{.*}} = select <16 x i1> {{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  typedef short __v16hi __attribute__((__vector_size__(32)));
  return __builtin_ia32_vcvtph2ps512_mask((__v16hi)a, _mm512_setzero_ps(), k, 4);
}

__m512 test_mm512_cvt_roundph_ps(__m256i a) {
  // CIR-LABEL: cir.func no_inline dso_local @test_mm512_cvt_roundph_ps
  // CIR: %{{.*}} = cir.call_llvm_intrinsic "x86.avx512.mask.vcvtph2ps.512" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<16 x !s16i>, !cir.vector<16 x !cir.float>, !u16i, !s32i) -> !cir.vector<16 x !cir.float>

  // LLVM-LABEL: @test_mm512_cvt_roundph_ps
  // LLVM: call <16 x float> @llvm.x86.avx512.mask.vcvtph2ps.512(<16 x i16> %{{.*}}, <16 x float> %{{.*}}, i16 -1, i32 8)

  // OGCG-LABEL: @test_mm512_cvt_roundph_ps
  // OGCG: call <16 x float> @llvm.x86.avx512.mask.vcvtph2ps.512(<16 x i16> %{{.*}}, <16 x float> zeroinitializer, i16 -1, i32 8)
  typedef short __v16hi __attribute__((__vector_size__(32)));
  return _mm512_cvt_roundph_ps((__v16hi)a, _MM_FROUND_NO_EXC);
}

__m512 test_mm512_mask_cvt_roundph_ps(__m512 w, __mmask16 u, __m256i a) {
  // CIR-LABEL: cir.func no_inline dso_local @test_mm512_mask_cvt_roundph_ps
  // CIR: %{{.*}} = cir.call_llvm_intrinsic "x86.avx512.mask.vcvtph2ps.512" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<16 x !s16i>, !cir.vector<16 x !cir.float>, !u16i, !s32i) -> !cir.vector<16 x !cir.float>

  // LLVM-LABEL: @test_mm512_mask_cvt_roundph_ps
  // LLVM: call <16 x float> @llvm.x86.avx512.mask.vcvtph2ps.512(<16 x i16> %{{.*}}, <16 x float> %{{.*}}, i16 %{{.*}}, i32 8)

  // OGCG-LABEL: @test_mm512_mask_cvt_roundph_ps
  // OGCG: call <16 x float> @llvm.x86.avx512.mask.vcvtph2ps.512(<16 x i16> %{{.*}}, <16 x float> %{{.*}}, i16 %{{.*}}, i32 8)
  typedef short __v16hi __attribute__((__vector_size__(32)));
  return _mm512_mask_cvt_roundph_ps(w, u, (__v16hi)a, _MM_FROUND_NO_EXC);
}

__m512 test_mm512_maskz_cvt_roundph_ps(__mmask16 u, __m256i a) {
  // CIR-LABEL: cir.func no_inline dso_local @test_mm512_maskz_cvt_roundph_ps
  // CIR: %{{.*}} = cir.call_llvm_intrinsic "x86.avx512.mask.vcvtph2ps.512" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<16 x !s16i>, !cir.vector<16 x !cir.float>, !u16i, !s32i) -> !cir.vector<16 x !cir.float>

  // LLVM-LABEL: @test_mm512_maskz_cvt_roundph_ps
  // LLVM: call <16 x float> @llvm.x86.avx512.mask.vcvtph2ps.512(<16 x i16> %{{.*}}, <16 x float> %{{.*}}, i16 %{{.*}}, i32 8)

  // OGCG-LABEL: @test_mm512_maskz_cvt_roundph_ps
  // OGCG: call <16 x float> @llvm.x86.avx512.mask.vcvtph2ps.512(<16 x i16> %{{.*}}, <16 x float> %{{.*}}, i16 %{{.*}}, i32 8)
  typedef short __v16hi __attribute__((__vector_size__(32)));
  return _mm512_maskz_cvt_roundph_ps(u, (__v16hi)a, _MM_FROUND_NO_EXC);
}

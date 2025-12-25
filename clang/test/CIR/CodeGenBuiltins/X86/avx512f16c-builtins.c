// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512fp16 -target-feature +avx512f -target-feature +avx512vl -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion 
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512fp16 -target-feature +avx512f -target-feature +avx512vl -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512fp16 -target-feature +avx512f -target-feature +avx512vl -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=OGCG --input-file=%t.ll %s

#include <immintrin.h>

__m128 test_vcvtph2ps_mask(__m128i a, __m128 src, __mmask8 k) {
  // CIR-LABEL: cir.func no_inline dso_local @test_vcvtph2ps_mask
  // CIR: %[[LOAD_A:.*]] = cir.load {{.*}} : !cir.ptr<!cir.vector<2 x !s64i>>, !cir.vector<2 x !s64i>
  // CIR: %[[VEC_I:.*]] = cir.cast bitcast %[[LOAD_A]] : !cir.vector<2 x !s64i> -> !cir.vector<8 x !s16i>
  // CIR: %[[LOAD_SRC:.*]] = cir.load {{.*}} : !cir.ptr<!cir.vector<4 x !cir.float>>, !cir.vector<4 x !cir.float>
  // CIR: %[[MASK_VAL:.*]] = cir.load {{.*}} : !cir.ptr<!u8i>, !u8i
  // CIR: %[[SHUFFLE:.*]] = cir.vec.shuffle(%[[VEC_I]], {{.*}} : !cir.vector<8 x !s16i>) {{.*}} : !cir.vector<4 x !s16i>
  // CIR: %[[BITCAST:.*]] = cir.cast bitcast %[[SHUFFLE]] : !cir.vector<4 x !s16i> -> !cir.vector<4 x !cir.f16>
  // CIR: %[[FLOAT_EXT:.*]] = cir.cast floating %[[BITCAST]] : !cir.vector<4 x !cir.f16> -> !cir.vector<4 x !cir.float>
  // CIR: %[[BOOL_VEC:.*]] = cir.cast bitcast %[[MASK_VAL]] : !u8i -> !cir.vector<8 x !cir.bool>
  // CIR: %[[FINAL_MASK:.*]] = cir.vec.shuffle(%[[BOOL_VEC]], %[[BOOL_VEC]] : !cir.vector<8 x !cir.bool>) {{.*}} : !cir.vector<4 x !cir.bool>
  // CIR: cir.select if %[[FINAL_MASK]] then %[[FLOAT_EXT]] else %[[LOAD_SRC]]

    // LLVM-LABEL: @test_vcvtph2ps_mask
    // LLVM: %[[VEC_128:.*]] = bitcast <2 x i64> {{.*}} to <8 x i16>
    // LLVM: %[[NARROWED:.*]] = shufflevector <8 x i16> %[[VEC_128]], <8 x i16> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
    // LLVM: %[[HALF_VEC:.*]] = bitcast <4 x i16> %[[NARROWED]] to <4 x half>
    // LLVM: %[[FLOAT_VEC:.*]] = fpext <4 x half> %[[HALF_VEC]] to <4 x float>
    // LLVM: %[[MASK:.*]] = shufflevector <8 x i1> {{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
    // LLVM: %[[RESULT:.*]] = select <4 x i1> %[[MASK]], <4 x float> %[[FLOAT_VEC]], <4 x float> {{.*}}
    // LLVM: ret <4 x float> {{.*}}

    // OGCG-LABEL: @test_vcvtph2ps_mask
    // OGCG: %[[VEC_128:.*]] = bitcast <2 x i64> {{.*}} to <8 x i16>
    // OGCG: %[[NARROWED:.*]] = shufflevector <8 x i16> %[[VEC_128]], <8 x i16> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
    // OGCG: %[[HALF_VEC:.*]] = bitcast <4 x i16> %[[NARROWED]] to <4 x half>
    // OGCG: %[[FLOAT_VEC:.*]] = fpext <4 x half> %[[HALF_VEC]] to <4 x float>
    // OGCG: %[[MASK:.*]] = shufflevector <8 x i1> {{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
    // OGCG: %[[RESULT:.*]] = select <4 x i1> %[[MASK]], <4 x float> %[[FLOAT_VEC]], <4 x float> {{.*}}
    // OGCG: ret <4 x float> {{.*}}
  typedef short __v8hi __attribute__((__vector_size__(16)));
  return __builtin_ia32_vcvtph2ps_mask((__v8hi)a, src, k);
}

__m256 test_vcvtph2ps256_mask(__m128i a, __m256 src, __mmask8 k) {
  // CIR-LABEL: cir.func no_inline dso_local @test_vcvtph2ps256_mask
  // CIR: %[[LOAD_A:.*]] = cir.load {{.*}} : !cir.ptr<!cir.vector<2 x !s64i>>, !cir.vector<2 x !s64i>
  // CIR: %[[VEC_I:.*]] = cir.cast bitcast %[[LOAD_A]] : !cir.vector<2 x !s64i> -> !cir.vector<8 x !s16i>
  // CIR: %[[LOAD_SRC:.*]] = cir.load {{.*}} : !cir.ptr<!cir.vector<8 x !cir.float>>, !cir.vector<8 x !cir.float>
  // CIR: %[[MASK_VAL:.*]] = cir.load {{.*}} : !cir.ptr<!u8i>, !u8i
  // CIR: %[[BITCAST:.*]] = cir.cast bitcast %[[VEC_I]] : !cir.vector<8 x !s16i> -> !cir.vector<8 x !cir.f16>
  // CIR: %[[FLOAT_EXT:.*]] = cir.cast floating %[[BITCAST]] : !cir.vector<8 x !cir.f16> -> !cir.vector<8 x !cir.float>
  // CIR: %[[BOOL_VEC:.*]] = cir.cast bitcast %[[MASK_VAL]] : !u8i -> !cir.vector<8 x !cir.bool>
  // CIR: cir.select if %[[BOOL_VEC]] then %[[FLOAT_EXT]] else %[[LOAD_SRC]]

  // LLVM-LABEL: @test_vcvtph2ps256_mask
  // LLVM: %[[BITCAST_I:.*]] = bitcast <2 x i64> {{.*}} to <8 x i16>
  // LLVM: %[[BITCAST_H:.*]] = bitcast <8 x i16> %[[BITCAST_I]] to <8 x half>
  // LLVM: %[[FPEXT:.*]] = fpext <8 x half> %[[BITCAST_H]] to <8 x float>
  // LLVM: %[[MASK:.*]] = bitcast i8 {{.*}} to <8 x i1>
  // LLVM: %[[RESULT:.*]] = select <8 x i1> %[[MASK]], <8 x float> %[[FPEXT]], <8 x float> {{.*}}
  // LLVM: ret <8 x float> {{.*}}

  // OGCG-LABEL: @test_vcvtph2ps256_mask
  // OGCG: %[[BITCAST_I:.*]] = bitcast <2 x i64> {{.*}} to <8 x i16>
  // OGCG: %[[BITCAST_H:.*]] = bitcast <8 x i16> %[[BITCAST_I]] to <8 x half>
  // OGCG: %[[FPEXT:.*]] = fpext <8 x half> %[[BITCAST_H]] to <8 x float>
  // OGCG: %[[MASK:.*]] = bitcast i8 {{.*}} to <8 x i1>
  // OGCG: %[[RESULT:.*]] = select <8 x i1> %[[MASK]], <8 x float> %[[FPEXT]], <8 x float> {{.*}}
  // OGCG: ret <8 x float> {{.*}}
  typedef short __v8hi __attribute__((__vector_size__(16)));
  return __builtin_ia32_vcvtph2ps256_mask((__v8hi)a, src, k);
}

__m512 test_vcvtph2ps512_mask(__m256i a, __m512 src, __mmask16 k) {
  // CIR-LABEL: cir.func no_inline dso_local @test_vcvtph2ps512_mask
  // CIR: %[[LOAD_A:.*]] = cir.load {{.*}} : !cir.ptr<!cir.vector<4 x !s64i>>, !cir.vector<4 x !s64i>
  // CIR: %[[VEC_I:.*]] = cir.cast bitcast %[[LOAD_A]] : !cir.vector<4 x !s64i> -> !cir.vector<16 x !s16i>
  // CIR: %[[LOAD_SRC:.*]] = cir.load {{.*}} : !cir.ptr<!cir.vector<16 x !cir.float>>, !cir.vector<16 x !cir.float>
  // CIR: %[[MASK_VAL:.*]] = cir.load {{.*}} : !cir.ptr<!u16i>, !u16i
  // CIR: %[[BITCAST:.*]] = cir.cast bitcast %[[VEC_I]] : !cir.vector<16 x !s16i> -> !cir.vector<16 x !cir.f16>
  // CIR: %[[FLOAT_EXT:.*]] = cir.cast floating %[[BITCAST]] : !cir.vector<16 x !cir.f16> -> !cir.vector<16 x !cir.float>
  // CIR: %[[BOOL_VEC:.*]] = cir.cast bitcast %[[MASK_VAL]] : !u16i -> !cir.vector<16 x !cir.bool>
  // CIR: cir.select if %[[BOOL_VEC]] then %[[FLOAT_EXT]] else %[[LOAD_SRC]]

  // LLVM-LABEL: @test_vcvtph2ps512_mask
  // LLVM: %[[BITCAST_I:.*]] = bitcast <4 x i64> {{.*}} to <16 x i16>
  // LLVM: %[[BITCAST_H:.*]] = bitcast <16 x i16> %[[BITCAST_I]] to <16 x half>
  // LLVM: %[[FPEXT:.*]] = fpext <16 x half> %[[BITCAST_H]] to <16 x float>
  // LLVM: %[[MASK:.*]] = bitcast i16 {{.*}} to <16 x i1>
  // LLVM: %[[RESULT:.*]] = select <16 x i1> %[[MASK]], <16 x float> %[[FPEXT]], <16 x float> {{.*}}
  // LLVM: ret <16 x float> {{.*}}

  // OGCG-LABEL: @test_vcvtph2ps512_mask
  // OGCG: %[[BITCAST_I:.*]] = bitcast <4 x i64> {{.*}} to <16 x i16>
  // OGCG: %[[BITCAST_H:.*]] = bitcast <16 x i16> %[[BITCAST_I]] to <16 x half>
  // OGCG: %[[FPEXT:.*]] = fpext <16 x half> %[[BITCAST_H]] to <16 x float>
  // OGCG: %[[MASK:.*]] = bitcast i16 {{.*}} to <16 x i1>
  // OGCG: %[[RESULT:.*]] = select <16 x i1> %[[MASK]], <16 x float> %[[FPEXT]], <16 x float> {{.*}}
  // OGCG: ret <16 x float> {{.*}}
  typedef short __v16hi __attribute__((__vector_size__(32)));
  return __builtin_ia32_vcvtph2ps512_mask((__v16hi)a, src, k, 4);
}

__m128 test_vcvtph2ps_maskz(__m128i a, __mmask8 k) {
  // CIR-LABEL: cir.func always_inline internal private dso_local @_mm_maskz_cvtph_ps
  // CIR: %[[LOAD_VAL:.*]] = cir.load {{.*}} : !cir.ptr<!cir.vector<2 x !s64i>>, !cir.vector<2 x !s64i>
  // CIR: %[[VEC:.*]] = cir.cast bitcast %[[LOAD_VAL]] : !cir.vector<2 x !s64i> -> !cir.vector<8 x !s16i>
  // CIR: %[[ZERO:.*]] = cir.call @_mm_setzero_ps()
  // CIR: %[[MASK_VAL:.*]] = cir.load {{.*}} : !cir.ptr<!u8i>, !u8i
  // CIR: %[[SHUFFLE:.*]] = cir.vec.shuffle(%[[VEC]], {{.*}} : !cir.vector<8 x !s16i>) {{.*}} : !cir.vector<4 x !s16i>
  // CIR: %[[F16_VEC:.*]] = cir.cast bitcast %[[SHUFFLE]] : !cir.vector<4 x !s16i> -> !cir.vector<4 x !cir.f16>
  // CIR: %[[CONV:.*]] = cir.cast floating %[[F16_VEC]] : !cir.vector<4 x !cir.f16> -> !cir.vector<4 x !cir.float>
  // CIR: %[[BOOL_VEC:.*]] = cir.cast bitcast %[[MASK_VAL]] : !u8i -> !cir.vector<8 x !cir.bool>
  // CIR: %[[FINAL_MASK:.*]] = cir.vec.shuffle(%[[BOOL_VEC]], %[[BOOL_VEC]] : !cir.vector<8 x !cir.bool>) {{.*}} : !cir.vector<4 x !cir.bool>
  // CIR: cir.select if %[[FINAL_MASK]] then %[[CONV]] else %[[ZERO]]

  // CIR-LABEL: cir.func no_inline dso_local @test_vcvtph2ps_maskz
  // CIR: cir.call @_mm_maskz_cvtph_ps({{.*}}, {{.*}})

  // LLVM-LABEL: @test_vcvtph2ps_maskz
  // LLVM: %[[BITCAST_I:.*]] = bitcast <2 x i64> {{.*}} to <8 x i16>
  // LLVM: %[[NARROW:.*]] = shufflevector <8 x i16> %[[BITCAST_I]], <8 x i16> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // LLVM: %[[BITCAST_H:.*]] = bitcast <4 x i16> %[[NARROW]] to <4 x half>
  // LLVM: %[[CONV:.*]] = fpext <4 x half> %[[BITCAST_H]] to <4 x float>
  // LLVM: %[[MASK:.*]] = shufflevector <8 x i1> {{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // LLVM: %[[RESULT:.*]] = select <4 x i1> %[[MASK]], <4 x float> %[[CONV]], <4 x float> {{.*}}
  // LLVM: ret <4 x float> {{.*}}

  // OGCG-LABEL: @test_vcvtph2ps_maskz
  // OGCG: %[[BITCAST_I:.*]] = bitcast <2 x i64> {{.*}} to <8 x i16>
  // OGCG: %[[NARROW:.*]] = shufflevector <8 x i16> %[[BITCAST_I]], <8 x i16> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // OGCG: %[[BITCAST_H:.*]] = bitcast <4 x i16> %[[NARROW]] to <4 x half>
  // OGCG: %[[CONV:.*]] = fpext <4 x half> %[[BITCAST_H]] to <4 x float>
  // OGCG: %[[MASK:.*]] = shufflevector <8 x i1> {{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // OGCG: %[[RESULT:.*]] = select <4 x i1> %[[MASK]], <4 x float> %[[CONV]], <4 x float> {{.*}}
  // OGCG: ret <4 x float> {{.*}}

  return _mm_maskz_cvtph_ps(k, a);
}

__m256 test_vcvtph2ps256_maskz(__m128i a, __mmask8 k) {
  // CIR-LABEL: cir.func always_inline internal private dso_local @_mm256_maskz_cvtph_ps
  // CIR: %[[LOAD_VAL:.*]] = cir.load {{.*}} : !cir.ptr<!cir.vector<2 x !s64i>>, !cir.vector<2 x !s64i>
  // CIR: %[[VEC_I:.*]] = cir.cast bitcast %[[LOAD_VAL]] : !cir.vector<2 x !s64i> -> !cir.vector<8 x !s16i>
  // CIR: %[[ZERO:.*]] = cir.call @_mm256_setzero_ps()
  // CIR: %[[MASK_VAL:.*]] = cir.load {{.*}} : !cir.ptr<!u8i>, !u8i
  // CIR: %[[CONV_H:.*]] = cir.cast bitcast %[[VEC_I]] : !cir.vector<8 x !s16i> -> !cir.vector<8 x !cir.f16>

  // CIR-LABEL: cir.func no_inline dso_local @test_vcvtph2ps256_maskz
  // CIR: cir.call @_mm256_maskz_cvtph_ps({{.*}}, {{.*}}) 


  // LLVM-LABEL: @test_vcvtph2ps256_maskz
  // LLVM: %[[BITCAST_I:.*]] = bitcast <2 x i64> {{.*}} to <8 x i16>
  // LLVM: %[[BITCAST_H:.*]] = bitcast <8 x i16> %[[BITCAST_I]] to <8 x half>
  // LLVM: %[[CONV:.*]] = fpext <8 x half> %[[BITCAST_H]] to <8 x float>
  // LLVM: %[[MASK:.*]] = bitcast i8 {{.*}} to <8 x i1>
  // LLVM: %[[RESULT:.*]] = select <8 x i1> %[[MASK]], <8 x float> %[[CONV]], <8 x float> {{.*}}
  // LLVM: ret <8 x float> {{.*}} 

  // OGCG-LABEL: @test_vcvtph2ps256_maskz
  // OGCG: %[[BITCAST_I:.*]] = bitcast <2 x i64> {{.*}} to <8 x i16>
  // OGCG: %[[BITCAST_H:.*]] = bitcast <8 x i16> %[[BITCAST_I]] to <8 x half>
  // OGCG: %[[CONV:.*]] = fpext <8 x half> %[[BITCAST_H]] to <8 x float>
  // OGCG: %[[MASK:.*]] = bitcast i8 {{.*}} to <8 x i1>
  // OGCG: %[[RESULT:.*]] = select <8 x i1> %[[MASK]], <8 x float> %[[CONV]], <8 x float> {{.*}}
  // OGCG: ret <8 x float> {{.*}}
   return _mm256_maskz_cvtph_ps(k, a);
}

__m512 test_vcvtph2ps512_maskz(__m256i a, __mmask16 k) {
  // CIR-LABEL: cir.func always_inline internal private dso_local @_mm512_maskz_cvtph_ps
  // CIR: %[[LOAD_VAL:.*]] = cir.load {{.*}} : !cir.ptr<!cir.vector<4 x !s64i>>, !cir.vector<4 x !s64i>
  // CIR: %[[VEC_I:.*]] = cir.cast bitcast %[[LOAD_VAL]] : !cir.vector<4 x !s64i> -> !cir.vector<16 x !s16i>
  // CIR: %[[ZERO:.*]] = cir.call @_mm512_setzero_ps()
  // CIR: %[[MASK_VAL:.*]] = cir.load {{.*}} : !cir.ptr<!u16i>, !u16i
  // CIR: %[[CONV_H:.*]] = cir.cast bitcast %[[VEC_I]] : !cir.vector<16 x !s16i> -> !cir.vector<16 x !cir.f16>

  // CIR-LABEL: cir.func no_inline dso_local @test_vcvtph2ps512_maskz
  // CIR: cir.call @_mm512_maskz_cvtph_ps({{.*}}, {{.*}})

  // LLVM-LABEL: @test_vcvtph2ps512_maskz
  // LLVM: %[[BI:.*]] = bitcast <4 x i64> {{.*}} to <16 x i16>
  // LLVM: %[[BH:.*]] = bitcast <16 x i16> %[[BI]] to <16 x half>
  // LLVM: %[[CONV:.*]] = fpext <16 x half> %[[BH]] to <16 x float>
  // LLVM: %[[MASK:.*]] = bitcast i16 {{.*}} to <16 x i1>
  // LLVM: %[[RES:.*]] = select <16 x i1> %[[MASK]], <16 x float> %[[CONV]], <16 x float> {{.*}}
  // LLVM: ret <16 x float> {{.*}}
  
  // OGCG-LABEL: @test_vcvtph2ps512_maskz
  // OGCG: %[[BI:.*]] = bitcast <4 x i64> {{.*}} to <16 x i16>
  // OGCG: %[[BH:.*]] = bitcast <16 x i16> %[[BI]] to <16 x half>
  // OGCG: %[[CONV:.*]] = fpext <16 x half> %[[BH]] to <16 x float>
  // OGCG: %[[MASK:.*]] = bitcast i16 {{.*}} to <16 x i1>
  // OGCG: %[[RES:.*]] = select <16 x i1> %[[MASK]], <16 x float> %[[CONV]], <16 x float> {{.*}}
  // OGCG: ret <16 x float> {{.*}}
  return _mm512_maskz_cvtph_ps(k, a);
}

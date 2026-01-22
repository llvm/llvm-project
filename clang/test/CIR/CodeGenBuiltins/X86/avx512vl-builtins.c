// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -target-feature +avx512vl -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion 
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -target-feature +avx512vl -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -target-feature +avx512vl -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=OGCG --input-file=%t.ll %s

#include <immintrin.h>

__m128d test_mm_mmask_i64gather_pd(__m128d __v1_old, __mmask8 __mask, __m128i __index, void const *__addr) {
  // CIR-LABEL: test_mm_mmask_i64gather_pd
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.gather3div2.df"

  // LLVM-LABEL: @test_mm_mmask_i64gather_pd
  // LLVM: @llvm.x86.avx512.mask.gather3div2.df

  // OGCG-LABEL: @test_mm_mmask_i64gather_pd
  // OGCG: @llvm.x86.avx512.mask.gather3div2.df
  return _mm_mmask_i64gather_pd(__v1_old, __mask, __index, __addr, 2); 
}

__m128i test_mm_mmask_i64gather_epi64(__m128i __v1_old, __mmask8 __mask, __m128i __index, void const *__addr) {
  // CIR-LABEL: test_mm_mmask_i64gather_epi64
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.gather3div2.di"

  // LLVM-LABEL: @test_mm_mmask_i64gather_epi64
  // LLVM: @llvm.x86.avx512.mask.gather3div2.di

  // OGCG-LABEL: @test_mm_mmask_i64gather_epi64
  // OGCG: @llvm.x86.avx512.mask.gather3div2.di
  return _mm_mmask_i64gather_epi64(__v1_old, __mask, __index, __addr, 2); 
}

__m256d test_mm256_mmask_i64gather_pd(__m256d __v1_old, __mmask8 __mask, __m256i __index, void const *__addr) {
  // CIR-LABEL: test_mm256_mmask_i64gather_pd
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.gather3div4.df"

  // LLVM-LABEL: @test_mm256_mmask_i64gather_pd
  // LLVM: @llvm.x86.avx512.mask.gather3div4.df

  // OGCG-LABEL: @test_mm256_mmask_i64gather_pd
  // OGCG: @llvm.x86.avx512.mask.gather3div4.df
  return _mm256_mmask_i64gather_pd(__v1_old, __mask, __index, __addr, 2); 
}

__m256i test_mm256_mmask_i64gather_epi64(__m256i __v1_old, __mmask8 __mask, __m256i __index, void const *__addr) {
  // CIR-LABEL: test_mm256_mmask_i64gather_epi64
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.gather3div4.di"

  // LLVM-LABEL: @test_mm256_mmask_i64gather_epi64
  // LLVM: @llvm.x86.avx512.mask.gather3div4.di

  // OGCG-LABEL: @test_mm256_mmask_i64gather_epi64
  // OGCG: @llvm.x86.avx512.mask.gather3div4.di
  return _mm256_mmask_i64gather_epi64(__v1_old, __mask, __index, __addr, 2); 
}

__m128 test_mm_mmask_i64gather_ps(__m128 __v1_old, __mmask8 __mask, __m128i __index, void const *__addr) {
  // CIR-LABEL: test_mm_mmask_i64gather_ps
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.gather3div4.sf"

  // LLVM-LABEL: @test_mm_mmask_i64gather_ps
  // LLVM: @llvm.x86.avx512.mask.gather3div4.sf

  // OGCG-LABEL: @test_mm_mmask_i64gather_ps
  // OGCG: @llvm.x86.avx512.mask.gather3div4.sf
  return _mm_mmask_i64gather_ps(__v1_old, __mask, __index, __addr, 2); 
}

__m128i test_mm_mmask_i64gather_epi32(__m128i __v1_old, __mmask8 __mask, __m128i __index, void const *__addr) {
  // CIR-LABEL: test_mm_mmask_i64gather_epi32
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.gather3div4.si"

  // LLVM-LABEL: @test_mm_mmask_i64gather_epi32
  // LLVM: @llvm.x86.avx512.mask.gather3div4.si

  // OGCG-LABEL: @test_mm_mmask_i64gather_epi32
  // OGCG: @llvm.x86.avx512.mask.gather3div4.si
  return _mm_mmask_i64gather_epi32(__v1_old, __mask, __index, __addr, 2); 
}

__m128 test_mm256_mmask_i64gather_ps(__m128 __v1_old, __mmask8 __mask, __m256i __index, void const *__addr) {
  // CIR-LABEL: test_mm256_mmask_i64gather_ps
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.gather3div8.sf"

  // LLVM-LABEL: @test_mm256_mmask_i64gather_ps
  // LLVM: @llvm.x86.avx512.mask.gather3div8.sf

  // OGCG-LABEL: @test_mm256_mmask_i64gather_ps
  // OGCG: @llvm.x86.avx512.mask.gather3div8.sf
  return _mm256_mmask_i64gather_ps(__v1_old, __mask, __index, __addr, 2); 
}

__m128i test_mm256_mmask_i64gather_epi32(__m128i __v1_old, __mmask8 __mask, __m256i __index, void const *__addr) {
  // CIR-LABEL: test_mm256_mmask_i64gather_epi32
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.gather3div8.si"

  // LLVM-LABEL: @test_mm256_mmask_i64gather_epi32
  // LLVM: @llvm.x86.avx512.mask.gather3div8.si

  // OGCG-LABEL: @test_mm256_mmask_i64gather_epi32
  // OGCG: @llvm.x86.avx512.mask.gather3div8.si
  return _mm256_mmask_i64gather_epi32(__v1_old, __mask, __index, __addr, 2); 
}

__m128d test_mm_mask_i32gather_pd(__m128d __v1_old, __mmask8 __mask, __m128i __index, void const *__addr) {
  // CIR-LABEL: test_mm_mask_i32gather_pd
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.gather3siv2.df"

  // LLVM-LABEL: @test_mm_mask_i32gather_pd
  // LLVM: @llvm.x86.avx512.mask.gather3siv2.df

  // OGCG-LABEL: @test_mm_mask_i32gather_pd
  // OGCG: @llvm.x86.avx512.mask.gather3siv2.df
  return _mm_mmask_i32gather_pd(__v1_old, __mask, __index, __addr, 2); 
}

__m128i test_mm_mask_i32gather_epi64(__m128i __v1_old, __mmask8 __mask, __m128i __index, void const *__addr) {
  // CIR-LABEL: test_mm_mask_i32gather_epi64
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.gather3siv2.di"

  // LLVM-LABEL: @test_mm_mask_i32gather_epi64
  // LLVM: @llvm.x86.avx512.mask.gather3siv2.di

  // OGCG-LABEL: @test_mm_mask_i32gather_epi64
  // OGCG: @llvm.x86.avx512.mask.gather3siv2.di
  return _mm_mmask_i32gather_epi64(__v1_old, __mask, __index, __addr, 2); 
}

__m256d test_mm256_mask_i32gather_pd(__m256d __v1_old, __mmask8 __mask, __m128i __index, void const *__addr) {
  // CIR-LABEL: test_mm256_mask_i32gather_pd
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.gather3siv4.df"

  // LLVM-LABEL: @test_mm256_mask_i32gather_pd
  // LLVM: @llvm.x86.avx512.mask.gather3siv4.df

  // OGCG-LABEL: @test_mm256_mask_i32gather_pd
  // OGCG: @llvm.x86.avx512.mask.gather3siv4.df
  return _mm256_mmask_i32gather_pd(__v1_old, __mask, __index, __addr, 2); 
}

__m256i test_mm256_mask_i32gather_epi64(__m256i __v1_old, __mmask8 __mask, __m128i __index, void const *__addr) {
  // CIR-LABEL: test_mm256_mask_i32gather_epi64
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.gather3siv4.di"

  // LLVM-LABEL: @test_mm256_mask_i32gather_epi64
  // LLVM: @llvm.x86.avx512.mask.gather3siv4.di

  // OGCG-LABEL: @test_mm256_mask_i32gather_epi64
  // OGCG: @llvm.x86.avx512.mask.gather3siv4.di
  return _mm256_mmask_i32gather_epi64(__v1_old, __mask, __index, __addr, 2); 
}

__m128 test_mm_mask_i32gather_ps(__m128 __v1_old, __mmask8 __mask, __m128i __index, void const *__addr) {
  // CIR-LABEL: test_mm_mask_i32gather_ps
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.gather3siv4.sf"

  // LLVM-LABEL: @test_mm_mask_i32gather_ps
  // LLVM: @llvm.x86.avx512.mask.gather3siv4.sf

  // OGCG-LABEL: @test_mm_mask_i32gather_ps
  // OGCG: @llvm.x86.avx512.mask.gather3siv4.sf
  return _mm_mmask_i32gather_ps(__v1_old, __mask, __index, __addr, 2); 
}

__m128i test_mm_mask_i32gather_epi32(__m128i __v1_old, __mmask8 __mask, __m128i __index, void const *__addr) {
  // CIR-LABEL: test_mm_mask_i32gather_epi32
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.gather3siv4.si"

  // LLVM-LABEL: @test_mm_mask_i32gather_epi32
  // LLVM: @llvm.x86.avx512.mask.gather3siv4.si

  // OGCG-LABEL: @test_mm_mask_i32gather_epi32
  // OGCG: @llvm.x86.avx512.mask.gather3siv4.si
  return _mm_mmask_i32gather_epi32(__v1_old, __mask, __index, __addr, 2); 
}

__m256 test_mm256_mask_i32gather_ps(__m256 __v1_old, __mmask8 __mask, __m256i __index, void const *__addr) {
  // CIR-LABEL: test_mm256_mask_i32gather_ps
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.gather3siv8.sf"

  // LLVM-LABEL: @test_mm256_mask_i32gather_ps
  // LLVM: @llvm.x86.avx512.mask.gather3siv8.sf

  // OGCG-LABEL: @test_mm256_mask_i32gather_ps
  // OGCG: @llvm.x86.avx512.mask.gather3siv8.sf
  return _mm256_mmask_i32gather_ps(__v1_old, __mask, __index, __addr, 2); 
}

__m256i test_mm256_mask_i32gather_epi32(__m256i __v1_old, __mmask8 __mask, __m256i __index, void const *__addr) {
  // CIR-LABEL: test_mm256_mask_i32gather_epi32
  // CIR: cir.call_llvm_intrinsic "x86.avx512.mask.gather3siv8.si"

  // LLVM-LABEL: @test_mm256_mask_i32gather_epi32
  // LLVM: @llvm.x86.avx512.mask.gather3siv8.si

  // OGCG-LABEL: @test_mm256_mask_i32gather_epi32
  // OGCG: @llvm.x86.avx512.mask.gather3siv8.si
  return _mm256_mmask_i32gather_epi32(__v1_old, __mask, __index, __addr, 2); 
}

__m128i test_mm_ror_epi32(__m128i __A) {
  // CIR-LABEL: test_mm_ror_epi32
  // CIR: cir.vec.splat %{{.*}} : !u32i, !cir.vector<4 x !u32i>
  // CIR: cir.call_llvm_intrinsic "fshr" %{{.*}}: (!cir.vector<4 x !s32i>, !cir.vector<4 x !s32i>, !cir.vector<4 x !u32i>) -> !cir.vector<4 x !s32i>

  // LLVM-LABEL: @test_mm_ror_epi32
  // LLVM: %[[CASTED_VAR:.*]] = bitcast <2 x i64> %{{.*}} to <4 x i32>
  // LLVM: call <4 x i32> @llvm.fshr.v4i32(<4 x i32> %[[CASTED_VAR]], <4 x i32> %[[CASTED_VAR]], <4 x i32> splat (i32 5))

  // OGCG-LABEL: @test_mm_ror_epi32
  // OGCG: %[[CASTED_VAR:.*]] = bitcast <2 x i64> %{{.*}} to <4 x i32>
  // OGCG: call <4 x i32> @llvm.fshr.v4i32(<4 x i32> %[[CASTED_VAR]], <4 x i32> %[[CASTED_VAR]], <4 x i32> splat (i32 5))
  return _mm_ror_epi32(__A, 5);
}

__m256i test_mm256_ror_epi32(__m256i __A) {
  // CIR-LABEL: test_mm256_ror_epi32
  // CIR: cir.vec.splat %{{.*}} : !u32i, !cir.vector<8 x !u32i>
  // CIR: cir.call_llvm_intrinsic "fshr" %{{.*}}: (!cir.vector<8 x !s32i>, !cir.vector<8 x !s32i>, !cir.vector<8 x !u32i>) -> !cir.vector<8 x !s32i>

  // LLVM-LABEL: @test_mm256_ror_epi32
  // LLVM: %[[CASTED_VAR:.*]] = bitcast <4 x i64> %{{.*}} to <8 x i32>
  // LLVM: call <8 x i32> @llvm.fshr.v8i32(<8 x i32> %[[CASTED_VAR]], <8 x i32> %[[CASTED_VAR]], <8 x i32> splat (i32 5))

  // OGCG-LABEL: @test_mm256_ror_epi32
  // OGCG: %[[CASTED_VAR:.*]] = bitcast <4 x i64> %{{.*}} to <8 x i32>
  // OGCG: call <8 x i32> @llvm.fshr.v8i32(<8 x i32> %[[CASTED_VAR]], <8 x i32> %[[CASTED_VAR]], <8 x i32> splat (i32 5))
  return _mm256_ror_epi32(__A, 5);
}

__m128i test_mm_ror_epi64(__m128i __A) {
  // CIR-LABEL: test_mm_ror_epi64
  // CIR: cir.vec.splat %{{.*}} : !u64i, !cir.vector<2 x !u64i>
  // CIR: cir.call_llvm_intrinsic "fshr" %{{.*}}: (!cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.vector<2 x !u64i>) -> !cir.vector<2 x !s64i>

  // LLVM-LABEL: @test_mm_ror_epi64
  // LLVM: call <2 x i64> @llvm.fshr.v2i64(<2 x i64> %[[V:.*]], <2 x i64> %[[V]], <2 x i64> splat (i64 5))

  // OGCG-LABEL: @test_mm_ror_epi64
  // OGCG: call <2 x i64> @llvm.fshr.v2i64(<2 x i64> %[[V:.*]], <2 x i64> %[[V]], <2 x i64> splat (i64 5))
  return _mm_ror_epi64(__A, 5);
}

__m256i test_mm256_ror_epi64(__m256i __A) {
  // CIR-LABEL: test_mm256_ror_epi64
  // CIR: cir.vec.splat %{{.*}} : !u64i, !cir.vector<4 x !u64i>
  // CIR: cir.call_llvm_intrinsic "fshr" %{{.*}}: (!cir.vector<4 x !s64i>, !cir.vector<4 x !s64i>, !cir.vector<4 x !u64i>) -> !cir.vector<4 x !s64i>

  // LLVM-LABEL: @test_mm256_ror_epi64
  // LLVM: call <4 x i64> @llvm.fshr.v4i64(<4 x i64> %[[V:.*]], <4 x i64> %[[V]], <4 x i64> splat (i64 5))

  // OGCG-LABEL: @test_mm256_ror_epi64
  // OGCG: call <4 x i64> @llvm.fshr.v4i64(<4 x i64> %[[V:.*]], <4 x i64> %[[V]], <4 x i64> splat (i64 5))
  return _mm256_ror_epi64(__A, 5);
}

__m256 test_mm256_insertf32x4(__m256 __A, __m128 __B) {
  // CIR-LABEL: test_mm256_insertf32x4
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !cir.float>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<8> : !s32i, #cir.int<9> : !s32i, #cir.int<10> : !s32i, #cir.int<11> : !s32i] : !cir.vector<8 x !cir.float>

  // LLVM-LABEL: @test_mm256_insertf32x4
  // LLVM: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>

  // OGCG-LABEL: @test_mm256_insertf32x4
  // OGCG: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>
  return _mm256_insertf32x4(__A, __B, 1);
}

__m256i test_mm256_inserti32x4(__m256i __A, __m128i __B) {
  // CIR-LABEL: test_mm256_inserti32x4
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !s32i>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<8> : !s32i, #cir.int<9> : !s32i, #cir.int<10> : !s32i, #cir.int<11> : !s32i] : !cir.vector<8 x !s32i>

  // LLVM-LABEL: @test_mm256_inserti32x4
  // LLVM: shufflevector <8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>

  // OGCG-LABEL: @test_mm256_inserti32x4
  // OGCG: shufflevector <8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>
  return _mm256_inserti32x4(__A, __B, 1);
}

__m128d test_mm_mask_expand_pd(__m128d __W, __mmask8 __U, __m128d __A) {
  // CIR-LABEL: _mm_mask_expand_pd
  // CIR: %[[MASK:.*]] = cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.int<s, 1>>
  // CIR: %[[SHUF:.*]] = cir.vec.shuffle(%[[MASK]], %[[MASK]] : !cir.vector<8 x !cir.int<s, 1>>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i] : !cir.vector<2 x !cir.int<s, 1>>

  // LLVM-LABEL: test_mm_mask_expand_pd
  // LLVM: %[[BC:.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // LLVM: %[[SHUF:.*]] = shufflevector <8 x i1> %[[BC]], <8 x i1> %[[BC]], <2 x i32> <i32 0, i32 1>

  // OGCG-LABEL: test_mm_mask_expand_pd
  // OGCG: %[[BC:.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // OGCG: %[[SHUF:.*]] = shufflevector <8 x i1> %[[BC]], <8 x i1> %[[BC]], <2 x i32> <i32 0, i32 1>

  return _mm_mask_expand_pd(__W,__U,__A);
}

__m128d test_mm_maskz_expand_pd(__mmask8 __U, __m128d __A) {
  // CIR-LABEL: _mm_maskz_expand_pd
  // CIR: %[[MASK:.*]] = cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.int<s, 1>>
  // CIR: %[[SHUF:.*]] = cir.vec.shuffle(%[[MASK]], %[[MASK]] : !cir.vector<8 x !cir.int<s, 1>>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i] : !cir.vector<2 x !cir.int<s, 1>>

  // LLVM-LABEL: test_mm_maskz_expand_pd
  // LLVM: %[[BC:.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // LLVM: %[[SHUF:.*]] = shufflevector <8 x i1> %[[BC]], <8 x i1> %[[BC]], <2 x i32> <i32 0, i32 1>

  // OGCG-LABEL: test_mm_maskz_expand_pd
  // OGCG: %[[BC:.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // OGCG: %[[SHUF:.*]] = shufflevector <8 x i1> %[[BC]], <8 x i1> %[[BC]], <2 x i32> <i32 0, i32 1>

  return _mm_maskz_expand_pd(__U,__A);
}


__m256 test_mm256_shuffle_f32x4(__m256 a, __m256 b) {
  // CIR-LABEL: test_mm256_shuffle_f32x4
  // CIR:   cir.vec.shuffle(%{{.+}}, %{{.+}} : !cir.vector<8 x !cir.float>)
  // CIR-SAME: [#cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i, #cir.int<12> : !s32i, #cir.int<13> : !s32i, #cir.int<14> : !s32i, #cir.int<15> : !s32i]

  // LLVM-LABEL: test_mm256_shuffle_f32x4
  // LLVM: shufflevector <8 x float> %{{.+}}, <8 x float> %{{.+}}, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 12, i32 13, i32 14, i32 15>

  // OGCG-LABEL: test_mm256_shuffle_f32x4
  // OGCG: shufflevector <8 x float> %{{.+}}, <8 x float> %{{.+}}, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 12, i32 13, i32 14, i32 15>
  return _mm256_shuffle_f32x4(a, b, 0x03); // 1, 1
}

__m256d test_mm256_shuffle_f64x2(__m256d a, __m256d b) {
  // CIR-LABEL: test_mm256_shuffle_f64x2
  // CIR:   cir.vec.shuffle(%{{.+}}, %{{.+}} : !cir.vector<4 x !cir.double>)
  // CIR-SAME: [#cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i]

  // LLVM-LABEL: test_mm256_shuffle_f64x2
  // LLVM: shufflevector <4 x double> %{{.+}}, <4 x double> %{{.+}}, <4 x i32> <i32 2, i32 3, i32 6, i32 7>

  // OGCG-LABEL: test_mm256_shuffle_f64x2
  // OGCG: shufflevector <4 x double> %{{.+}}, <4 x double> %{{.+}}, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  return _mm256_shuffle_f64x2(a, b, 0x03);
}

__m256i test_mm256_shuffle_i32x4(__m256i a, __m256i b) {
  // CIR-LABEL: test_mm256_shuffle_i32x4
  // CIR:   cir.vec.shuffle(%{{.+}}, %{{.+}} : !cir.vector<8 x !s32i>)
  // CIR-SAME: [#cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i, #cir.int<12> : !s32i, #cir.int<13> : !s32i, #cir.int<14> : !s32i, #cir.int<15> : !s32i]

  // LLVM-LABEL: test_mm256_shuffle_i32x4
  // LLVM: shufflevector <8 x i32> %{{.+}}, <8 x i32> %{{.+}}, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 12, i32 13, i32 14, i32 15>

  // OGCG-LABEL: test_mm256_shuffle_i32x4
  // OGCG: shufflevector <8 x i32> %{{.+}}, <8 x i32> %{{.+}}, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 12, i32 13, i32 14, i32 15>
  return _mm256_shuffle_i32x4(a, b, 0x03);
}

__m256i test_mm256_shuffle_i64x2(__m256i a, __m256i b) {
  // CIR-LABEL: test_mm256_shuffle_i64x2
  // CIR:   cir.vec.shuffle(%{{.+}}, %{{.+}} : !cir.vector<4 x !s64i>)
  // CIR-SAME: [#cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i]

  // LLVM-LABEL: test_mm256_shuffle_i64x2
  // LLVM: shufflevector <4 x i64> %{{.+}}, <4 x i64> %{{.+}}, <4 x i32> <i32 2, i32 3, i32 6, i32 7>

  // OGCG-LABEL: test_mm256_shuffle_i64x2
  // OGCG: shufflevector <4 x i64> %{{.+}}, <4 x i64> %{{.+}}, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  return _mm256_shuffle_i64x2(a, b, 0x03);
}

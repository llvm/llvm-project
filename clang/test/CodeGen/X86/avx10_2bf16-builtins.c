// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64 -target-feature +avx10.2-256 -emit-llvm -o - -Wno-invalid-feature-combination -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=i386 -target-feature +avx10.2-256 -emit-llvm -o - -Wno-invalid-feature-combination -Wall -Werror | FileCheck %s

#include <immintrin.h>

__m256bh test_mm256_setzero_pbh() {
  // CHECK-LABEL: @test_mm256_setzero_pbh
  // CHECK: zeroinitializer
  return _mm256_setzero_pbh();
}

__m128bh test_mm_setzero_pbh() {
  // CHECK-LABEL: @test_mm_setzero_pbh
  // CHECK: zeroinitializer
  return _mm_setzero_pbh();
}

__m256bh test_mm256_undefined_pbh(void) {
  // CHECK-LABEL: @test_mm256_undefined_pbh
  // CHECK: ret <16 x bfloat> zeroinitializer
  return _mm256_undefined_pbh();
}

__m128bh test_mm_undefined_pbh(void) {
  // CHECK-LABEL: @test_mm_undefined_pbh
  // CHECK: ret <8 x bfloat> zeroinitializer
  return _mm_undefined_pbh();
}

__bf16 test_mm_cvtsbh_bf16(__m128bh __A) {
  // CHECK-LABEL: @test_mm_cvtsbh_bf16
  // CHECK: extractelement <8 x bfloat> %{{.*}}, i32 0
  return _mm_cvtsbh_bf16(__A);
}

__bf16 test_mm256_cvtsbh_bf16(__m256bh __A) {
  // CHECK-LABEL: @test_mm256_cvtsbh_bf16
  // CHECK: extractelement <16 x bfloat> %{{.*}}, i32 0
  return _mm256_cvtsbh_bf16(__A);
}

__m128bh test_mm_set_sbh(__bf16 h) {
  // CHECK-LABEL: @test_mm_set_sbh
  // CHECK: insertelement <8 x bfloat> {{.*}}, i32 0
  // CHECK: insertelement <8 x bfloat> %{{.*}}, bfloat %{{.*}}, i32 1
  // CHECK: insertelement <8 x bfloat> %{{.*}}, bfloat %{{.*}}, i32 2
  // CHECK: insertelement <8 x bfloat> %{{.*}}, bfloat %{{.*}}, i32 3
  // CHECK: insertelement <8 x bfloat> %{{.*}}, bfloat %{{.*}}, i32 4
  // CHECK: insertelement <8 x bfloat> %{{.*}}, bfloat %{{.*}}, i32 5
  // CHECK: insertelement <8 x bfloat> %{{.*}}, bfloat %{{.*}}, i32 6
  // CHECK: insertelement <8 x bfloat> %{{.*}}, bfloat %{{.*}}, i32 7
  return _mm_set_sbh(h);
}

__m128bh test_mm_set1_pbh(__bf16 h) {
  // CHECK-LABEL: @test_mm_set1_pbh
  // CHECK: insertelement <8 x bfloat> {{.*}}, i32 0
  // CHECK: insertelement <8 x bfloat> {{.*}}, i32 1
  // CHECK: insertelement <8 x bfloat> {{.*}}, i32 2
  // CHECK: insertelement <8 x bfloat> {{.*}}, i32 3
  // CHECK: insertelement <8 x bfloat> {{.*}}, i32 4
  // CHECK: insertelement <8 x bfloat> {{.*}}, i32 5
  // CHECK: insertelement <8 x bfloat> {{.*}}, i32 6
  // CHECK: insertelement <8 x bfloat> {{.*}}, i32 7
  return _mm_set1_pbh(h);
}

__m256bh test_mm256_set1_pbh(__bf16 h) {
  // CHECK-LABEL: @test_mm256_set1_pbh
  // CHECK: insertelement <16 x bfloat> {{.*}}, i32 0
  // CHECK: insertelement <16 x bfloat> {{.*}}, i32 1
  // CHECK: insertelement <16 x bfloat> {{.*}}, i32 2
  // CHECK: insertelement <16 x bfloat> {{.*}}, i32 3
  // CHECK: insertelement <16 x bfloat> {{.*}}, i32 4
  // CHECK: insertelement <16 x bfloat> {{.*}}, i32 5
  // CHECK: insertelement <16 x bfloat> {{.*}}, i32 6
  // CHECK: insertelement <16 x bfloat> {{.*}}, i32 7
  // CHECK: insertelement <16 x bfloat> {{.*}}, i32 8
  // CHECK: insertelement <16 x bfloat> {{.*}}, i32 9
  // CHECK: insertelement <16 x bfloat> {{.*}}, i32 10
  // CHECK: insertelement <16 x bfloat> {{.*}}, i32 11
  // CHECK: insertelement <16 x bfloat> {{.*}}, i32 12
  // CHECK: insertelement <16 x bfloat> {{.*}}, i32 13
  // CHECK: insertelement <16 x bfloat> {{.*}}, i32 14
  // CHECK: insertelement <16 x bfloat> {{.*}}, i32 15
  return _mm256_set1_pbh(h);
}

__m128bh test_mm_set_pbh(__bf16 bf1, __bf16 bf2, __bf16 bf3, __bf16 bf4,
                       __bf16 bf5, __bf16 bf6, __bf16 bf7, __bf16 bf8) {
  // CHECK-LABEL: @test_mm_set_pbh
  // CHECK: insertelement <8 x bfloat> {{.*}}, i32 0
  // CHECK: insertelement <8 x bfloat> {{.*}}, i32 1
  // CHECK: insertelement <8 x bfloat> {{.*}}, i32 2
  // CHECK: insertelement <8 x bfloat> {{.*}}, i32 3
  // CHECK: insertelement <8 x bfloat> {{.*}}, i32 4
  // CHECK: insertelement <8 x bfloat> {{.*}}, i32 5
  // CHECK: insertelement <8 x bfloat> {{.*}}, i32 6
  // CHECK: insertelement <8 x bfloat> {{.*}}, i32 7
  return _mm_set_pbh(bf1, bf2, bf3, bf4, bf5, bf6, bf7, bf8);
}

__m256bh test_mm256_set_pbh(__bf16 bf1, __bf16 bf2, __bf16 bf3, __bf16 bf4,
                          __bf16 bf5, __bf16 bf6, __bf16 bf7, __bf16 bf8,
                          __bf16 bf9, __bf16 bf10, __bf16 bf11, __bf16 bf12,
                          __bf16 bf13, __bf16 bf14, __bf16 bf15, __bf16 bf16) {
  // CHECK-LABEL: @test_mm256_set_pbh
  // CHECK: insertelement <16 x bfloat> {{.*}}, i32 0
  // CHECK: insertelement <16 x bfloat> {{.*}}, i32 1
  // CHECK: insertelement <16 x bfloat> {{.*}}, i32 2
  // CHECK: insertelement <16 x bfloat> {{.*}}, i32 3
  // CHECK: insertelement <16 x bfloat> {{.*}}, i32 4
  // CHECK: insertelement <16 x bfloat> {{.*}}, i32 5
  // CHECK: insertelement <16 x bfloat> {{.*}}, i32 6
  // CHECK: insertelement <16 x bfloat> {{.*}}, i32 7
  // CHECK: insertelement <16 x bfloat> {{.*}}, i32 8
  // CHECK: insertelement <16 x bfloat> {{.*}}, i32 9
  // CHECK: insertelement <16 x bfloat> {{.*}}, i32 10
  // CHECK: insertelement <16 x bfloat> {{.*}}, i32 11
  // CHECK: insertelement <16 x bfloat> {{.*}}, i32 12
  // CHECK: insertelement <16 x bfloat> {{.*}}, i32 13
  // CHECK: insertelement <16 x bfloat> {{.*}}, i32 14
  // CHECK: insertelement <16 x bfloat> {{.*}}, i32 15
  return _mm256_set_pbh(bf1, bf2, bf3, bf4, bf5, bf6, bf7, bf8,
                       bf9, bf10, bf11, bf12, bf13, bf14, bf15, bf16);
}

__m128bh test_mm_setr_pbh(__bf16 bf1, __bf16 bf2, __bf16 bf3, __bf16 bf4,
                        __bf16 bf5, __bf16 bf6, __bf16 bf7, __bf16 bf8) {
  // CHECK-LABEL: @test_mm_setr_pbh
  // CHECK: insertelement <8 x bfloat> {{.*}}, i32 0
  // CHECK: insertelement <8 x bfloat> {{.*}}, i32 1
  // CHECK: insertelement <8 x bfloat> {{.*}}, i32 2
  // CHECK: insertelement <8 x bfloat> {{.*}}, i32 3
  // CHECK: insertelement <8 x bfloat> {{.*}}, i32 4
  // CHECK: insertelement <8 x bfloat> {{.*}}, i32 5
  // CHECK: insertelement <8 x bfloat> {{.*}}, i32 6
  // CHECK: insertelement <8 x bfloat> {{.*}}, i32 7
  return _mm_setr_pbh(bf1, bf2, bf3, bf4, bf5, bf6, bf7, bf8);
}

__m256bh test_mm256_setr_pbh(__bf16 bf1, __bf16 bf2, __bf16 bf3, __bf16 bf4,
                           __bf16 bf5, __bf16 bf6, __bf16 bf7, __bf16 bf8,
                           __bf16 bf9, __bf16 bf10, __bf16 bf11, __bf16 bf12,
                           __bf16 bf13, __bf16 bf14, __bf16 bf15, __bf16 bf16) {
  // CHECK-LABEL: @test_mm256_setr_pbh
  // CHECK: insertelement <16 x bfloat> {{.*}}, i32 0
  // CHECK: insertelement <16 x bfloat> {{.*}}, i32 1
  // CHECK: insertelement <16 x bfloat> {{.*}}, i32 2
  // CHECK: insertelement <16 x bfloat> {{.*}}, i32 3
  // CHECK: insertelement <16 x bfloat> {{.*}}, i32 4
  // CHECK: insertelement <16 x bfloat> {{.*}}, i32 5
  // CHECK: insertelement <16 x bfloat> {{.*}}, i32 6
  // CHECK: insertelement <16 x bfloat> {{.*}}, i32 7
  // CHECK: insertelement <16 x bfloat> {{.*}}, i32 8
  // CHECK: insertelement <16 x bfloat> {{.*}}, i32 9
  // CHECK: insertelement <16 x bfloat> {{.*}}, i32 10
  // CHECK: insertelement <16 x bfloat> {{.*}}, i32 11
  // CHECK: insertelement <16 x bfloat> {{.*}}, i32 12
  // CHECK: insertelement <16 x bfloat> {{.*}}, i32 13
  // CHECK: insertelement <16 x bfloat> {{.*}}, i32 14
  // CHECK: insertelement <16 x bfloat> {{.*}}, i32 15
  return _mm256_setr_pbh(bf1, bf2, bf3, bf4, bf5, bf6, bf7, bf8,
                        bf9, bf10, bf11, bf12, bf13, bf14, bf15, bf16);
}

__m128 test_mm_castpbf16_ps(__m128bh A) {
  // CHECK-LABEL: test_mm_castpbf16_ps
  // CHECK: bitcast <8 x bfloat> %{{.*}} to <4 x float>
  return _mm_castpbf16_ps(A);
}

__m256 test_mm256_castpbf16_ps(__m256bh A) {
  // CHECK-LABEL: test_mm256_castpbf16_ps
  // CHECK: bitcast <16 x bfloat> %{{.*}} to <8 x float>
  return _mm256_castpbf16_ps(A);
}

__m128i test_mm_castpbf16_si128(__m128bh A) {
  // CHECK-LABEL: test_mm_castpbf16_si128
  // CHECK: bitcast <8 x bfloat> %{{.*}} to <2 x i64>
  return _mm_castpbf16_si128(A);
}

__m256i test_mm256_castpbf16_si256(__m256bh A) {
  // CHECK-LABEL: test_mm256_castpbf16_si256
  // CHECK: bitcast <16 x bfloat> %{{.*}} to <4 x i64>
  return _mm256_castpbf16_si256(A);
}

__m128bh test_mm_castps_pbh(__m128 A) {
  // CHECK-LABEL: test_mm_castps_pbh
  // CHECK: bitcast <4 x float> %{{.*}} to <8 x bfloat>
  return _mm_castps_pbh(A);
}

__m256bh test_mm256_castps_pbh(__m256 A) {
  // CHECK-LABEL: test_mm256_castps_pbh
  // CHECK: bitcast <8 x float> %{{.*}} to <16 x bfloat>
  return _mm256_castps_pbh(A);
}

__m128bh test_mm_castpd_pbh(__m128d A) {
  // CHECK-LABEL: test_mm_castpd_pbh
  // CHECK: bitcast <2 x double> %{{.*}} to <8 x bfloat>
  return _mm_castpd_pbh(A);
}

__m256bh test_mm256_castpd_pbh(__m256d A) {
  // CHECK-LABEL: test_mm256_castpd_pbh
  // CHECK: bitcast <4 x double> %{{.*}} to <16 x bfloat>
  return _mm256_castpd_pbh(A);
}

__m128bh test_mm_castsi128_pbh(__m128i A) {
  // CHECK-LABEL: test_mm_castsi128_pbh
  // CHECK: bitcast <2 x i64> %{{.*}} to <8 x bfloat>
  return _mm_castsi128_pbh(A);
}

__m256bh test_mm256_castsi256_pbh(__m256i A) {
  // CHECK-LABEL: test_mm256_castsi256_pbh
  // CHECK: bitcast <4 x i64> %{{.*}} to <16 x bfloat>
  return _mm256_castsi256_pbh(A);
}

__m128d test_mm_castpbf16_pd(__m128bh A) {
  // CHECK-LABEL: test_mm_castpbf16_pd
  // CHECK: bitcast <8 x bfloat> %{{.*}} to <2 x double>
  return _mm_castpbf16_pd(A);
}

__m128bh test_mm256_castpbf16256_pbh128(__m256bh __a) {
  // CHECK-LABEL: test_mm256_castpbf16256_pbh128
  // CHECK: shufflevector <16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  return _mm256_castpbf16256_pbh128(__a);
}

__m256bh test_mm256_castpbf16128_pbh256(__m128bh __a) {
  // CHECK-LABEL: test_mm256_castpbf16128_pbh256
  // CHECK: shufflevector <8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  return _mm256_castpbf16128_pbh256(__a);
}

__m256d test_mm256_castpbf16_pd(__m256bh A) {
  // CHECK-LABEL: test_mm256_castpbf16_pd
  // CHECK: bitcast <16 x bfloat> %{{.*}} to <4 x double>
  return _mm256_castpbf16_pd(A);
}

__m256bh test_mm256_zextpbf16128_pbh256(__m128bh __a) {
  // CHECK-LABEL: test_mm256_zextpbf16128_pbh256
  // CHECK: shufflevector <8 x bfloat> %{{.*}}, <8 x bfloat> {{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  return _mm256_zextpbf16128_pbh256(__a);
}

__m128bh test_mm_abs_pbh(__m128bh a) {
  // CHECK-LABEL: @test_mm_abs_pbh
  // CHECK: and <4 x i32>
  return _mm_abs_pbh(a);
}

__m256bh test_mm256_abs_pbh(__m256bh a) {
  // CHECK-LABEL: @test_mm256_abs_pbh
  // CHECK: and <8 x i32>
  return _mm256_abs_pbh(a);
}

__m256bh test_mm256_loadu_pbh(void *p) {
  // CHECK-LABEL: @test_mm256_loadu_pbh
  // CHECK: load <16 x bfloat>, ptr {{.*}}, align 1{{$}}
  return _mm256_loadu_pbh(p);
}

__m128bh test_mm_load_sbh(void const *A) {
  // CHECK-LABEL: test_mm_load_sbh
  // CHECK: %{{.*}} = call <8 x bfloat> @llvm.masked.load.v8bf16.p0(ptr %{{.*}}, i32 1, <8 x i1> bitcast (<1 x i8> splat (i8 1) to <8 x i1>), <8 x bfloat> %{{.*}})
  return _mm_load_sbh(A);
}

__m256bh test_mm256_load_pbh(void *p) {
  // CHECK-LABEL: @test_mm256_load_pbh
  // CHECK: load <16 x bfloat>, ptr %{{.*}}, align 32
  return _mm256_load_pbh(p);
}

__m128bh test_mm_load_pbh(void *p) {
  // CHECK-LABEL: @test_mm_load_pbh
  // CHECK: load <8 x bfloat>, ptr %{{.*}}, align 16
  return _mm_load_pbh(p);
}

__m128bh test_mm_loadu_pbh(void *p) {
  // CHECK-LABEL: @test_mm_loadu_pbh
  // CHECK: load <8 x bfloat>, ptr {{.*}}, align 1{{$}}
  return _mm_loadu_pbh(p);
}

void test_mm_store_sbh(void *A, __m128bh B) {
  // CHECK-LABEL: test_mm_store_sbh
  // CHECK: extractelement <8 x bfloat> %{{.*}}, i32 0
  // CHECK: store bfloat %{{.*}}, ptr %{{.*}}, align 1{{$}}
  _mm_store_sbh(A, B);
}

void test_mm_mask_store_sbh(void *__P, __mmask8 __U, __m128bh __A) {
  // CHECK-LABEL: @test_mm_mask_store_sbh
  // CHECK: call void @llvm.masked.store.v8bf16.p0(<8 x bfloat> %{{.*}}, ptr %{{.*}}, i32 1, <8 x i1> %{{.*}})
  _mm_mask_store_sbh(__P, __U, __A);
}

void test_mm256_store_pbh(void *p, __m256bh a) {
  // CHECK-LABEL: @test_mm256_store_pbh
  // CHECK: store <16 x bfloat> %{{.*}}, ptr %{{.*}}, align 32
  _mm256_store_pbh(p, a);
}

void test_mm_store_pbh(void *p, __m128bh a) {
  // CHECK-LABEL: @test_mm_store_pbh
  // CHECK: store <8 x bfloat> %{{.*}}, ptr %{{.*}}, align 16
  _mm_store_pbh(p, a);
}

__m128bh test_mm_mask_load_sbh(__m128bh __A, __mmask8 __U, const void *__W) {
  // CHECK-LABEL: @test_mm_mask_load_sbh
  // CHECK: %{{.*}} = call <8 x bfloat> @llvm.masked.load.v8bf16.p0(ptr %{{.*}}, i32 1, <8 x i1> %{{.*}}, <8 x bfloat> %{{.*}})
  return _mm_mask_load_sbh(__A, __U, __W);
}

__m128bh test_mm_maskz_load_sbh(__mmask8 __U, const void *__W) {
  // CHECK-LABEL: @test_mm_maskz_load_sbh
  // CHECK: %{{.*}} = call <8 x bfloat> @llvm.masked.load.v8bf16.p0(ptr %{{.*}}, i32 1, <8 x i1> %{{.*}}, <8 x bfloat> %{{.*}})
  return _mm_maskz_load_sbh(__U, __W);
}

void test_mm256_storeu_pbh(void *p, __m256bh a) {
  // CHECK-LABEL: @test_mm256_storeu_pbh
  // CHECK: store <16 x bfloat> %{{.*}}, ptr %{{.*}}, align 1{{$}}
  // CHECK-NEXT: ret void
  _mm256_storeu_pbh(p, a);
}

void test_mm_storeu_pbh(void *p, __m128bh a) {
  // CHECK-LABEL: @test_mm_storeu_pbh
  // CHECK: store <8 x bfloat> %{{.*}}, ptr %{{.*}}, align 1{{$}}
  // CHECK-NEXT: ret void
  _mm_storeu_pbh(p, a);
}

__m128bh test_mm_move_sbh(__m128bh A, __m128bh B) {
  // CHECK-LABEL: test_mm_move_sbh
  // CHECK: extractelement <8 x bfloat> %{{.*}}, i32 0
  // CHECK: insertelement <8 x bfloat> %{{.*}}, bfloat %{{.*}}, i32 0
  return _mm_move_sbh(A, B);
}

__m128bh test_mm_mask_move_sbh(__m128bh __W, __mmask8 __U, __m128bh __A, __m128bh __B) {
  // CHECK-LABEL: @test_mm_mask_move_sbh
  // CHECK: [[EXT:%.*]] = extractelement <8 x bfloat> %{{.*}}, i32 0
  // CHECK: insertelement <8 x bfloat> %{{.*}}, bfloat [[EXT]], i32 0
  // CHECK: [[A:%.*]] = extractelement <8 x bfloat> [[VEC:%.*]], i64 0
  // CHECK-NEXT: [[B:%.*]] = extractelement <8 x bfloat> %{{.*}}, i64 0
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.*]] = select i1 %{{.*}}, bfloat [[A]], bfloat [[B]]
  // CHECK-NEXT: insertelement <8 x bfloat> [[VEC]], bfloat [[SEL]], i64 0
  return _mm_mask_move_sbh(__W, __U, __A, __B);
}

__m128bh test_mm_maskz_move_sbh(__mmask8 __U, __m128bh __A, __m128bh __B) {
  // CHECK-LABEL: @test_mm_maskz_move_sbh
  // CHECK: [[EXT:%.*]] = extractelement <8 x bfloat> %{{.*}}, i32 0
  // CHECK: insertelement <8 x bfloat> %{{.*}}, bfloat [[EXT]], i32 0
  // CHECK: [[A:%.*]] = extractelement <8 x bfloat> [[VEC:%.*]], i64 0
  // CHECK-NEXT: [[B:%.*]] = extractelement <8 x bfloat> %{{.*}}, i64 0
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.*]] = select i1 %{{.*}}, bfloat [[A]], bfloat [[B]]
  // CHECK-NEXT: insertelement <8 x bfloat> [[VEC]], bfloat [[SEL]], i64 0
  return _mm_maskz_move_sbh(__U, __A, __B);
}

__m128bh test_mm_mask_blend_pbh(__mmask8 __U, __m128bh __A, __m128bh __W) {
  // CHECK-LABEL: @test_mm_mask_blend_pbh
  // CHECK:  %{{.*}} = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK:  %{{.*}} = select <8 x i1> %{{.*}}, <8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}}
  return _mm_mask_blend_pbh(__U, __A, __W);
}

__m256bh test_mm256_mask_blend_pbh(__mmask16 __U, __m256bh __A, __m256bh __W) {
  // CHECK-LABEL: @test_mm256_mask_blend_pbh
  // CHECK:  %{{.*}} = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK:  %{{.*}} = select <16 x i1> %{{.*}}, <16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}}
  return _mm256_mask_blend_pbh(__U, __A, __W);
}

__m128bh test_mm_permutex2var_pbh(__m128bh __A, __m128i __I, __m128bh __B) {
  // CHECK-LABEL: @test_mm_permutex2var_pbh
  // CHECK:  %{{.*}} = bitcast <8 x bfloat> %{{.*}} to <8 x i16>
  // CHECK:  %{{.*}} = bitcast <2 x i64> %{{.*}} to <8 x i16>
  // CHECK:  %{{.*}} = bitcast <8 x bfloat> %{{.*}} to <8 x i16>
  // CHECK:  %{{.*}} = call <8 x i16> @llvm.x86.avx512.vpermi2var.hi.128(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK:  %{{.*}} = bitcast <8 x i16> %{{.*}} to <8 x bfloat>
  return _mm_permutex2var_pbh(__A, __I, __B);
}

__m256bh test_mm256_permutex2var_pbh(__m256bh __A, __m256i __I, __m256bh __B) {
  // CHECK-LABEL: @test_mm256_permutex2var_pbh
  // CHECK:  %{{.*}} = bitcast <16 x bfloat> %{{.*}} to <16 x i16>
  // CHECK:  %{{.*}} = bitcast <4 x i64> %{{.*}} to <16 x i16>
  // CHECK:  %{{.*}} = bitcast <16 x bfloat> %{{.*}} to <16 x i16>
  // CHECK:  %{{.*}} = call <16 x i16> @llvm.x86.avx512.vpermi2var.hi.256(<16 x i16> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  // CHECK:  %{{.*}} = bitcast <16 x i16> %{{.*}} to <16 x bfloat>
  return _mm256_permutex2var_pbh(__A, __I, __B);
}

__m128bh test_mm_permutexvar_pbh(__m128i __A, __m128bh __B) {
  // CHECK-LABEL: @test_mm_permutexvar_pbh
  // CHECK:  %{{.*}} = bitcast <8 x bfloat> %{{.*}} to <8 x i16>
  // CHECK:  %{{.*}} = bitcast <2 x i64> %{{.*}} to <8 x i16>
  // CHECK:  %{{.*}} = call <8 x i16> @llvm.x86.avx512.permvar.hi.128(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK:  %{{.*}} = bitcast <8 x i16> %{{.*}} to <8 x bfloat>
  return _mm_permutexvar_pbh(__A, __B);
}

__m256bh test_mm256_permutexvar_pbh(__m256i __A, __m256bh __B) {
  // CHECK-LABEL: @test_mm256_permutexvar_pbh
  // CHECK:  %{{.*}} = bitcast <16 x bfloat> %{{.*}} to <16 x i16>
  // CHECK:  %{{.*}} = bitcast <4 x i64> %{{.*}} to <16 x i16>
  // CHECK:  %{{.*}} = call <16 x i16> @llvm.x86.avx512.permvar.hi.256(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  // CHECK:  %{{.*}} = bitcast <16 x i16> %{{.*}} to <16 x bfloat>
  return _mm256_permutexvar_pbh(__A, __B);
}

__m256bh test_mm256_addne_pbh(__m256bh __A, __m256bh __B) {
  // CHECK-LABEL: @test_mm256_addne_pbh
  // CHECK: %{{.*}} = fadd <16 x bfloat> %{{.*}}, %{{.*}}
  return _mm256_addne_pbh(__A, __B);
}

__m256bh test_mm256_mask_addne_pbh(__m256bh __W, __mmask16 __U, __m256bh __A, __m256bh __B) {
  // CHECK: %{{.*}} = fadd <16 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: select <16 x i1> %{{.*}}, <16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}}
  return (__m256bh)_mm256_mask_addne_pbh(__W, __U, __A, __B);
}

__m256bh test_mm256_maskz_addne_pbh(__mmask16 __U, __m256bh __A, __m256bh __B) {
  // CHECK: %{{.*}} = fadd <16 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: select <16 x i1> %{{.*}}, <16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}}
  return _mm256_maskz_addne_pbh(__U, __A, __B);
}

__m128bh test_mm_addne_pbh(__m128bh __A, __m128bh __B) {
  // CHECK-LABEL: @test_mm_addne_pbh
  // CHECK: %{{.*}} = fadd <8 x bfloat> %{{.*}}, %{{.*}}
  return _mm_addne_pbh(__A, __B);
}

__m128bh test_mm_mask_addne_pbh(__m128bh __W, __mmask16 __U, __m128bh __A, __m128bh __B) {
  // CHECK: %{{.*}} = fadd <8 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: select <8 x i1> %{{.*}}, <8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}}
  return (__m128bh)_mm_mask_addne_pbh(__W, __U, __A, __B);
}

__m128bh test_mm_maskz_addne_pbh(__mmask16 __U, __m128bh __A, __m128bh __B) {
  // CHECK: %{{.*}} = fadd <8 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: select <8 x i1> %{{.*}}, <8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}}
  return _mm_maskz_addne_pbh(__U, __A, __B);
}

__m256bh test_mm256_subne_pbh(__m256bh __A, __m256bh __B) {
  // CHECK-LABEL: @test_mm256_subne_pbh
  // CHECK: %{{.*}} = fsub <16 x bfloat> %{{.*}}, %{{.*}}
  return _mm256_subne_pbh(__A, __B);
}

__m256bh test_mm256_mask_subne_pbh(__m256bh __W, __mmask16 __U, __m256bh __A, __m256bh __B) {
  // CHECK: %{{.*}} = fsub <16 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: select <16 x i1> %{{.*}}, <16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}}
  return (__m256bh)_mm256_mask_subne_pbh(__W, __U, __A, __B);
}

__m256bh test_mm256_maskz_subne_pbh(__mmask16 __U, __m256bh __A, __m256bh __B) {
  // CHECK: %{{.*}} = fsub <16 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: select <16 x i1> %{{.*}}, <16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}}
  return _mm256_maskz_subne_pbh(__U, __A, __B);
}

__m128bh test_mm_subne_pbh(__m128bh __A, __m128bh __B) {
  // CHECK-LABEL: @test_mm_subne_pbh
  // CHECK: %{{.*}} = fsub <8 x bfloat> %{{.*}}, %{{.*}}
  return _mm_subne_pbh(__A, __B);
}

__m128bh test_mm_mask_subne_pbh(__m128bh __W, __mmask16 __U, __m128bh __A, __m128bh __B) {
  // CHECK: %{{.*}} = fsub <8 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: select <8 x i1> %{{.*}}, <8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}}
  return (__m128bh)_mm_mask_subne_pbh(__W, __U, __A, __B);
}

__m128bh test_mm_maskz_subne_pbh(__mmask16 __U, __m128bh __A, __m128bh __B) {
  // CHECK: %{{.*}} = fsub <8 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: select <8 x i1> %{{.*}}, <8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}}
  return _mm_maskz_subne_pbh(__U, __A, __B);
}

__m256bh test_mm256_mulne_pbh(__m256bh __A, __m256bh __B) {
  // CHECK-LABEL: @test_mm256_mulne_pbh
  // CHECK: %{{.*}} = fmul <16 x bfloat> %{{.*}}, %{{.*}}
  return _mm256_mulne_pbh(__A, __B);
}

__m256bh test_mm256_mask_mulne_pbh(__m256bh __W, __mmask16 __U, __m256bh __A, __m256bh __B) {
  // CHECK: %{{.*}} = fmul <16 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: select <16 x i1> %{{.*}}, <16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}}
  return (__m256bh)_mm256_mask_mulne_pbh(__W, __U, __A, __B);
}

__m256bh test_mm256_maskz_mulne_pbh(__mmask16 __U, __m256bh __A, __m256bh __B) {
  // CHECK: %{{.*}} = fmul <16 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: select <16 x i1> %{{.*}}, <16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}}
  return _mm256_maskz_mulne_pbh(__U, __A, __B);
}

__m128bh test_mm_mulne_pbh(__m128bh __A, __m128bh __B) {
  // CHECK-LABEL: @test_mm_mulne_pbh
  // CHECK: %{{.*}} = fmul <8 x bfloat> %{{.*}}, %{{.*}}
  return _mm_mulne_pbh(__A, __B);
}

__m128bh test_mm_mask_mulne_pbh(__m128bh __W, __mmask16 __U, __m128bh __A, __m128bh __B) {
  // CHECK: %{{.*}} = fmul <8 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: select <8 x i1> %{{.*}}, <8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}}
  return (__m128bh)_mm_mask_mulne_pbh(__W, __U, __A, __B);
}

__m128bh test_mm_maskz_mulne_pbh(__mmask16 __U, __m128bh __A, __m128bh __B) {
  // CHECK: %{{.*}} = fmul <8 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: select <8 x i1> %{{.*}}, <8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}}
  return _mm_maskz_mulne_pbh(__U, __A, __B);
}

__m256bh test_mm256_divne_pbh(__m256bh __A, __m256bh __B) {
  // CHECK-LABEL: @test_mm256_divne_pbh
  // CHECK: %{{.*}} = fdiv <16 x bfloat> %{{.*}}, %{{.*}}
  return _mm256_divne_pbh(__A, __B);
}

__m256bh test_mm256_mask_divne_pbh(__m256bh __W, __mmask16 __U, __m256bh __A, __m256bh __B) {
  // CHECK: %{{.*}} = fdiv <16 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: select <16 x i1> %{{.*}}, <16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}}
  return (__m256bh)_mm256_mask_divne_pbh(__W, __U, __A, __B);
}

__m256bh test_mm256_maskz_divne_pbh(__mmask16 __U, __m256bh __A, __m256bh __B) {
  // CHECK: %{{.*}} = fdiv <16 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: select <16 x i1> %{{.*}}, <16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}}
  return _mm256_maskz_divne_pbh(__U, __A, __B);
}

__m128bh test_mm_divne_pbh(__m128bh __A, __m128bh __B) {
  // CHECK-LABEL: @test_mm_divne_pbh
  // CHECK: %{{.*}} = fdiv <8 x bfloat> %{{.*}}, %{{.*}}
  return _mm_divne_pbh(__A, __B);
}

__m128bh test_mm_mask_divne_pbh(__m128bh __W, __mmask16 __U, __m128bh __A, __m128bh __B) {
  // CHECK: %{{.*}} = fdiv <8 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: select <8 x i1> %{{.*}}, <8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}}
  return (__m128bh)_mm_mask_divne_pbh(__W, __U, __A, __B);
}

__m128bh test_mm_maskz_divne_pbh(__mmask16 __U, __m128bh __A, __m128bh __B) {
  // CHECK: %{{.*}} = fdiv <8 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: select <8 x i1> %{{.*}}, <8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}}
  return _mm_maskz_divne_pbh(__U, __A, __B);
}

__m256bh test_mm256_max_pbh(__m256bh __A, __m256bh __B) {
  // CHECK-LABEL: @test_mm256_max_pbh
  // CHECK: @llvm.x86.avx10.vmaxpbf16256(
  return _mm256_max_pbh(__A, __B);
}

__m256bh test_mm256_mask_max_pbh(__m256bh __W, __mmask16 __U, __m256bh __A, __m256bh __B) {
  // CHECK: @llvm.x86.avx10.vmaxpbf16256
  // CHECK: select <16 x i1> %{{.*}}, <16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}}
  return (__m256bh)_mm256_mask_max_pbh(__W, __U, __A, __B);
}

__m256bh test_mm256_maskz_max_pbh(__mmask16 __U, __m256bh __A, __m256bh __B) {
  // CHECK: @llvm.x86.avx10.vmaxpbf16256
  // CHECK: select <16 x i1> %{{.*}}, <16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}}
  return _mm256_maskz_max_pbh(__U, __A, __B);
}

__m128bh test_mm_max_pbh(__m128bh __A, __m128bh __B) {
  // CHECK-LABEL: @test_mm_max_pbh
  // CHECK: @llvm.x86.avx10.vmaxpbf16128(
  return _mm_max_pbh(__A, __B);
}

__m128bh test_mm_mask_max_pbh(__m128bh __W, __mmask16 __U, __m128bh __A, __m128bh __B) {
  // CHECK: @llvm.x86.avx10.vmaxpbf16128
  // CHECK: select <8 x i1> %{{.*}}, <8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}}
  return (__m128bh)_mm_mask_max_pbh(__W, __U, __A, __B);
}

__m128bh test_mm_maskz_max_pbh(__mmask16 __U, __m128bh __A, __m128bh __B) {
  // CHECK: @llvm.x86.avx10.vmaxpbf16128
  // CHECK: select <8 x i1> %{{.*}}, <8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}}
  return _mm_maskz_max_pbh(__U, __A, __B);
}

__m256bh test_mm256_min_pbh(__m256bh __A, __m256bh __B) {
  // CHECK-LABEL: @test_mm256_min_pbh
  // CHECK: @llvm.x86.avx10.vminpbf16256(
  return _mm256_min_pbh(__A, __B);
}

__m256bh test_mm256_mask_min_pbh(__m256bh __W, __mmask16 __U, __m256bh __A, __m256bh __B) {
  // CHECK: @llvm.x86.avx10.vminpbf16256
  // CHECK: select <16 x i1> %{{.*}}, <16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}}
  return (__m256bh)_mm256_mask_min_pbh(__W, __U, __A, __B);
}

__m256bh test_mm256_maskz_min_pbh(__mmask16 __U, __m256bh __A, __m256bh __B) {
  // CHECK: @llvm.x86.avx10.vminpbf16256
  // CHECK: select <16 x i1> %{{.*}}, <16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}}
  return _mm256_maskz_min_pbh(__U, __A, __B);
}

__m128bh test_mm_min_pbh(__m128bh __A, __m128bh __B) {
  // CHECK-LABEL: @test_mm_min_pbh
  // CHECK: @llvm.x86.avx10.vminpbf16128(
  return _mm_min_pbh(__A, __B);
}

__m128bh test_mm_mask_min_pbh(__m128bh __W, __mmask16 __U, __m128bh __A, __m128bh __B) {
  // CHECK: @llvm.x86.avx10.vminpbf16128
  // CHECK: select <8 x i1> %{{.*}}, <8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}}
  return (__m128bh)_mm_mask_min_pbh(__W, __U, __A, __B);
}

__m128bh test_mm_maskz_min_pbh(__mmask16 __U, __m128bh __A, __m128bh __B) {
  // CHECK: @llvm.x86.avx10.vminpbf16128
  // CHECK: select <8 x i1> %{{.*}}, <8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}}
  return _mm_maskz_min_pbh(__U, __A, __B);
}

int test_mm_comeqsbh(__m128bh __A, __m128bh __B) {
  // CHECK-LABEL: test_mm_comeqsbh
  // CHECK: %{{.}} = call i32 @llvm.x86.avx10.vcomsbf16eq(<8 x bfloat> %{{.}}, <8 x bfloat> %{{.}})
  return _mm_comeqsbh(__A, __B);
}

int test_mm_comltsbh(__m128bh __A, __m128bh __B) {
  // CHECK-LABEL: test_mm_comltsbh
  // CHECK: %{{.}} = call i32 @llvm.x86.avx10.vcomsbf16lt(<8 x bfloat> %{{.}}, <8 x bfloat> %{{.}})
  return _mm_comltsbh(__A, __B);
}

int test_mm_comlesbh(__m128bh __A, __m128bh __B) {
  // CHECK-LABEL: test_mm_comlesbh
  // CHECK: %{{.}} = call i32 @llvm.x86.avx10.vcomsbf16le(<8 x bfloat> %{{.}}, <8 x bfloat> %{{.}})
  return _mm_comlesbh(__A, __B);
}

int test_mm_comgtsbh(__m128bh __A, __m128bh __B) {
  // CHECK-LABEL: test_mm_comgtsbh
  // CHECK: %{{.}} = call i32 @llvm.x86.avx10.vcomsbf16gt(<8 x bfloat> %{{.}}, <8 x bfloat> %{{.}})
  return _mm_comgtsbh(__A, __B);
}

int test_mm_comgesbh(__m128bh __A, __m128bh __B) {
  // CHECK-LABEL: test_mm_comgesbh
  // CHECK: %{{.}} = call i32 @llvm.x86.avx10.vcomsbf16ge(<8 x bfloat> %{{.}}, <8 x bfloat> %{{.}})
  return _mm_comgesbh(__A, __B);
}

int test_mm_comneqsbh(__m128bh __A, __m128bh __B) {
  // CHECK-LABEL: test_mm_comneqsbh
  // CHECK: %{{.}} = call i32 @llvm.x86.avx10.vcomsbf16neq(<8 x bfloat> %{{.}}, <8 x bfloat> %{{.}})
  return _mm_comneqsbh(__A, __B);
}

__mmask16 test_mm256_cmp_pbh_mask_eq_oq(__m256bh a, __m256bh b) {
  // CHECK-LABEL: @test_mm256_cmp_pbh_mask_eq_oq
  // CHECK: fcmp oeq <16 x bfloat> %{{.*}}, %{{.*}}
  return _mm256_cmp_pbh_mask(a, b, _CMP_EQ_OQ);
}

__mmask16 test_mm256_cmp_pbh_mask_lt_os(__m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_cmp_pbh_mask_lt_os
  // CHECK: fcmp olt <16 x bfloat> %{{.*}}, %{{.*}}
  return _mm256_cmp_pbh_mask(a, b, _CMP_LT_OS);
}

__mmask16 test_mm256_cmp_pbh_mask_le_os(__m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_cmp_pbh_mask_le_os
  // CHECK: fcmp ole <16 x bfloat> %{{.*}}, %{{.*}}
  return _mm256_cmp_pbh_mask(a, b, _CMP_LE_OS);
}

__mmask16 test_mm256_cmp_pbh_mask_unord_q(__m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_cmp_pbh_mask_unord_q
  // CHECK: fcmp uno <16 x bfloat> %{{.*}}, %{{.*}}
  return _mm256_cmp_pbh_mask(a, b, _CMP_UNORD_Q);
}

__mmask16 test_mm256_cmp_pbh_mask_neq_uq(__m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_cmp_pbh_mask_neq_uq
  // CHECK: fcmp une <16 x bfloat> %{{.*}}, %{{.*}}
  return _mm256_cmp_pbh_mask(a, b, _CMP_NEQ_UQ);
}

__mmask16 test_mm256_cmp_pbh_mask_nlt_us(__m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_cmp_pbh_mask_nlt_us
  // CHECK: fcmp uge <16 x bfloat> %{{.*}}, %{{.*}}
  return _mm256_cmp_pbh_mask(a, b, _CMP_NLT_US);
}

__mmask16 test_mm256_cmp_pbh_mask_nle_us(__m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_cmp_pbh_mask_nle_us
  // CHECK: fcmp ugt <16 x bfloat> %{{.*}}, %{{.*}}
  return _mm256_cmp_pbh_mask(a, b, _CMP_NLE_US);
}

__mmask16 test_mm256_cmp_pbh_mask_ord_q(__m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_cmp_pbh_mask_ord_q
  // CHECK: fcmp ord <16 x bfloat> %{{.*}}, %{{.*}}
  return _mm256_cmp_pbh_mask(a, b, _CMP_ORD_Q);
}

__mmask16 test_mm256_cmp_pbh_mask_eq_uq(__m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_cmp_pbh_mask_eq_uq
  // CHECK: fcmp ueq <16 x bfloat> %{{.*}}, %{{.*}}
  return _mm256_cmp_pbh_mask(a, b, _CMP_EQ_UQ);
}

__mmask16 test_mm256_cmp_pbh_mask_nge_us(__m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_cmp_pbh_mask_nge_us
  // CHECK: fcmp ult <16 x bfloat> %{{.*}}, %{{.*}}
  return _mm256_cmp_pbh_mask(a, b, _CMP_NGE_US);
}

__mmask16 test_mm256_cmp_pbh_mask_ngt_us(__m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_cmp_pbh_mask_ngt_us
  // CHECK: fcmp ule <16 x bfloat> %{{.*}}, %{{.*}}
  return _mm256_cmp_pbh_mask(a, b, _CMP_NGT_US);
}

__mmask16 test_mm256_cmp_pbh_mask_false_oq(__m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_cmp_pbh_mask_false_oq
  // CHECK: fcmp false <16 x bfloat> %{{.*}}, %{{.*}}
  return _mm256_cmp_pbh_mask(a, b, _CMP_FALSE_OQ);
}

__mmask16 test_mm256_cmp_pbh_mask_neq_oq(__m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_cmp_pbh_mask_neq_oq
  // CHECK: fcmp one <16 x bfloat> %{{.*}}, %{{.*}}
  return _mm256_cmp_pbh_mask(a, b, _CMP_NEQ_OQ);
}

__mmask16 test_mm256_cmp_pbh_mask_ge_os(__m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_cmp_pbh_mask_ge_os
  // CHECK: fcmp oge <16 x bfloat> %{{.*}}, %{{.*}}
  return _mm256_cmp_pbh_mask(a, b, _CMP_GE_OS);
}

__mmask16 test_mm256_cmp_pbh_mask_gt_os(__m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_cmp_pbh_mask_gt_os
  // CHECK: fcmp ogt <16 x bfloat> %{{.*}}, %{{.*}}
  return _mm256_cmp_pbh_mask(a, b, _CMP_GT_OS);
}

__mmask16 test_mm256_cmp_pbh_mask_true_uq(__m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_cmp_pbh_mask_true_uq
  // CHECK: fcmp true <16 x bfloat> %{{.*}}, %{{.*}}
  return _mm256_cmp_pbh_mask(a, b, _CMP_TRUE_UQ);
}

__mmask16 test_mm256_cmp_pbh_mask_eq_os(__m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_cmp_pbh_mask_eq_os
  // CHECK: fcmp oeq <16 x bfloat> %{{.*}}, %{{.*}}
  return _mm256_cmp_pbh_mask(a, b, _CMP_EQ_OS);
}

__mmask16 test_mm256_cmp_pbh_mask_lt_oq(__m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_cmp_pbh_mask_lt_oq
  // CHECK: fcmp olt <16 x bfloat> %{{.*}}, %{{.*}}
  return _mm256_cmp_pbh_mask(a, b, _CMP_LT_OQ);
}

__mmask16 test_mm256_cmp_pbh_mask_le_oq(__m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_cmp_pbh_mask_le_oq
  // CHECK: fcmp ole <16 x bfloat> %{{.*}}, %{{.*}}
  return _mm256_cmp_pbh_mask(a, b, _CMP_LE_OQ);
}

__mmask16 test_mm256_cmp_pbh_mask_unord_s(__m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_cmp_pbh_mask_unord_s
  // CHECK: fcmp uno <16 x bfloat> %{{.*}}, %{{.*}}
  return _mm256_cmp_pbh_mask(a, b, _CMP_UNORD_S);
}

__mmask16 test_mm256_cmp_pbh_mask_neq_us(__m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_cmp_pbh_mask_neq_us
  // CHECK: fcmp une <16 x bfloat> %{{.*}}, %{{.*}}
  return _mm256_cmp_pbh_mask(a, b, _CMP_NEQ_US);
}

__mmask16 test_mm256_cmp_pbh_mask_nlt_uq(__m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_cmp_pbh_mask_nlt_uq
  // CHECK: fcmp uge <16 x bfloat> %{{.*}}, %{{.*}}
  return _mm256_cmp_pbh_mask(a, b, _CMP_NLT_UQ);
}

__mmask16 test_mm256_cmp_pbh_mask_nle_uq(__m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_cmp_pbh_mask_nle_uq
  // CHECK: fcmp ugt <16 x bfloat> %{{.*}}, %{{.*}}
  return _mm256_cmp_pbh_mask(a, b, _CMP_NLE_UQ);
}

__mmask16 test_mm256_cmp_pbh_mask_ord_s(__m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_cmp_pbh_mask_ord_s
  // CHECK: fcmp ord <16 x bfloat> %{{.*}}, %{{.*}}
  return _mm256_cmp_pbh_mask(a, b, _CMP_ORD_S);
}

__mmask16 test_mm256_cmp_pbh_mask_eq_us(__m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_cmp_pbh_mask_eq_us
  // CHECK: fcmp ueq <16 x bfloat> %{{.*}}, %{{.*}}
  return _mm256_cmp_pbh_mask(a, b, _CMP_EQ_US);
}

__mmask16 test_mm256_cmp_pbh_mask_nge_uq(__m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_cmp_pbh_mask_nge_uq
  // CHECK: fcmp ult <16 x bfloat> %{{.*}}, %{{.*}}
  return _mm256_cmp_pbh_mask(a, b, _CMP_NGE_UQ);
}

__mmask16 test_mm256_cmp_pbh_mask_ngt_uq(__m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_cmp_pbh_mask_ngt_uq
  // CHECK: fcmp ule <16 x bfloat> %{{.*}}, %{{.*}}
  return _mm256_cmp_pbh_mask(a, b, _CMP_NGT_UQ);
}

__mmask16 test_mm256_cmp_pbh_mask_false_os(__m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_cmp_pbh_mask_false_os
  // CHECK: fcmp false <16 x bfloat> %{{.*}}, %{{.*}}
  return _mm256_cmp_pbh_mask(a, b, _CMP_FALSE_OS);
}

__mmask16 test_mm256_cmp_pbh_mask_neq_os(__m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_cmp_pbh_mask_neq_os
  // CHECK: fcmp one <16 x bfloat> %{{.*}}, %{{.*}}
  return _mm256_cmp_pbh_mask(a, b, _CMP_NEQ_OS);
}

__mmask16 test_mm256_cmp_pbh_mask_ge_oq(__m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_cmp_pbh_mask_ge_oq
  // CHECK: fcmp oge <16 x bfloat> %{{.*}}, %{{.*}}
  return _mm256_cmp_pbh_mask(a, b, _CMP_GE_OQ);
}

__mmask16 test_mm256_cmp_pbh_mask_gt_oq(__m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_cmp_pbh_mask_gt_oq
  // CHECK: fcmp ogt <16 x bfloat> %{{.*}}, %{{.*}}
  return _mm256_cmp_pbh_mask(a, b, _CMP_GT_OQ);
}

__mmask16 test_mm256_cmp_pbh_mask_true_us(__m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_cmp_pbh_mask_true_us
  // CHECK: fcmp true <16 x bfloat> %{{.*}}, %{{.*}}
  return _mm256_cmp_pbh_mask(a, b, _CMP_TRUE_US);
}

__mmask16 test_mm256_mask_cmp_pbh_mask_eq_oq(__mmask16 m, __m256bh a, __m256bh b) {
  // CHECK-LABEL: @test_mm256_mask_cmp_pbh_mask_eq_oq
  // CHECK: fcmp oeq <16 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return _mm256_mask_cmp_pbh_mask(m, a, b, _CMP_EQ_OQ);
}

__mmask16 test_mm256_mask_cmp_pbh_mask_lt_os(__mmask16 m, __m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pbh_mask_lt_os
  // CHECK: fcmp olt <16 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return _mm256_mask_cmp_pbh_mask(m, a, b, _CMP_LT_OS);
}

__mmask16 test_mm256_mask_cmp_pbh_mask_le_os(__mmask16 m, __m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pbh_mask_le_os
  // CHECK: fcmp ole <16 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return _mm256_mask_cmp_pbh_mask(m, a, b, _CMP_LE_OS);
}

__mmask16 test_mm256_mask_cmp_pbh_mask_unord_q(__mmask16 m, __m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pbh_mask_unord_q
  // CHECK: fcmp uno <16 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return _mm256_mask_cmp_pbh_mask(m, a, b, _CMP_UNORD_Q);
}

__mmask16 test_mm256_mask_cmp_pbh_mask_neq_uq(__mmask16 m, __m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pbh_mask_neq_uq
  // CHECK: fcmp une <16 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return _mm256_mask_cmp_pbh_mask(m, a, b, _CMP_NEQ_UQ);
}

__mmask16 test_mm256_mask_cmp_pbh_mask_nlt_us(__mmask16 m, __m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pbh_mask_nlt_us
  // CHECK: fcmp uge <16 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return _mm256_mask_cmp_pbh_mask(m, a, b, _CMP_NLT_US);
}

__mmask16 test_mm256_mask_cmp_pbh_mask_nle_us(__mmask16 m, __m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pbh_mask_nle_us
  // CHECK: fcmp ugt <16 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return _mm256_mask_cmp_pbh_mask(m, a, b, _CMP_NLE_US);
}

__mmask16 test_mm256_mask_cmp_pbh_mask_ord_q(__mmask16 m, __m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pbh_mask_ord_q
  // CHECK: fcmp ord <16 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return _mm256_mask_cmp_pbh_mask(m, a, b, _CMP_ORD_Q);
}

__mmask16 test_mm256_mask_cmp_pbh_mask_eq_uq(__mmask16 m, __m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pbh_mask_eq_uq
  // CHECK: fcmp ueq <16 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return _mm256_mask_cmp_pbh_mask(m, a, b, _CMP_EQ_UQ);
}

__mmask16 test_mm256_mask_cmp_pbh_mask_nge_us(__mmask16 m, __m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pbh_mask_nge_us
  // CHECK: fcmp ult <16 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return _mm256_mask_cmp_pbh_mask(m, a, b, _CMP_NGE_US);
}

__mmask16 test_mm256_mask_cmp_pbh_mask_ngt_us(__mmask16 m, __m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pbh_mask_ngt_us
  // CHECK: fcmp ule <16 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return _mm256_mask_cmp_pbh_mask(m, a, b, _CMP_NGT_US);
}

__mmask16 test_mm256_mask_cmp_pbh_mask_false_oq(__mmask16 m, __m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pbh_mask_false_oq
  // CHECK: fcmp false <16 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return _mm256_mask_cmp_pbh_mask(m, a, b, _CMP_FALSE_OQ);
}

__mmask16 test_mm256_mask_cmp_pbh_mask_neq_oq(__mmask16 m, __m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pbh_mask_neq_oq
  // CHECK: fcmp one <16 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return _mm256_mask_cmp_pbh_mask(m, a, b, _CMP_NEQ_OQ);
}

__mmask16 test_mm256_mask_cmp_pbh_mask_ge_os(__mmask16 m, __m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pbh_mask_ge_os
  // CHECK: fcmp oge <16 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return _mm256_mask_cmp_pbh_mask(m, a, b, _CMP_GE_OS);
}

__mmask16 test_mm256_mask_cmp_pbh_mask_gt_os(__mmask16 m, __m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pbh_mask_gt_os
  // CHECK: fcmp ogt <16 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return _mm256_mask_cmp_pbh_mask(m, a, b, _CMP_GT_OS);
}

__mmask16 test_mm256_mask_cmp_pbh_mask_true_uq(__mmask16 m, __m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pbh_mask_true_uq
  // CHECK: fcmp true <16 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return _mm256_mask_cmp_pbh_mask(m, a, b, _CMP_TRUE_UQ);
}

__mmask16 test_mm256_mask_cmp_pbh_mask_eq_os(__mmask16 m, __m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pbh_mask_eq_os
  // CHECK: fcmp oeq <16 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return _mm256_mask_cmp_pbh_mask(m, a, b, _CMP_EQ_OS);
}

__mmask16 test_mm256_mask_cmp_pbh_mask_lt_oq(__mmask16 m, __m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pbh_mask_lt_oq
  // CHECK: fcmp olt <16 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return _mm256_mask_cmp_pbh_mask(m, a, b, _CMP_LT_OQ);
}

__mmask16 test_mm256_mask_cmp_pbh_mask_le_oq(__mmask16 m, __m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pbh_mask_le_oq
  // CHECK: fcmp ole <16 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return _mm256_mask_cmp_pbh_mask(m, a, b, _CMP_LE_OQ);
}

__mmask16 test_mm256_mask_cmp_pbh_mask_unord_s(__mmask16 m, __m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pbh_mask_unord_s
  // CHECK: fcmp uno <16 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return _mm256_mask_cmp_pbh_mask(m, a, b, _CMP_UNORD_S);
}

__mmask16 test_mm256_mask_cmp_pbh_mask_neq_us(__mmask16 m, __m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pbh_mask_neq_us
  // CHECK: fcmp une <16 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return _mm256_mask_cmp_pbh_mask(m, a, b, _CMP_NEQ_US);
}

__mmask16 test_mm256_mask_cmp_pbh_mask_nlt_uq(__mmask16 m, __m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pbh_mask_nlt_uq
  // CHECK: fcmp uge <16 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return _mm256_mask_cmp_pbh_mask(m, a, b, _CMP_NLT_UQ);
}

__mmask16 test_mm256_mask_cmp_pbh_mask_nle_uq(__mmask16 m, __m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pbh_mask_nle_uq
  // CHECK: fcmp ugt <16 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return _mm256_mask_cmp_pbh_mask(m, a, b, _CMP_NLE_UQ);
}

__mmask16 test_mm256_mask_cmp_pbh_mask_ord_s(__mmask16 m, __m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pbh_mask_ord_s
  // CHECK: fcmp ord <16 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return _mm256_mask_cmp_pbh_mask(m, a, b, _CMP_ORD_S);
}

__mmask16 test_mm256_mask_cmp_pbh_mask_eq_us(__mmask16 m, __m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pbh_mask_eq_us
  // CHECK: fcmp ueq <16 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return _mm256_mask_cmp_pbh_mask(m, a, b, _CMP_EQ_US);
}

__mmask16 test_mm256_mask_cmp_pbh_mask_nge_uq(__mmask16 m, __m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pbh_mask_nge_uq
  // CHECK: fcmp ult <16 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return _mm256_mask_cmp_pbh_mask(m, a, b, _CMP_NGE_UQ);
}

__mmask16 test_mm256_mask_cmp_pbh_mask_ngt_uq(__mmask16 m, __m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pbh_mask_ngt_uq
  // CHECK: fcmp ule <16 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return _mm256_mask_cmp_pbh_mask(m, a, b, _CMP_NGT_UQ);
}

__mmask16 test_mm256_mask_cmp_pbh_mask_false_os(__mmask16 m, __m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pbh_mask_false_os
  // CHECK: fcmp false <16 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return _mm256_mask_cmp_pbh_mask(m, a, b, _CMP_FALSE_OS);
}

__mmask16 test_mm256_mask_cmp_pbh_mask_neq_os(__mmask16 m, __m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pbh_mask_neq_os
  // CHECK: fcmp one <16 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return _mm256_mask_cmp_pbh_mask(m, a, b, _CMP_NEQ_OS);
}

__mmask16 test_mm256_mask_cmp_pbh_mask_ge_oq(__mmask16 m, __m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pbh_mask_ge_oq
  // CHECK: fcmp oge <16 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return _mm256_mask_cmp_pbh_mask(m, a, b, _CMP_GE_OQ);
}

__mmask16 test_mm256_mask_cmp_pbh_mask_gt_oq(__mmask16 m, __m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pbh_mask_gt_oq
  // CHECK: fcmp ogt <16 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return _mm256_mask_cmp_pbh_mask(m, a, b, _CMP_GT_OQ);
}

__mmask16 test_mm256_mask_cmp_pbh_mask_true_us(__mmask16 m, __m256bh a, __m256bh b) {
  // CHECK-LABEL: test_mm256_mask_cmp_pbh_mask_true_us
  // CHECK: fcmp true <16 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return _mm256_mask_cmp_pbh_mask(m, a, b, _CMP_TRUE_US);
}

__mmask8 test_mm_cmp_pbh_mask_eq_oq(__m128bh a, __m128bh b) {
  // CHECK-LABEL: @test_mm_cmp_pbh_mask_eq_oq
  // CHECK: fcmp oeq <8 x bfloat> %{{.*}}, %{{.*}}
  return _mm_cmp_pbh_mask(a, b, _CMP_EQ_OQ);
}

__mmask8 test_mm_cmp_pbh_mask_lt_os(__m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_cmp_pbh_mask_lt_os
  // CHECK: fcmp olt <8 x bfloat> %{{.*}}, %{{.*}}
  return _mm_cmp_pbh_mask(a, b, _CMP_LT_OS);
}

__mmask8 test_mm_cmp_pbh_mask_le_os(__m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_cmp_pbh_mask_le_os
  // CHECK: fcmp ole <8 x bfloat> %{{.*}}, %{{.*}}
  return _mm_cmp_pbh_mask(a, b, _CMP_LE_OS);
}

__mmask8 test_mm_cmp_pbh_mask_unord_q(__m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_cmp_pbh_mask_unord_q
  // CHECK: fcmp uno <8 x bfloat> %{{.*}}, %{{.*}}
  return _mm_cmp_pbh_mask(a, b, _CMP_UNORD_Q);
}

__mmask8 test_mm_cmp_pbh_mask_neq_uq(__m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_cmp_pbh_mask_neq_uq
  // CHECK: fcmp une <8 x bfloat> %{{.*}}, %{{.*}}
  return _mm_cmp_pbh_mask(a, b, _CMP_NEQ_UQ);
}

__mmask8 test_mm_cmp_pbh_mask_nlt_us(__m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_cmp_pbh_mask_nlt_us
  // CHECK: fcmp uge <8 x bfloat> %{{.*}}, %{{.*}}
  return _mm_cmp_pbh_mask(a, b, _CMP_NLT_US);
}

__mmask8 test_mm_cmp_pbh_mask_nle_us(__m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_cmp_pbh_mask_nle_us
  // CHECK: fcmp ugt <8 x bfloat> %{{.*}}, %{{.*}}
  return _mm_cmp_pbh_mask(a, b, _CMP_NLE_US);
}

__mmask8 test_mm_cmp_pbh_mask_ord_q(__m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_cmp_pbh_mask_ord_q
  // CHECK: fcmp ord <8 x bfloat> %{{.*}}, %{{.*}}
  return _mm_cmp_pbh_mask(a, b, _CMP_ORD_Q);
}

__mmask8 test_mm_cmp_pbh_mask_eq_uq(__m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_cmp_pbh_mask_eq_uq
  // CHECK: fcmp ueq <8 x bfloat> %{{.*}}, %{{.*}}
  return _mm_cmp_pbh_mask(a, b, _CMP_EQ_UQ);
}

__mmask8 test_mm_cmp_pbh_mask_nge_us(__m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_cmp_pbh_mask_nge_us
  // CHECK: fcmp ult <8 x bfloat> %{{.*}}, %{{.*}}
  return _mm_cmp_pbh_mask(a, b, _CMP_NGE_US);
}

__mmask8 test_mm_cmp_pbh_mask_ngt_us(__m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_cmp_pbh_mask_ngt_us
  // CHECK: fcmp ule <8 x bfloat> %{{.*}}, %{{.*}}
  return _mm_cmp_pbh_mask(a, b, _CMP_NGT_US);
}

__mmask8 test_mm_cmp_pbh_mask_false_oq(__m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_cmp_pbh_mask_false_oq
  // CHECK: fcmp false <8 x bfloat> %{{.*}}, %{{.*}}
  return _mm_cmp_pbh_mask(a, b, _CMP_FALSE_OQ);
}

__mmask8 test_mm_cmp_pbh_mask_neq_oq(__m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_cmp_pbh_mask_neq_oq
  // CHECK: fcmp one <8 x bfloat> %{{.*}}, %{{.*}}
  return _mm_cmp_pbh_mask(a, b, _CMP_NEQ_OQ);
}

__mmask8 test_mm_cmp_pbh_mask_ge_os(__m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_cmp_pbh_mask_ge_os
  // CHECK: fcmp oge <8 x bfloat> %{{.*}}, %{{.*}}
  return _mm_cmp_pbh_mask(a, b, _CMP_GE_OS);
}

__mmask8 test_mm_cmp_pbh_mask_gt_os(__m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_cmp_pbh_mask_gt_os
  // CHECK: fcmp ogt <8 x bfloat> %{{.*}}, %{{.*}}
  return _mm_cmp_pbh_mask(a, b, _CMP_GT_OS);
}

__mmask8 test_mm_cmp_pbh_mask_true_uq(__m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_cmp_pbh_mask_true_uq
  // CHECK: fcmp true <8 x bfloat> %{{.*}}, %{{.*}}
  return _mm_cmp_pbh_mask(a, b, _CMP_TRUE_UQ);
}

__mmask8 test_mm_cmp_pbh_mask_eq_os(__m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_cmp_pbh_mask_eq_os
  // CHECK: fcmp oeq <8 x bfloat> %{{.*}}, %{{.*}}
  return _mm_cmp_pbh_mask(a, b, _CMP_EQ_OS);
}

__mmask8 test_mm_cmp_pbh_mask_lt_oq(__m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_cmp_pbh_mask_lt_oq
  // CHECK: fcmp olt <8 x bfloat> %{{.*}}, %{{.*}}
  return _mm_cmp_pbh_mask(a, b, _CMP_LT_OQ);
}

__mmask8 test_mm_cmp_pbh_mask_le_oq(__m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_cmp_pbh_mask_le_oq
  // CHECK: fcmp ole <8 x bfloat> %{{.*}}, %{{.*}}
  return _mm_cmp_pbh_mask(a, b, _CMP_LE_OQ);
}

__mmask8 test_mm_cmp_pbh_mask_unord_s(__m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_cmp_pbh_mask_unord_s
  // CHECK: fcmp uno <8 x bfloat> %{{.*}}, %{{.*}}
  return _mm_cmp_pbh_mask(a, b, _CMP_UNORD_S);
}

__mmask8 test_mm_cmp_pbh_mask_neq_us(__m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_cmp_pbh_mask_neq_us
  // CHECK: fcmp une <8 x bfloat> %{{.*}}, %{{.*}}
  return _mm_cmp_pbh_mask(a, b, _CMP_NEQ_US);
}

__mmask8 test_mm_cmp_pbh_mask_nlt_uq(__m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_cmp_pbh_mask_nlt_uq
  // CHECK: fcmp uge <8 x bfloat> %{{.*}}, %{{.*}}
  return _mm_cmp_pbh_mask(a, b, _CMP_NLT_UQ);
}

__mmask8 test_mm_cmp_pbh_mask_nle_uq(__m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_cmp_pbh_mask_nle_uq
  // CHECK: fcmp ugt <8 x bfloat> %{{.*}}, %{{.*}}
  return _mm_cmp_pbh_mask(a, b, _CMP_NLE_UQ);
}

__mmask8 test_mm_cmp_pbh_mask_ord_s(__m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_cmp_pbh_mask_ord_s
  // CHECK: fcmp ord <8 x bfloat> %{{.*}}, %{{.*}}
  return _mm_cmp_pbh_mask(a, b, _CMP_ORD_S);
}

__mmask8 test_mm_cmp_pbh_mask_eq_us(__m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_cmp_pbh_mask_eq_us
  // CHECK: fcmp ueq <8 x bfloat> %{{.*}}, %{{.*}}
  return _mm_cmp_pbh_mask(a, b, _CMP_EQ_US);
}

__mmask8 test_mm_cmp_pbh_mask_nge_uq(__m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_cmp_pbh_mask_nge_uq
  // CHECK: fcmp ult <8 x bfloat> %{{.*}}, %{{.*}}
  return _mm_cmp_pbh_mask(a, b, _CMP_NGE_UQ);
}

__mmask8 test_mm_cmp_pbh_mask_ngt_uq(__m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_cmp_pbh_mask_ngt_uq
  // CHECK: fcmp ule <8 x bfloat> %{{.*}}, %{{.*}}
  return _mm_cmp_pbh_mask(a, b, _CMP_NGT_UQ);
}

__mmask8 test_mm_cmp_pbh_mask_false_os(__m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_cmp_pbh_mask_false_os
  // CHECK: fcmp false <8 x bfloat> %{{.*}}, %{{.*}}
  return _mm_cmp_pbh_mask(a, b, _CMP_FALSE_OS);
}

__mmask8 test_mm_cmp_pbh_mask_neq_os(__m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_cmp_pbh_mask_neq_os
  // CHECK: fcmp one <8 x bfloat> %{{.*}}, %{{.*}}
  return _mm_cmp_pbh_mask(a, b, _CMP_NEQ_OS);
}

__mmask8 test_mm_cmp_pbh_mask_ge_oq(__m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_cmp_pbh_mask_ge_oq
  // CHECK: fcmp oge <8 x bfloat> %{{.*}}, %{{.*}}
  return _mm_cmp_pbh_mask(a, b, _CMP_GE_OQ);
}

__mmask8 test_mm_cmp_pbh_mask_gt_oq(__m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_cmp_pbh_mask_gt_oq
  // CHECK: fcmp ogt <8 x bfloat> %{{.*}}, %{{.*}}
  return _mm_cmp_pbh_mask(a, b, _CMP_GT_OQ);
}

__mmask8 test_mm_cmp_pbh_mask_true_us(__m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_cmp_pbh_mask_true_us
  // CHECK: fcmp true <8 x bfloat> %{{.*}}, %{{.*}}
  return _mm_cmp_pbh_mask(a, b, _CMP_TRUE_US);
}

__mmask8 test_mm_mask_cmp_pbh_mask_eq_oq(__mmask8 m, __m128bh a, __m128bh b) {
  // CHECK-LABEL: @test_mm_mask_cmp_pbh_mask_eq_oq
  // CHECK: fcmp oeq <8 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return _mm_mask_cmp_pbh_mask(m, a, b, _CMP_EQ_OQ);
}

__mmask8 test_mm_mask_cmp_pbh_mask_lt_os(__mmask8 m, __m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_mask_cmp_pbh_mask_lt_os
  // CHECK: fcmp olt <8 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return _mm_mask_cmp_pbh_mask(m, a, b, _CMP_LT_OS);
}

__mmask8 test_mm_mask_cmp_pbh_mask_le_os(__mmask8 m, __m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_mask_cmp_pbh_mask_le_os
  // CHECK: fcmp ole <8 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return _mm_mask_cmp_pbh_mask(m, a, b, _CMP_LE_OS);
}

__mmask8 test_mm_mask_cmp_pbh_mask_unord_q(__mmask8 m, __m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_mask_cmp_pbh_mask_unord_q
  // CHECK: fcmp uno <8 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return _mm_mask_cmp_pbh_mask(m, a, b, _CMP_UNORD_Q);
}

__mmask8 test_mm_mask_cmp_pbh_mask_neq_uq(__mmask8 m, __m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_mask_cmp_pbh_mask_neq_uq
  // CHECK: fcmp une <8 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return _mm_mask_cmp_pbh_mask(m, a, b, _CMP_NEQ_UQ);
}

__mmask8 test_mm_mask_cmp_pbh_mask_nlt_us(__mmask8 m, __m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_mask_cmp_pbh_mask_nlt_us
  // CHECK: fcmp uge <8 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return _mm_mask_cmp_pbh_mask(m, a, b, _CMP_NLT_US);
}

__mmask8 test_mm_mask_cmp_pbh_mask_nle_us(__mmask8 m, __m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_mask_cmp_pbh_mask_nle_us
  // CHECK: fcmp ugt <8 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return _mm_mask_cmp_pbh_mask(m, a, b, _CMP_NLE_US);
}

__mmask8 test_mm_mask_cmp_pbh_mask_ord_q(__mmask8 m, __m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_mask_cmp_pbh_mask_ord_q
  // CHECK: fcmp ord <8 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return _mm_mask_cmp_pbh_mask(m, a, b, _CMP_ORD_Q);
}

__mmask8 test_mm_mask_cmp_pbh_mask_eq_uq(__mmask8 m, __m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_mask_cmp_pbh_mask_eq_uq
  // CHECK: fcmp ueq <8 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return _mm_mask_cmp_pbh_mask(m, a, b, _CMP_EQ_UQ);
}

__mmask8 test_mm_mask_cmp_pbh_mask_nge_us(__mmask8 m, __m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_mask_cmp_pbh_mask_nge_us
  // CHECK: fcmp ult <8 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return _mm_mask_cmp_pbh_mask(m, a, b, _CMP_NGE_US);
}

__mmask8 test_mm_mask_cmp_pbh_mask_ngt_us(__mmask8 m, __m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_mask_cmp_pbh_mask_ngt_us
  // CHECK: fcmp ule <8 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return _mm_mask_cmp_pbh_mask(m, a, b, _CMP_NGT_US);
}

__mmask8 test_mm_mask_cmp_pbh_mask_false_oq(__mmask8 m, __m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_mask_cmp_pbh_mask_false_oq
  // CHECK: fcmp false <8 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return _mm_mask_cmp_pbh_mask(m, a, b, _CMP_FALSE_OQ);
}

__mmask8 test_mm_mask_cmp_pbh_mask_neq_oq(__mmask8 m, __m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_mask_cmp_pbh_mask_neq_oq
  // CHECK: fcmp one <8 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return _mm_mask_cmp_pbh_mask(m, a, b, _CMP_NEQ_OQ);
}

__mmask8 test_mm_mask_cmp_pbh_mask_ge_os(__mmask8 m, __m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_mask_cmp_pbh_mask_ge_os
  // CHECK: fcmp oge <8 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return _mm_mask_cmp_pbh_mask(m, a, b, _CMP_GE_OS);
}

__mmask8 test_mm_mask_cmp_pbh_mask_gt_os(__mmask8 m, __m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_mask_cmp_pbh_mask_gt_os
  // CHECK: fcmp ogt <8 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return _mm_mask_cmp_pbh_mask(m, a, b, _CMP_GT_OS);
}

__mmask8 test_mm_mask_cmp_pbh_mask_true_uq(__mmask8 m, __m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_mask_cmp_pbh_mask_true_uq
  // CHECK: fcmp true <8 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return _mm_mask_cmp_pbh_mask(m, a, b, _CMP_TRUE_UQ);
}

__mmask8 test_mm_mask_cmp_pbh_mask_eq_os(__mmask8 m, __m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_mask_cmp_pbh_mask_eq_os
  // CHECK: fcmp oeq <8 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return _mm_mask_cmp_pbh_mask(m, a, b, _CMP_EQ_OS);
}

__mmask8 test_mm_mask_cmp_pbh_mask_lt_oq(__mmask8 m, __m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_mask_cmp_pbh_mask_lt_oq
  // CHECK: fcmp olt <8 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return _mm_mask_cmp_pbh_mask(m, a, b, _CMP_LT_OQ);
}

__mmask8 test_mm_mask_cmp_pbh_mask_le_oq(__mmask8 m, __m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_mask_cmp_pbh_mask_le_oq
  // CHECK: fcmp ole <8 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return _mm_mask_cmp_pbh_mask(m, a, b, _CMP_LE_OQ);
}

__mmask8 test_mm_mask_cmp_pbh_mask_unord_s(__mmask8 m, __m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_mask_cmp_pbh_mask_unord_s
  // CHECK: fcmp uno <8 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return _mm_mask_cmp_pbh_mask(m, a, b, _CMP_UNORD_S);
}

__mmask8 test_mm_mask_cmp_pbh_mask_neq_us(__mmask8 m, __m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_mask_cmp_pbh_mask_neq_us
  // CHECK: fcmp une <8 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return _mm_mask_cmp_pbh_mask(m, a, b, _CMP_NEQ_US);
}

__mmask8 test_mm_mask_cmp_pbh_mask_nlt_uq(__mmask8 m, __m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_mask_cmp_pbh_mask_nlt_uq
  // CHECK: fcmp uge <8 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return _mm_mask_cmp_pbh_mask(m, a, b, _CMP_NLT_UQ);
}

__mmask8 test_mm_mask_cmp_pbh_mask_nle_uq(__mmask8 m, __m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_mask_cmp_pbh_mask_nle_uq
  // CHECK: fcmp ugt <8 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return _mm_mask_cmp_pbh_mask(m, a, b, _CMP_NLE_UQ);
}

__mmask8 test_mm_mask_cmp_pbh_mask_ord_s(__mmask8 m, __m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_mask_cmp_pbh_mask_ord_s
  // CHECK: fcmp ord <8 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return _mm_mask_cmp_pbh_mask(m, a, b, _CMP_ORD_S);
}

__mmask8 test_mm_mask_cmp_pbh_mask_eq_us(__mmask8 m, __m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_mask_cmp_pbh_mask_eq_us
  // CHECK: fcmp ueq <8 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return _mm_mask_cmp_pbh_mask(m, a, b, _CMP_EQ_US);
}

__mmask8 test_mm_mask_cmp_pbh_mask_nge_uq(__mmask8 m, __m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_mask_cmp_pbh_mask_nge_uq
  // CHECK: fcmp ult <8 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return _mm_mask_cmp_pbh_mask(m, a, b, _CMP_NGE_UQ);
}

__mmask8 test_mm_mask_cmp_pbh_mask_ngt_uq(__mmask8 m, __m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_mask_cmp_pbh_mask_ngt_uq
  // CHECK: fcmp ule <8 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return _mm_mask_cmp_pbh_mask(m, a, b, _CMP_NGT_UQ);
}

__mmask8 test_mm_mask_cmp_pbh_mask_false_os(__mmask8 m, __m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_mask_cmp_pbh_mask_false_os
  // CHECK: fcmp false <8 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return _mm_mask_cmp_pbh_mask(m, a, b, _CMP_FALSE_OS);
}

__mmask8 test_mm_mask_cmp_pbh_mask_neq_os(__mmask8 m, __m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_mask_cmp_pbh_mask_neq_os
  // CHECK: fcmp one <8 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return _mm_mask_cmp_pbh_mask(m, a, b, _CMP_NEQ_OS);
}

__mmask8 test_mm_mask_cmp_pbh_mask_ge_oq(__mmask8 m, __m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_mask_cmp_pbh_mask_ge_oq
  // CHECK: fcmp oge <8 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return _mm_mask_cmp_pbh_mask(m, a, b, _CMP_GE_OQ);
}

__mmask8 test_mm_mask_cmp_pbh_mask_gt_oq(__mmask8 m, __m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_mask_cmp_pbh_mask_gt_oq
  // CHECK: fcmp ogt <8 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return _mm_mask_cmp_pbh_mask(m, a, b, _CMP_GT_OQ);
}

__mmask8 test_mm_mask_cmp_pbh_mask_true_us(__mmask8 m, __m128bh a, __m128bh b) {
  // CHECK-LABEL: test_mm_mask_cmp_pbh_mask_true_us
  // CHECK: fcmp true <8 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return _mm_mask_cmp_pbh_mask(m, a, b, _CMP_TRUE_US);
}


__mmask16 test_mm256_mask_fpclass_pbh_mask(__mmask16 __U, __m256bh __A) {
  // CHECK-LABEL: @test_mm256_mask_fpclass_pbh_mask
  // CHECK: @llvm.x86.avx10.fpclass.nepbf16.256
  return _mm256_mask_fpclass_pbh_mask(__U, __A, 4);
}

__mmask16 test_mm256_fpclass_pbh_mask(__m256bh __A) {
  // CHECK-LABEL: @test_mm256_fpclass_pbh_mask
  // CHECK: @llvm.x86.avx10.fpclass.nepbf16.256
  return _mm256_fpclass_pbh_mask(__A, 4);
}

__mmask8 test_mm_mask_fpclass_pbh_mask(__mmask8 __U, __m128bh __A) {
  // CHECK-LABEL: @test_mm_mask_fpclass_pbh_mask
  // CHECK: @llvm.x86.avx10.fpclass.nepbf16.128
  return _mm_mask_fpclass_pbh_mask(__U, __A, 4);
}

__mmask8 test_mm_fpclass_pbh_mask(__m128bh __A) {
  // CHECK-LABEL: @test_mm_fpclass_pbh_mask
  // CHECK: @llvm.x86.avx10.fpclass.nepbf16.128
  return _mm_fpclass_pbh_mask(__A, 4);
}

__m256bh test_mm256_scalef_pbh(__m256bh __A, __m256bh __B) {
  // CHECK-LABEL: @test_mm256_scalef_pbh
  // CHECK: @llvm.x86.avx10.mask.scalef.nepbf16.256
  return _mm256_scalef_pbh(__A, __B);
}

__m256bh test_mm256_mask_scalef_pbh(__m256bh __W, __mmask16 __U, __m256bh __A, __m256bh __B) {
  // CHECK-LABEL: @test_mm256_mask_scalef_pbh
  // CHECK: @llvm.x86.avx10.mask.scalef.nepbf16.256
  return _mm256_mask_scalef_pbh(__W, __U, __A, __B);
}

__m256bh test_mm256_maskz_scalef_pbh(__mmask16 __U, __m256bh __A, __m256bh __B) {
  // CHECK-LABEL: @test_mm256_maskz_scalef_pbh
  // CHECK: @llvm.x86.avx10.mask.scalef.nepbf16.256
  return _mm256_maskz_scalef_pbh(__U, __A, __B);
}

__m256bh test_mm256_rcp_pbh(__m256bh __A) {
  // CHECK-LABEL: @test_mm256_rcp_pbh
  // CHECK: @llvm.x86.avx10.mask.rcp.nepbf16.256
  return _mm256_rcp_pbh(__A);
}

__m256bh test_mm256_mask_rcp_pbh(__m256bh __W, __mmask16 __U, __m256bh __A) {
  // CHECK-LABEL: @test_mm256_mask_rcp_pbh
  // CHECK: @llvm.x86.avx10.mask.rcp.nepbf16.256
  return (__m256bh)_mm256_mask_rcp_pbh(__W, __U, __A);
}

__m256bh test_mm256_maskz_rcp_pbh(__mmask16 __U, __m256bh __A) {
  // CHECK-LABEL: @test_mm256_maskz_rcp_pbh
  // CHECK: @llvm.x86.avx10.mask.rcp.nepbf16.256
  return _mm256_maskz_rcp_pbh(__U, __A);
}

__m256bh test_mm256_getexp_pbh(__m256bh __A) {
  // CHECK-LABEL: @test_mm256_getexp_pbh
  // CHECK: @llvm.x86.avx10.mask.getexp.nepbf16.256
  return _mm256_getexp_pbh(__A);
}

__m256bh test_mm256_mask_getexp_pbh(__m256bh __W, __mmask16 __U, __m256bh __A) {
  // CHECK-LABEL: @test_mm256_mask_getexp_pbh
  // CHECK: @llvm.x86.avx10.mask.getexp.nepbf16.256
  return _mm256_mask_getexp_pbh(__W, __U, __A);
}

__m256bh test_mm256_maskz_getexp_pbh(__mmask16 __U, __m256bh __A) {
  // CHECK-LABEL: @test_mm256_maskz_getexp_pbh
  // CHECK: @llvm.x86.avx10.mask.getexp.nepbf16.256
  return _mm256_maskz_getexp_pbh(__U, __A);
}

__m256bh test_mm256_rsqrt_pbh(__m256bh __A) {
  // CHECK-LABEL: @test_mm256_rsqrt_pbh
  // CHECK: @llvm.x86.avx10.mask.rsqrt.nepbf16.256
  return _mm256_rsqrt_pbh(__A);
}

__m256bh test_mm256_mask_rsqrt_pbh(__m256bh __W, __mmask16 __U, __m256bh __A) {
  // CHECK-LABEL: @test_mm256_mask_rsqrt_pbh
  // CHECK: @llvm.x86.avx10.mask.rsqrt.nepbf16.256
  return (__m256bh)_mm256_mask_rsqrt_pbh(__W, __U, __A);
}

__m256bh test_mm256_maskz_rsqrt_pbh(__mmask16 __U, __m256bh __A) {
  // CHECK-LABEL: @test_mm256_maskz_rsqrt_pbh
  // CHECK: @llvm.x86.avx10.mask.rsqrt.nepbf16.256
  return _mm256_maskz_rsqrt_pbh(__U, __A);
}

__m256bh test_mm256_reducene_pbh(__m256bh __A) {
  // CHECK-LABEL: @test_mm256_reducene_pbh
  // CHECK: @llvm.x86.avx10.mask.reduce.nepbf16.256
  return _mm256_reducene_pbh(__A, 3);
}

__m256bh test_mm256_mask_reducene_pbh(__m256bh __W, __mmask16 __U, __m256bh __A) {
  // CHECK-LABEL: @test_mm256_mask_reducene_pbh
  // CHECK: @llvm.x86.avx10.mask.reduce.nepbf16.256
  return _mm256_mask_reducene_pbh(__W, __U, __A, 1);
}

__m256bh test_mm256_maskz_reducene_pbh(__mmask16 __U, __m256bh __A) {
  // CHECK-LABEL: @test_mm256_maskz_reducene_pbh
  // CHECK: @llvm.x86.avx10.mask.reduce.nepbf16.256
  return _mm256_maskz_reducene_pbh(__U, __A, 1);
}

__m256bh test_mm256_roundscalene_pbh(__m256bh __A) {
  // CHECK-LABEL: @test_mm256_roundscalene_pbh
  // CHECK: @llvm.x86.avx10.mask.rndscale.nepbf16.256
  return _mm256_roundscalene_pbh(__A, 3);
}

__m256bh test_mm256_mask_roundscalene_pbh(__m256bh __W, __mmask16 __U, __m256bh __A) {
  // CHECK-LABEL: @test_mm256_mask_roundscalene_pbh
  // CHECK: @llvm.x86.avx10.mask.rndscale.nepbf16.256
  return _mm256_mask_roundscalene_pbh(__W, __U, __A, 1);
}

__m256bh test_mm256_maskz_roundscalene_pbh(__mmask16 __U, __m256bh __A) {
  // CHECK-LABEL: @test_mm256_maskz_roundscalene_pbh
  // CHECK: @llvm.x86.avx10.mask.rndscale.nepbf16.256
  return _mm256_maskz_roundscalene_pbh(__U, __A, 1 );
}

__m256bh test_mm256_getmant_pbh(__m256bh __A) {
  // CHECK-LABEL: @test_mm256_getmant_pbh
  // CHECK: @llvm.x86.avx10.mask.getmant.nepbf16.256
  return _mm256_getmant_pbh(__A, _MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan);
}

__m256bh test_mm256_mask_getmant_pbh(__m256bh __W, __mmask16 __U, __m256bh __A) {
  // CHECK-LABEL: @test_mm256_mask_getmant_pbh
  // CHECK: @llvm.x86.avx10.mask.getmant.nepbf16.256
  return _mm256_mask_getmant_pbh(__W, __U, __A, _MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan);
}

__m256bh test_mm256_maskz_getmant_pbh(__mmask16 __U, __m256bh __A) {
  // CHECK-LABEL: @test_mm256_maskz_getmant_pbh
  // CHECK: @llvm.x86.avx10.mask.getmant.nepbf16.256
  return _mm256_maskz_getmant_pbh(__U, __A, _MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan);
}

__m256bh test_mm256_sqrt_pbh(__m256bh __A) {
  // CHECK-LABEL: @test_mm256_sqrt_pbh
  // CHECK: call <16 x bfloat> @llvm.sqrt.v16bf16(<16 x bfloat> %{{.*}})
  return _mm256_sqrt_pbh(__A);
}

__m256bh test_mm256_mask_sqrt_pbh(__m256bh __W, __mmask16 __U, __m256bh __A) {
  // CHECK-LABEL: @test_mm256_mask_sqrt_pbh
  // CHECK: @llvm.sqrt.v16bf16
  // CHECK: select <16 x i1> %{{.*}}, <16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}}
  return (__m256bh)_mm256_mask_sqrt_pbh(__W, __U, __A);
}

__m256bh test_mm256_maskz_sqrt_pbh(__mmask16 __U, __m256bh __A) {
  // CHECK-LABEL: @test_mm256_maskz_sqrt_pbh
  // CHECK: @llvm.sqrt.v16bf16
  // CHECK: select <16 x i1> %{{.*}}, <16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}}
  return _mm256_maskz_sqrt_pbh(__U, __A);
}

__m128bh test_mm_scalef_pbh(__m128bh __A, __m128bh __B) {
  // CHECK-LABEL: @test_mm_scalef_pbh
  // CHECK: @llvm.x86.avx10.mask.scalef.nepbf16.128
  return _mm_scalef_pbh(__A, __B);
}

__m128bh test_mm_mask_scalef_pbh(__m128bh __W, __mmask8 __U, __m128bh __A, __m128bh __B) {
  // CHECK-LABEL: @test_mm_mask_scalef_pbh
  // CHECK: @llvm.x86.avx10.mask.scalef.nepbf16.128
  return _mm_mask_scalef_pbh(__W, __U, __A, __B);
}

__m128bh test_mm_maskz_scalef_pbh(__mmask8 __U, __m128bh __A, __m128bh __B) {
  // CHECK-LABEL: @test_mm_maskz_scalef_pbh
  // CHECK: @llvm.x86.avx10.mask.scalef.nepbf16.128
  return _mm_maskz_scalef_pbh(__U, __A, __B);
}

__m128bh test_mm_rcp_pbh(__m128bh __A) {
  // CHECK-LABEL: @test_mm_rcp_pbh
  // CHECK: @llvm.x86.avx10.mask.rcp.nepbf16.128
  return _mm_rcp_pbh(__A);
}

__m128bh test_mm_mask_rcp_pbh(__m128bh __W, __mmask8 __U, __m128bh __A) {
  // CHECK-LABEL: @test_mm_mask_rcp_pbh
  // CHECK: @llvm.x86.avx10.mask.rcp.nepbf16.128
  return (__m128bh)_mm_mask_rcp_pbh(__W, __U, __A);
}

__m128bh test_mm_maskz_rcp_pbh(__mmask8 __U, __m128bh __A) {
  // CHECK-LABEL: @test_mm_maskz_rcp_pbh
  // CHECK: @llvm.x86.avx10.mask.rcp.nepbf16.128
  return _mm_maskz_rcp_pbh(__U, __A);
}

__m128bh test_mm_getexp_pbh(__m128bh __A) {
  // CHECK-LABEL: @test_mm_getexp_pbh
  // CHECK: @llvm.x86.avx10.mask.getexp.nepbf16.128
  return _mm_getexp_pbh(__A);
}

__m128bh test_mm_mask_getexp_pbh(__m128bh __W, __mmask8 __U, __m128bh __A) {
  // CHECK-LABEL: @test_mm_mask_getexp_pbh
  // CHECK: @llvm.x86.avx10.mask.getexp.nepbf16.128
  return _mm_mask_getexp_pbh(__W, __U, __A);
}

__m128bh test_mm_maskz_getexp_pbh(__mmask8 __U, __m128bh __A) {
  // CHECK-LABEL: @test_mm_maskz_getexp_pbh
  // CHECK: @llvm.x86.avx10.mask.getexp.nepbf16.128
  return _mm_maskz_getexp_pbh(__U, __A);
}

__m128bh test_mm_rsqrt_pbh(__m128bh __A) {
  // CHECK-LABEL: @test_mm_rsqrt_pbh
  // CHECK: @llvm.x86.avx10.mask.rsqrt.nepbf16.128
  return _mm_rsqrt_pbh(__A);
}

__m128bh test_mm_mask_rsqrt_pbh(__m128bh __W, __mmask8 __U, __m128bh __A) {
  // CHECK-LABEL: @test_mm_mask_rsqrt_pbh
  // CHECK: @llvm.x86.avx10.mask.rsqrt.nepbf16.128
  return (__m128bh)_mm_mask_rsqrt_pbh(__W, __U, __A);
}

__m128bh test_mm_maskz_rsqrt_pbh(__mmask8 __U, __m128bh __A) {
  // CHECK-LABEL: @test_mm_maskz_rsqrt_pbh
  // CHECK: @llvm.x86.avx10.mask.rsqrt.nepbf16.128
  return _mm_maskz_rsqrt_pbh(__U, __A);
}

__m128bh test_mm_reducene_pbh(__m128bh __A) {
  // CHECK-LABEL: @test_mm_reducene_pbh
  // CHECK: @llvm.x86.avx10.mask.reduce.nepbf16.128
  return _mm_reducene_pbh(__A, 3);
}

__m128bh test_mm_mask_reducene_pbh(__m128bh __W, __mmask8 __U, __m128bh __A) {
  // CHECK-LABEL: @test_mm_mask_reducene_pbh
  // CHECK: @llvm.x86.avx10.mask.reduce.nepbf16.128
  return _mm_mask_reducene_pbh(__W, __U, __A, 1);
}

__m128bh test_mm_maskz_reducene_pbh(__mmask8 __U, __m128bh __A) {
  // CHECK-LABEL: @test_mm_maskz_reducene_pbh
  // CHECK: @llvm.x86.avx10.mask.reduce.nepbf16.128
  return _mm_maskz_reducene_pbh(__U, __A, 1);
}

__m128bh test_mm_roundscalene_pbh(__m128bh __A) {
  // CHECK-LABEL: @test_mm_roundscalene_pbh
  // CHECK: @llvm.x86.avx10.mask.rndscale.nepbf16.128
  return _mm_roundscalene_pbh(__A, 3);
}

__m128bh test_mm_mask_roundscalene_pbh(__m128bh __W, __mmask8 __U, __m128bh __A) {
  // CHECK-LABEL: @test_mm_mask_roundscalene_pbh
  // CHECK: @llvm.x86.avx10.mask.rndscale.nepbf16.128
  return _mm_mask_roundscalene_pbh(__W, __U, __A, 1);
}

__m128bh test_mm_maskz_roundscalene_pbh(__mmask8 __U, __m128bh __A) {
  // CHECK-LABEL: @test_mm_maskz_roundscalene_pbh
  // CHECK: @llvm.x86.avx10.mask.rndscale.nepbf16.128
  return _mm_maskz_roundscalene_pbh(__U, __A, 1 );
}

__m128bh test_mm_getmant_pbh(__m128bh __A) {
  // CHECK-LABEL: @test_mm_getmant_pbh
  // CHECK: @llvm.x86.avx10.mask.getmant.nepbf16.128
  return _mm_getmant_pbh(__A, _MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan);
}

__m128bh test_mm_mask_getmant_pbh(__m128bh __W, __mmask8 __U, __m128bh __A) {
  // CHECK-LABEL: @test_mm_mask_getmant_pbh
  // CHECK: @llvm.x86.avx10.mask.getmant.nepbf16.128
  return _mm_mask_getmant_pbh(__W, __U, __A, _MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan);
}

__m128bh test_mm_maskz_getmant_pbh(__mmask8 __U, __m128bh __A) {
  // CHECK-LABEL: @test_mm_maskz_getmant_pbh
  // CHECK: @llvm.x86.avx10.mask.getmant.nepbf16.128
  return _mm_maskz_getmant_pbh(__U, __A, _MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan);
}

__m128bh test_mm_sqrt_pbh(__m128bh __A) {
  // CHECK-LABEL: @test_mm_sqrt_pbh
  // CHECK: call <8 x bfloat> @llvm.sqrt.v8bf16(<8 x bfloat> {{.*}})
  return _mm_sqrt_pbh(__A);
}

__m128bh test_mm_mask_sqrt_pbh(__m128bh __W, __mmask8 __U, __m128bh __A) {
  // CHECK-LABEL: @test_mm_mask_sqrt_pbh
  // CHECK: call <8 x bfloat> @llvm.sqrt.v8bf16(<8 x bfloat> {{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}}
  return (__m128bh)_mm_mask_sqrt_pbh(__W, __U, __A);
}

__m128bh test_mm_maskz_sqrt_pbh(__mmask8 __U, __m128bh __A) {
  // CHECK-LABEL: @test_mm_maskz_sqrt_pbh
  // CHECK: call <8 x bfloat> @llvm.sqrt.v8bf16(<8 x bfloat> {{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}}
  return _mm_maskz_sqrt_pbh(__U, __A);
}

__m256bh test_mm256_fmaddne_pbh(__m256bh __A, __m256bh __B, __m256bh __C) {
  // CHECK-LABEL: @test_mm256_fmaddne_pbh
  // CHECK: call <16 x bfloat> @llvm.fma.v16bf16(<16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}})
  return _mm256_fmaddne_pbh(__A, __B, __C);
}

__m256bh test_mm256_mask_fmaddne_pbh(__m256bh __A, __mmask16 __U, __m256bh __B, __m256bh __C) {
  // CHECK-LABEL: @test_mm256_mask_fmaddne_pbh
  // CHECK: call <16 x bfloat> @llvm.fma.v16bf16(<16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}}
  return _mm256_mask_fmaddne_pbh(__A, __U, __B, __C);
}

__m256bh test_mm256_mask3_fmaddne_pbh(__m256bh __A, __m256bh __B, __m256bh __C, __mmask16 __U) {
  // CHECK-LABEL: @test_mm256_mask3_fmaddne_pbh
  // CHECK: call <16 x bfloat> @llvm.fma.v16bf16(<16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}}
  return _mm256_mask3_fmaddne_pbh(__A, __B, __C, __U);
}

__m256bh test_mm256_maskz_fmaddne_pbh(__mmask16 __U, __m256bh __A, __m256bh __B, __m256bh __C) {
  // CHECK-LABEL: @test_mm256_maskz_fmaddne_pbh
  // CHECK: call <16 x bfloat> @llvm.fma.v16bf16(<16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}}
  return _mm256_maskz_fmaddne_pbh(__U, __A, __B, __C);
}

__m256bh test_mm256_fmsubne_pbh(__m256bh __A, __m256bh __B, __m256bh __C) {
  // CHECK-LABEL: @test_mm256_fmsubne_pbh
  // CHECK: fneg
  // CHECK: call <16 x bfloat> @llvm.fma.v16bf16(<16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}})
  return _mm256_fmsubne_pbh(__A, __B, __C);
}

__m256bh test_mm256_mask_fmsubne_pbh(__m256bh __A, __mmask16 __U, __m256bh __B, __m256bh __C) {
  // CHECK-LABEL: @test_mm256_mask_fmsubne_pbh
  // CHECK: fneg
  // CHECK: call <16 x bfloat> @llvm.fma.v16bf16(<16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}}
  return _mm256_mask_fmsubne_pbh(__A, __U, __B, __C);
}

__m256bh test_mm256_mask3_fmsubne_pbh(__m256bh __A, __m256bh __B, __m256bh __C, __mmask16 __U) {
  // CHECK-LABEL: @test_mm256_mask3_fmsubne_pbh
  // CHECK: fneg
  // CHECK: call <16 x bfloat> @llvm.fma.v16bf16(<16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}}
  return _mm256_mask3_fmsubne_pbh(__A, __B, __C, __U);
}

__m256bh test_mm256_maskz_fmsubne_pbh(__mmask16 __U, __m256bh __A, __m256bh __B, __m256bh __C) {
  // CHECK-LABEL: @test_mm256_maskz_fmsubne_pbh
  // CHECK: fneg
  // CHECK: call <16 x bfloat> @llvm.fma.v16bf16(<16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}}
  return _mm256_maskz_fmsubne_pbh(__U, __A, __B, __C);
}

__m256bh test_mm256_fnmaddne_pbh(__m256bh __A, __m256bh __B, __m256bh __C) {
  // CHECK-LABEL: @test_mm256_fnmaddne_pbh
  // CHECK: fneg
  // CHECK: call <16 x bfloat> @llvm.fma.v16bf16(<16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}})
  return _mm256_fnmaddne_pbh(__A, __B, __C);
}

__m256bh test_mm256_mask_fnmaddne_pbh(__m256bh __A, __mmask16 __U, __m256bh __B, __m256bh __C) {
  // CHECK-LABEL: @test_mm256_mask_fnmaddne_pbh
  // CHECK: fneg
  // CHECK: call <16 x bfloat> @llvm.fma.v16bf16(<16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}}
  return _mm256_mask_fnmaddne_pbh(__A, __U, __B, __C);
}

__m256bh test_mm256_mask3_fnmaddne_pbh(__m256bh __A, __m256bh __B, __m256bh __C, __mmask16 __U) {
  // CHECK-LABEL: @test_mm256_mask3_fnmaddne_pbh
  // CHECK: fneg
  // CHECK: call <16 x bfloat> @llvm.fma.v16bf16(<16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}}
  return _mm256_mask3_fnmaddne_pbh(__A, __B, __C, __U);
}

__m256bh test_mm256_maskz_fnmaddne_pbh(__mmask16 __U, __m256bh __A, __m256bh __B, __m256bh __C) {
  // CHECK-LABEL: @test_mm256_maskz_fnmaddne_pbh
  // CHECK: fneg
  // CHECK: call <16 x bfloat> @llvm.fma.v16bf16(<16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}}
  return _mm256_maskz_fnmaddne_pbh(__U, __A, __B, __C);
}

__m256bh test_mm256_fnmsubne_pbh(__m256bh __A, __m256bh __B, __m256bh __C) {
  // CHECK-LABEL: @test_mm256_fnmsubne_pbh
  // CHECK: fneg
  // CHECK: fneg
  // CHECK: call <16 x bfloat> @llvm.fma.v16bf16(<16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}})
  return _mm256_fnmsubne_pbh(__A, __B, __C);
}

__m256bh test_mm256_mask_fnmsubne_pbh(__m256bh __A, __mmask16 __U, __m256bh __B, __m256bh __C) {
  // CHECK-LABEL: @test_mm256_mask_fnmsubne_pbh
  // CHECK: fneg
  // CHECK: fneg
  // CHECK: call <16 x bfloat> @llvm.fma.v16bf16(<16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}}
  return _mm256_mask_fnmsubne_pbh(__A, __U, __B, __C);
}

__m256bh test_mm256_mask3_fnmsubne_pbh(__m256bh __A, __m256bh __B, __m256bh __C, __mmask16 __U) {
  // CHECK-LABEL: @test_mm256_mask3_fnmsubne_pbh
  // CHECK: fneg
  // CHECK: fneg
  // CHECK: call <16 x bfloat> @llvm.fma.v16bf16(<16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}}
  return _mm256_mask3_fnmsubne_pbh(__A, __B, __C, __U);
}

__m256bh test_mm256_maskz_fnmsubne_pbh(__mmask16 __U, __m256bh __A, __m256bh __B, __m256bh __C) {
  // CHECK-LABEL: @test_mm256_maskz_fnmsubne_pbh
  // CHECK: fneg
  // CHECK: fneg
  // CHECK: call <16 x bfloat> @llvm.fma.v16bf16(<16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}}
  return _mm256_maskz_fnmsubne_pbh(__U, __A, __B, __C);
}

__m128bh test_mm_fmaddne_pbh(__m128bh __A, __m128bh __B, __m128bh __C) {
  // CHECK-LABEL: @test_mm_fmaddne_pbh
  // CHECK: call <8 x bfloat> @llvm.fma.v8bf16(<8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}})
  return _mm_fmaddne_pbh(__A, __B, __C);
}

__m128bh test_mm_mask_fmaddne_pbh(__m128bh __A, __mmask8 __U, __m128bh __B, __m128bh __C) {
  // CHECK-LABEL: @test_mm_mask_fmaddne_pbh
  // CHECK: call <8 x bfloat> @llvm.fma.v8bf16(<8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}}
  return _mm_mask_fmaddne_pbh(__A, __U, __B, __C);
}

__m128bh test_mm_mask3_fmaddne_pbh(__m128bh __A, __m128bh __B, __m128bh __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm_mask3_fmaddne_pbh
  // CHECK: call <8 x bfloat> @llvm.fma.v8bf16(<8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}}
  return _mm_mask3_fmaddne_pbh(__A, __B, __C, __U);
}

__m128bh test_mm_maskz_fmaddne_pbh(__mmask8 __U, __m128bh __A, __m128bh __B, __m128bh __C) {
  // CHECK-LABEL: @test_mm_maskz_fmaddne_pbh
  // CHECK: call <8 x bfloat> @llvm.fma.v8bf16(<8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}}
  return _mm_maskz_fmaddne_pbh(__U, __A, __B, __C);
}

__m128bh test_mm_fmsubne_pbh(__m128bh __A, __m128bh __B, __m128bh __C) {
  // CHECK-LABEL: @test_mm_fmsubne_pbh
  // CHECK: fneg
  // CHECK: call <8 x bfloat> @llvm.fma.v8bf16(<8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}})
  return _mm_fmsubne_pbh(__A, __B, __C);
}

__m128bh test_mm_mask_fmsubne_pbh(__m128bh __A, __mmask8 __U, __m128bh __B, __m128bh __C) {
  // CHECK-LABEL: @test_mm_mask_fmsubne_pbh
  // CHECK: fneg
  // CHECK: call <8 x bfloat> @llvm.fma.v8bf16(<8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}}
  return _mm_mask_fmsubne_pbh(__A, __U, __B, __C);
}

__m128bh test_mm_mask3_fmsubne_pbh(__m128bh __A, __m128bh __B, __m128bh __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm_mask3_fmsubne_pbh
  // CHECK: fneg
  // CHECK: call <8 x bfloat> @llvm.fma.v8bf16(<8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}}
  return _mm_mask3_fmsubne_pbh(__A, __B, __C, __U);
}

__m128bh test_mm_maskz_fmsubne_pbh(__mmask8 __U, __m128bh __A, __m128bh __B, __m128bh __C) {
  // CHECK-LABEL: @test_mm_maskz_fmsubne_pbh
  // CHECK: fneg
  // CHECK: call <8 x bfloat> @llvm.fma.v8bf16(<8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}}
  return _mm_maskz_fmsubne_pbh(__U, __A, __B, __C);
}

__m128bh test_mm_fnmaddne_pbh(__m128bh __A, __m128bh __B, __m128bh __C) {
  // CHECK-LABEL: @test_mm_fnmaddne_pbh
  // CHECK: fneg
  // CHECK: call <8 x bfloat> @llvm.fma.v8bf16(<8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}})
  return _mm_fnmaddne_pbh(__A, __B, __C);
}

__m128bh test_mm_mask_fnmaddne_pbh(__m128bh __A, __mmask8 __U, __m128bh __B, __m128bh __C) {
  // CHECK-LABEL: @test_mm_mask_fnmaddne_pbh
  // CHECK: fneg
  // CHECK: call <8 x bfloat> @llvm.fma.v8bf16(<8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}}
  return _mm_mask_fnmaddne_pbh(__A, __U, __B, __C);
}

__m128bh test_mm_mask3_fnmaddne_pbh(__m128bh __A, __m128bh __B, __m128bh __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm_mask3_fnmaddne_pbh
  // CHECK: fneg
  // CHECK: call <8 x bfloat> @llvm.fma.v8bf16(<8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}}
  return _mm_mask3_fnmaddne_pbh(__A, __B, __C, __U);
}

__m128bh test_mm_maskz_fnmaddne_pbh(__mmask8 __U, __m128bh __A, __m128bh __B, __m128bh __C) {
  // CHECK-LABEL: @test_mm_maskz_fnmaddne_pbh
  // CHECK: fneg
  // CHECK: call <8 x bfloat> @llvm.fma.v8bf16(<8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}}
  return _mm_maskz_fnmaddne_pbh(__U, __A, __B, __C);
}

__m128bh test_mm_fnmsubne_pbh(__m128bh __A, __m128bh __B, __m128bh __C) {
  // CHECK-LABEL: @test_mm_fnmsubne_pbh
  // CHECK: fneg
  // CHECK: fneg
  // CHECK: call <8 x bfloat> @llvm.fma.v8bf16(<8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}})
  return _mm_fnmsubne_pbh(__A, __B, __C);
}

__m128bh test_mm_mask_fnmsubne_pbh(__m128bh __A, __mmask8 __U, __m128bh __B, __m128bh __C) {
  // CHECK-LABEL: @test_mm_mask_fnmsubne_pbh
  // CHECK: fneg
  // CHECK: fneg
  // CHECK: call <8 x bfloat> @llvm.fma.v8bf16(<8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}}
  return _mm_mask_fnmsubne_pbh(__A, __U, __B, __C);
}

__m128bh test_mm_mask3_fnmsubne_pbh(__m128bh __A, __m128bh __B, __m128bh __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm_mask3_fnmsubne_pbh
  // CHECK: fneg
  // CHECK: fneg
  // CHECK: call <8 x bfloat> @llvm.fma.v8bf16(<8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}}
  return _mm_mask3_fnmsubne_pbh(__A, __B, __C, __U);
}

__m128bh test_mm_maskz_fnmsubne_pbh(__mmask8 __U, __m128bh __A, __m128bh __B, __m128bh __C) {
  // CHECK-LABEL: @test_mm_maskz_fnmsubne_pbh
  // CHECK: fneg
  // CHECK: fneg
  // CHECK: call <8 x bfloat> @llvm.fma.v8bf16(<8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}}
  return _mm_maskz_fnmsubne_pbh(__U, __A, __B, __C);
}

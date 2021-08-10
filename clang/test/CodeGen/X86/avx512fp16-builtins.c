// RUN: %clang_cc1 -ffreestanding -flax-vector-conversions=none %s -triple=x86_64-unknown-unknown -target-feature +avx512fp16 -emit-llvm -o - -Wall -Werror | FileCheck %s

#include <immintrin.h>

_Float16 test_mm512_cvtsh_h(__m512h __A) {
  // CHECK-LABEL: @test_mm512_cvtsh_h
  // CHECK: extractelement <32 x half> %{{.*}}, i32 0
  return _mm512_cvtsh_h(__A);
}

__m128h test_mm_setzero_ph() {
  // CHECK-LABEL: @test_mm_setzero_ph
  // CHECK: zeroinitializer
  return _mm_setzero_ph();
}

__m256h test_mm256_setzero_ph() {
  // CHECK-LABEL: @test_mm256_setzero_ph
  // CHECK: zeroinitializer
  return _mm256_setzero_ph();
}

__m256h test_mm256_undefined_ph() {
  // CHECK-LABEL: @test_mm256_undefined_ph
  // CHECK: ret <16 x half> zeroinitializer
  return _mm256_undefined_ph();
}

__m512h test_mm512_setzero_ph() {
  // CHECK-LABEL: @test_mm512_setzero_ph
  // CHECK: zeroinitializer
  return _mm512_setzero_ph();
}

__m128h test_mm_undefined_ph() {
  // CHECK-LABEL: @test_mm_undefined_ph
  // CHECK: ret <8 x half> zeroinitializer
  return _mm_undefined_ph();
}

__m512h test_mm512_undefined_ph() {
  // CHECK-LABEL: @test_mm512_undefined_ph
  // CHECK: ret <32 x half> zeroinitializer
  return _mm512_undefined_ph();
}

__m512h test_mm512_set1_ph(_Float16 h) {
  // CHECK-LABEL: @test_mm512_set1_ph
  // CHECK: insertelement <32 x half> {{.*}}, i32 0
  // CHECK: insertelement <32 x half> {{.*}}, i32 1
  // CHECK: insertelement <32 x half> {{.*}}, i32 2
  // CHECK: insertelement <32 x half> {{.*}}, i32 3
  // CHECK: insertelement <32 x half> {{.*}}, i32 4
  // CHECK: insertelement <32 x half> {{.*}}, i32 5
  // CHECK: insertelement <32 x half> {{.*}}, i32 6
  // CHECK: insertelement <32 x half> {{.*}}, i32 7
  // CHECK: insertelement <32 x half> {{.*}}, i32 8
  // CHECK: insertelement <32 x half> {{.*}}, i32 9
  // CHECK: insertelement <32 x half> {{.*}}, i32 10
  // CHECK: insertelement <32 x half> {{.*}}, i32 11
  // CHECK: insertelement <32 x half> {{.*}}, i32 12
  // CHECK: insertelement <32 x half> {{.*}}, i32 13
  // CHECK: insertelement <32 x half> {{.*}}, i32 14
  // CHECK: insertelement <32 x half> {{.*}}, i32 15
  // CHECK: insertelement <32 x half> {{.*}}, i32 16
  // CHECK: insertelement <32 x half> {{.*}}, i32 17
  // CHECK: insertelement <32 x half> {{.*}}, i32 18
  // CHECK: insertelement <32 x half> {{.*}}, i32 19
  // CHECK: insertelement <32 x half> {{.*}}, i32 20
  // CHECK: insertelement <32 x half> {{.*}}, i32 21
  // CHECK: insertelement <32 x half> {{.*}}, i32 22
  // CHECK: insertelement <32 x half> {{.*}}, i32 23
  // CHECK: insertelement <32 x half> {{.*}}, i32 24
  // CHECK: insertelement <32 x half> {{.*}}, i32 25
  // CHECK: insertelement <32 x half> {{.*}}, i32 26
  // CHECK: insertelement <32 x half> {{.*}}, i32 27
  // CHECK: insertelement <32 x half> {{.*}}, i32 28
  // CHECK: insertelement <32 x half> {{.*}}, i32 29
  // CHECK: insertelement <32 x half> {{.*}}, i32 30
  // CHECK: insertelement <32 x half> {{.*}}, i32 31
  return _mm512_set1_ph(h);
}

__m512h test_mm512_set_ph(_Float16 __h1, _Float16 __h2, _Float16 __h3, _Float16 __h4,
                          _Float16 __h5, _Float16 __h6, _Float16 __h7, _Float16 __h8,
                          _Float16 __h9, _Float16 __h10, _Float16 __h11, _Float16 __h12,
                          _Float16 __h13, _Float16 __h14, _Float16 __h15, _Float16 __h16,
                          _Float16 __h17, _Float16 __h18, _Float16 __h19, _Float16 __h20,
                          _Float16 __h21, _Float16 __h22, _Float16 __h23, _Float16 __h24,
                          _Float16 __h25, _Float16 __h26, _Float16 __h27, _Float16 __h28,
                          _Float16 __h29, _Float16 __h30, _Float16 __h31, _Float16 __h32) {
  // CHECK-LABEL: @test_mm512_set_ph
  // CHECK: insertelement <32 x half> {{.*}}, i32 0
  // CHECK: insertelement <32 x half> {{.*}}, i32 1
  // CHECK: insertelement <32 x half> {{.*}}, i32 2
  // CHECK: insertelement <32 x half> {{.*}}, i32 3
  // CHECK: insertelement <32 x half> {{.*}}, i32 4
  // CHECK: insertelement <32 x half> {{.*}}, i32 5
  // CHECK: insertelement <32 x half> {{.*}}, i32 6
  // CHECK: insertelement <32 x half> {{.*}}, i32 7
  // CHECK: insertelement <32 x half> {{.*}}, i32 8
  // CHECK: insertelement <32 x half> {{.*}}, i32 9
  // CHECK: insertelement <32 x half> {{.*}}, i32 10
  // CHECK: insertelement <32 x half> {{.*}}, i32 11
  // CHECK: insertelement <32 x half> {{.*}}, i32 12
  // CHECK: insertelement <32 x half> {{.*}}, i32 13
  // CHECK: insertelement <32 x half> {{.*}}, i32 14
  // CHECK: insertelement <32 x half> {{.*}}, i32 15
  // CHECK: insertelement <32 x half> {{.*}}, i32 16
  // CHECK: insertelement <32 x half> {{.*}}, i32 17
  // CHECK: insertelement <32 x half> {{.*}}, i32 18
  // CHECK: insertelement <32 x half> {{.*}}, i32 19
  // CHECK: insertelement <32 x half> {{.*}}, i32 20
  // CHECK: insertelement <32 x half> {{.*}}, i32 21
  // CHECK: insertelement <32 x half> {{.*}}, i32 22
  // CHECK: insertelement <32 x half> {{.*}}, i32 23
  // CHECK: insertelement <32 x half> {{.*}}, i32 24
  // CHECK: insertelement <32 x half> {{.*}}, i32 25
  // CHECK: insertelement <32 x half> {{.*}}, i32 26
  // CHECK: insertelement <32 x half> {{.*}}, i32 27
  // CHECK: insertelement <32 x half> {{.*}}, i32 28
  // CHECK: insertelement <32 x half> {{.*}}, i32 29
  // CHECK: insertelement <32 x half> {{.*}}, i32 30
  // CHECK: insertelement <32 x half> {{.*}}, i32 31
  return _mm512_set_ph(__h1, __h2, __h3, __h4, __h5, __h6, __h7, __h8,
                       __h9, __h10, __h11, __h12, __h13, __h14, __h15, __h16,
                       __h17, __h18, __h19, __h20, __h21, __h22, __h23, __h24,
                       __h25, __h26, __h27, __h28, __h29, __h30, __h31, __h32);
}

__m512h test_mm512_setr_ph(_Float16 __h1, _Float16 __h2, _Float16 __h3, _Float16 __h4,
                           _Float16 __h5, _Float16 __h6, _Float16 __h7, _Float16 __h8,
                           _Float16 __h9, _Float16 __h10, _Float16 __h11, _Float16 __h12,
                           _Float16 __h13, _Float16 __h14, _Float16 __h15, _Float16 __h16,
                           _Float16 __h17, _Float16 __h18, _Float16 __h19, _Float16 __h20,
                           _Float16 __h21, _Float16 __h22, _Float16 __h23, _Float16 __h24,
                           _Float16 __h25, _Float16 __h26, _Float16 __h27, _Float16 __h28,
                           _Float16 __h29, _Float16 __h30, _Float16 __h31, _Float16 __h32) {
  // CHECK-LABEL: @test_mm512_setr_ph
  // CHECK: insertelement <32 x half> {{.*}}, i32 0
  // CHECK: insertelement <32 x half> {{.*}}, i32 1
  // CHECK: insertelement <32 x half> {{.*}}, i32 2
  // CHECK: insertelement <32 x half> {{.*}}, i32 3
  // CHECK: insertelement <32 x half> {{.*}}, i32 4
  // CHECK: insertelement <32 x half> {{.*}}, i32 5
  // CHECK: insertelement <32 x half> {{.*}}, i32 6
  // CHECK: insertelement <32 x half> {{.*}}, i32 7
  // CHECK: insertelement <32 x half> {{.*}}, i32 8
  // CHECK: insertelement <32 x half> {{.*}}, i32 9
  // CHECK: insertelement <32 x half> {{.*}}, i32 10
  // CHECK: insertelement <32 x half> {{.*}}, i32 11
  // CHECK: insertelement <32 x half> {{.*}}, i32 12
  // CHECK: insertelement <32 x half> {{.*}}, i32 13
  // CHECK: insertelement <32 x half> {{.*}}, i32 14
  // CHECK: insertelement <32 x half> {{.*}}, i32 15
  // CHECK: insertelement <32 x half> {{.*}}, i32 16
  // CHECK: insertelement <32 x half> {{.*}}, i32 17
  // CHECK: insertelement <32 x half> {{.*}}, i32 18
  // CHECK: insertelement <32 x half> {{.*}}, i32 19
  // CHECK: insertelement <32 x half> {{.*}}, i32 20
  // CHECK: insertelement <32 x half> {{.*}}, i32 21
  // CHECK: insertelement <32 x half> {{.*}}, i32 22
  // CHECK: insertelement <32 x half> {{.*}}, i32 23
  // CHECK: insertelement <32 x half> {{.*}}, i32 24
  // CHECK: insertelement <32 x half> {{.*}}, i32 25
  // CHECK: insertelement <32 x half> {{.*}}, i32 26
  // CHECK: insertelement <32 x half> {{.*}}, i32 27
  // CHECK: insertelement <32 x half> {{.*}}, i32 28
  // CHECK: insertelement <32 x half> {{.*}}, i32 29
  // CHECK: insertelement <32 x half> {{.*}}, i32 30
  // CHECK: insertelement <32 x half> {{.*}}, i32 31
  return _mm512_setr_ph(__h1, __h2, __h3, __h4, __h5, __h6, __h7, __h8,
                        __h9, __h10, __h11, __h12, __h13, __h14, __h15, __h16,
                        __h17, __h18, __h19, __h20, __h21, __h22, __h23, __h24,
                        __h25, __h26, __h27, __h28, __h29, __h30, __h31, __h32);
}

__m128 test_mm_castph_ps(__m128h A) {
  // CHECK-LABEL: test_mm_castph_ps
  // CHECK: bitcast <8 x half> %{{.*}} to <4 x float>
  return _mm_castph_ps(A);
}

__m256 test_mm256_castph_ps(__m256h A) {
  // CHECK-LABEL: test_mm256_castph_ps
  // CHECK: bitcast <16 x half> %{{.*}} to <8 x float>
  return _mm256_castph_ps(A);
}

__m512 test_mm512_castph_ps(__m512h A) {
  // CHECK-LABEL: test_mm512_castph_ps
  // CHECK: bitcast <32 x half> %{{.*}} to <16 x float>
  return _mm512_castph_ps(A);
}

__m128d test_mm_castph_pd(__m128h A) {
  // CHECK-LABEL: test_mm_castph_pd
  // CHECK: bitcast <8 x half> %{{.*}} to <2 x double>
  return _mm_castph_pd(A);
}

__m256d test_mm256_castph_pd(__m256h A) {
  // CHECK-LABEL: test_mm256_castph_pd
  // CHECK: bitcast <16 x half> %{{.*}} to <4 x double>
  return _mm256_castph_pd(A);
}

__m512d test_mm512_castph_pd(__m512h A) {
  // CHECK-LABEL: test_mm512_castph_pd
  // CHECK: bitcast <32 x half> %{{.*}} to <8 x double>
  return _mm512_castph_pd(A);
}

__m128i test_mm_castph_si128(__m128h A) {
  // CHECK-LABEL: test_mm_castph_si128
  // CHECK: bitcast <8 x half> %{{.*}} to <2 x i64>
  return _mm_castph_si128(A);
}

__m256i test_mm256_castph_si256(__m256h A) {
  // CHECK-LABEL: test_mm256_castph_si256
  // CHECK: bitcast <16 x half> %{{.*}} to <4 x i64>
  return _mm256_castph_si256(A);
}

__m512i test_mm512_castph_si512(__m512h A) {
  // CHECK-LABEL: test_mm512_castph_si512
  // CHECK: bitcast <32 x half> %{{.*}} to <8 x i64>
  return _mm512_castph_si512(A);
}

__m128h test_mm_castps_ph(__m128 A) {
  // CHECK-LABEL: test_mm_castps_ph
  // CHECK: bitcast <4 x float> %{{.*}} to <8 x half>
  return _mm_castps_ph(A);
}

__m256h test_mm256_castps_ph(__m256 A) {
  // CHECK-LABEL: test_mm256_castps_ph
  // CHECK: bitcast <8 x float> %{{.*}} to <16 x half>
  return _mm256_castps_ph(A);
}

__m512h test_mm512_castps_ph(__m512 A) {
  // CHECK-LABEL: test_mm512_castps_ph
  // CHECK: bitcast <16 x float> %{{.*}} to <32 x half>
  return _mm512_castps_ph(A);
}

__m128h test_mm_castpd_ph(__m128d A) {
  // CHECK-LABEL: test_mm_castpd_ph
  // CHECK: bitcast <2 x double> %{{.*}} to <8 x half>
  return _mm_castpd_ph(A);
}

__m256h test_mm256_castpd_ph(__m256d A) {
  // CHECK-LABEL: test_mm256_castpd_ph
  // CHECK: bitcast <4 x double> %{{.*}} to <16 x half>
  return _mm256_castpd_ph(A);
}

__m512h test_mm512_castpd_ph(__m512d A) {
  // CHECK-LABEL: test_mm512_castpd_ph
  // CHECK: bitcast <8 x double> %{{.*}} to <32 x half>
  return _mm512_castpd_ph(A);
}

__m128h test_mm_castsi128_ph(__m128i A) {
  // CHECK-LABEL: test_mm_castsi128_ph
  // CHECK: bitcast <2 x i64> %{{.*}} to <8 x half>
  return _mm_castsi128_ph(A);
}

__m256h test_mm256_castsi256_ph(__m256i A) {
  // CHECK-LABEL: test_mm256_castsi256_ph
  // CHECK: bitcast <4 x i64> %{{.*}} to <16 x half>
  return _mm256_castsi256_ph(A);
}

__m512h test_mm512_castsi512_ph(__m512i A) {
  // CHECK-LABEL: test_mm512_castsi512_ph
  // CHECK: bitcast <8 x i64> %{{.*}} to <32 x half>
  return _mm512_castsi512_ph(A);
}

__m128h test_mm256_castph256_ph128(__m256h __a) {
  // CHECK-LABEL: test_mm256_castph256_ph128
  // CHECK: shufflevector <16 x half> %{{.*}}, <16 x half> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  return _mm256_castph256_ph128(__a);
}

__m128h test_mm512_castph512_ph128(__m512h __a) {
  // CHECK-LABEL: test_mm512_castph512_ph128
  // CHECK: shufflevector <32 x half> %{{.*}}, <32 x half> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  return _mm512_castph512_ph128(__a);
}

__m256h test_mm512_castph512_ph256(__m512h __a) {
  // CHECK-LABEL: test_mm512_castph512_ph256
  // CHECK: shufflevector <32 x half> %{{.*}}, <32 x half> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  return _mm512_castph512_ph256(__a);
}

__m256h test_mm256_castph128_ph256(__m128h __a) {
  // CHECK-LABEL: test_mm256_castph128_ph256
  // CHECK: shufflevector <8 x half> %{{.*}}, <8 x half> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  return _mm256_castph128_ph256(__a);
}

__m512h test_mm512_castph128_ph512(__m128h __a) {
  // CHECK-LABEL: test_mm512_castph128_ph512
  // CHECK: shufflevector <8 x half> %{{.*}}, <8 x half> %{{.*}}, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  return _mm512_castph128_ph512(__a);
}

__m512h test_mm512_castph256_ph512(__m256h __a) {
  // CHECK-LABEL: test_mm512_castph256_ph512
  // CHECK: shufflevector <16 x half> %{{.*}}, <16 x half> %{{.*}}, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  return _mm512_castph256_ph512(__a);
}

__m256h test_mm256_zextph128_ph256(__m128h __a) {
  // CHECK-LABEL: test_mm256_zextph128_ph256
  // CHECK: shufflevector <8 x half> %{{.*}}, <8 x half> {{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  return _mm256_zextph128_ph256(__a);
}

__m512h test_mm512_zextph128_ph512(__m128h __a) {
  // CHECK-LABEL: test_mm512_zextph128_ph512
  // CHECK: shufflevector <8 x half> %{{.*}}, <8 x half> {{.*}}, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  return _mm512_zextph128_ph512(__a);
}

__m512h test_mm512_zextph256_ph512(__m256h __a) {
  // CHECK-LABEL: test_mm512_zextph256_ph512
  // CHECK: shufflevector <16 x half> %{{.*}}, <16 x half> {{.*}}, <32 x i32>
  return _mm512_zextph256_ph512(__a);
}

__m512h test_mm512_abs_ph(__m512h a) {
  // CHECK-LABEL: @test_mm512_abs_ph
  // CHECK: and <16 x i32>
  return _mm512_abs_ph(a);
}

// VMOVSH

__m128h test_mm_load_sh(void const *A) {
  // CHECK-LABEL: test_mm_load_sh
  // CHECK: load half, half* %{{.*}}, align 1{{$}}
  return _mm_load_sh(A);
}

__m128h test_mm_mask_load_sh(__m128h __A, __mmask8 __U, const void *__W) {
  // CHECK-LABEL: @test_mm_mask_load_sh
  // CHECK: %{{.*}} = call <8 x half> @llvm.masked.load.v8f16.p0v8f16(<8 x half>* %{{.*}}, i32 1, <8 x i1> %{{.*}}, <8 x half> %{{.*}})
  return _mm_mask_load_sh(__A, __U, __W);
}

__m128h test_mm_maskz_load_sh(__mmask8 __U, const void *__W) {
  // CHECK-LABEL: @test_mm_maskz_load_sh
  // CHECK: %{{.*}} = call <8 x half> @llvm.masked.load.v8f16.p0v8f16(<8 x half>* %{{.*}}, i32 1, <8 x i1> %{{.*}}, <8 x half> %{{.*}})
  return _mm_maskz_load_sh(__U, __W);
}

__m512h test_mm512_load_ph(void *p) {
  // CHECK-LABEL: @test_mm512_load_ph
  // CHECK: load <32 x half>, <32 x half>* %{{.*}}, align 64
  return _mm512_load_ph(p);
}

__m256h test_mm256_load_ph(void *p) {
  // CHECK-LABEL: @test_mm256_load_ph
  // CHECK: load <16 x half>, <16 x half>* %{{.*}}, align 32
  return _mm256_load_ph(p);
}

__m128h test_mm_load_ph(void *p) {
  // CHECK-LABEL: @test_mm_load_ph
  // CHECK: load <8 x half>, <8 x half>* %{{.*}}, align 16
  return _mm_load_ph(p);
}

__m512h test_mm512_loadu_ph(void *p) {
  // CHECK-LABEL: @test_mm512_loadu_ph
  // CHECK: load <32 x half>, <32 x half>* {{.*}}, align 1{{$}}
  return _mm512_loadu_ph(p);
}

__m256h test_mm256_loadu_ph(void *p) {
  // CHECK-LABEL: @test_mm256_loadu_ph
  // CHECK: load <16 x half>, <16 x half>* {{.*}}, align 1{{$}}
  return _mm256_loadu_ph(p);
}

__m128h test_mm_loadu_ph(void *p) {
  // CHECK-LABEL: @test_mm_loadu_ph
  // CHECK: load <8 x half>, <8 x half>* {{.*}}, align 1{{$}}
  return _mm_loadu_ph(p);
}

void test_mm_store_sh(void *A, __m128h B) {
  // CHECK-LABEL: test_mm_store_sh
  // CHECK: extractelement <8 x half> %{{.*}}, i32 0
  // CHECK: store half %{{.*}}, half* %{{.*}}, align 1{{$}}
  _mm_store_sh(A, B);
}

void test_mm_mask_store_sh(void *__P, __mmask8 __U, __m128h __A) {
  // CHECK-LABEL: @test_mm_mask_store_sh
  // CHECK: call void @llvm.masked.store.v8f16.p0v8f16(<8 x half> %{{.*}}, <8 x half>* %{{.*}}, i32 1, <8 x i1> %{{.*}})
  _mm_mask_store_sh(__P, __U, __A);
}

void test_mm512_store_ph(void *p, __m512h a) {
  // CHECK-LABEL: @test_mm512_store_ph
  // CHECK: store <32 x half> %{{.*}}, <32 x half>* %{{.*}}, align 64
  _mm512_store_ph(p, a);
}

void test_mm256_store_ph(void *p, __m256h a) {
  // CHECK-LABEL: @test_mm256_store_ph
  // CHECK: store <16 x half> %{{.*}}, <16 x half>* %{{.*}}, align 32
  _mm256_store_ph(p, a);
}

void test_mm_store_ph(void *p, __m128h a) {
  // CHECK-LABEL: @test_mm_store_ph
  // CHECK: store <8 x half> %{{.*}}, <8 x half>* %{{.*}}, align 16
  _mm_store_ph(p, a);
}

void test_mm512_storeu_ph(void *p, __m512h a) {
  // CHECK-LABEL: @test_mm512_storeu_ph
  // CHECK: store <32 x half> %{{.*}}, <32 x half>* %{{.*}}, align 1{{$}}
  // CHECK-NEXT: ret void
  _mm512_storeu_ph(p, a);
}

void test_mm256_storeu_ph(void *p, __m256h a) {
  // CHECK-LABEL: @test_mm256_storeu_ph
  // CHECK: store <16 x half> %{{.*}}, <16 x half>* %{{.*}}, align 1{{$}}
  // CHECK-NEXT: ret void
  _mm256_storeu_ph(p, a);
}

void test_mm_storeu_ph(void *p, __m128h a) {
  // CHECK-LABEL: @test_mm_storeu_ph
  // CHECK: store <8 x half> %{{.*}}, <8 x half>* %{{.*}}, align 1{{$}}
  // CHECK-NEXT: ret void
  _mm_storeu_ph(p, a);
}

__m128h test_mm_move_sh(__m128h A, __m128h B) {
  // CHECK-LABEL: test_mm_move_sh
  // CHECK: extractelement <8 x half> %{{.*}}, i32 0
  // CHECK: insertelement <8 x half> %{{.*}}, half %{{.*}}, i32 0
  return _mm_move_sh(A, B);
}

__m128h test_mm_mask_move_sh(__m128h __W, __mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_mask_move_sh
  // CHECK: [[EXT:%.*]] = extractelement <8 x half> %{{.*}}, i32 0
  // CHECK: insertelement <8 x half> %{{.*}}, half [[EXT]], i32 0
  // CHECK: [[A:%.*]] = extractelement <8 x half> [[VEC:%.*]], i64 0
  // CHECK-NEXT: [[B:%.*]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.*]] = select i1 %{{.*}}, half [[A]], half [[B]]
  // CHECK-NEXT: insertelement <8 x half> [[VEC]], half [[SEL]], i64 0
  return _mm_mask_move_sh(__W, __U, __A, __B);
}

__m128h test_mm_maskz_move_sh(__mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_maskz_move_sh
  // CHECK: [[EXT:%.*]] = extractelement <8 x half> %{{.*}}, i32 0
  // CHECK: insertelement <8 x half> %{{.*}}, half [[EXT]], i32 0
  // CHECK: [[A:%.*]] = extractelement <8 x half> [[VEC:%.*]], i64 0
  // CHECK-NEXT: [[B:%.*]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.*]] = select i1 %{{.*}}, half [[A]], half [[B]]
  // CHECK-NEXT: insertelement <8 x half> [[VEC]], half [[SEL]], i64 0
  return _mm_maskz_move_sh(__U, __A, __B);
}

short test_mm_cvtsi128_si16(__m128i A) {
  // CHECK-LABEL: test_mm_cvtsi128_si16
  // CHECK: extractelement <8 x i16> %{{.*}}, i32 0
  return _mm_cvtsi128_si16(A);
}

__m128i test_mm_cvtsi16_si128(short A) {
  // CHECK-LABEL: test_mm_cvtsi16_si128
  // CHECK: insertelement <8 x i16> undef, i16 %{{.*}}, i32 0
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 0, i32 1
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 0, i32 2
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 0, i32 3
  return _mm_cvtsi16_si128(A);
}

__m512h test_mm512_mask_blend_ph(__mmask32 __U, __m512h __A, __m512h __W) {
  // CHECK-LABEL: @test_mm512_mask_blend_ph
  // CHECK:  %{{.*}} = bitcast i32 %{{.*}} to <32 x i1>
  // CHECK:  %{{.*}} = select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}
  return _mm512_mask_blend_ph(__U, __A, __W);
}

__m512h test_mm512_permutex2var_ph(__m512h __A, __m512i __I, __m512h __B) {
  // CHECK-LABEL: @test_mm512_permutex2var_ph
  // CHECK:  %{{.*}} = bitcast <32 x half> %{{.*}} to <32 x i16>
  // CHECK:  %{{.*}} = bitcast <8 x i64> %{{.*}} to <32 x i16>
  // CHECK:  %{{.*}} = bitcast <32 x half> %{{.*}} to <32 x i16>
  // CHECK:  %{{.*}} = call <32 x i16> @llvm.x86.avx512.vpermi2var.hi.512(<32 x i16> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}})
  // CHECK:  %{{.*}} = bitcast <32 x i16> %{{.*}} to <32 x half>
  return _mm512_permutex2var_ph(__A, __I, __B);
}

__m512h test_mm512_permutexvar_epi16(__m512i __A, __m512h __B) {
  // CHECK-LABEL: @test_mm512_permutexvar_epi16
  // CHECK:  %{{.*}} = bitcast <32 x half> %{{.*}} to <32 x i16>
  // CHECK:  %{{.*}} = bitcast <8 x i64> %{{.*}} to <32 x i16>
  // CHECK:  %{{.*}} = call <32 x i16> @llvm.x86.avx512.permvar.hi.512(<32 x i16> %{{.*}}, <32 x i16> %{{.*}})
  // CHECK:  %{{.*}} = bitcast <32 x i16> %{{.*}} to <32 x half>
  return _mm512_permutexvar_ph(__A, __B);
}

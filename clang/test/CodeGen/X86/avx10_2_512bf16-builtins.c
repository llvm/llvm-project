// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64 -target-feature +avx10.2-512 -emit-llvm -o - -Wno-invalid-feature-combination -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=i386 -target-feature +avx10.2-512 -emit-llvm -o - -Wno-invalid-feature-combination -Wall -Werror | FileCheck %s

#include <immintrin.h>

__m512bh test_mm512_setzero_pbh() {
  // CHECK-LABEL: @test_mm512_setzero_pbh
  // CHECK: zeroinitializer
  return _mm512_setzero_pbh();
}

__m512bh test_mm512_undefined_pbh(void) {
  // CHECK-LABEL: @test_mm512_undefined_pbh
  // CHECK: ret <32 x bfloat> zeroinitializer
  return _mm512_undefined_pbh();
}

__m512bh test_mm512_set1_pbh(__bf16 h) {
  // CHECK-LABEL: @test_mm512_set1_pbh
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 0
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 1
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 2
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 3
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 4
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 5
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 6
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 7
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 8
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 9
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 10
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 11
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 12
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 13
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 14
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 15
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 16
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 17
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 18
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 19
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 20
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 21
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 22
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 23
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 24
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 25
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 26
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 27
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 28
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 29
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 30
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 31
  return _mm512_set1_pbh(h);
}

__m512bh test_mm512_set_pbh(__bf16 bf1, __bf16 bf2, __bf16 bf3, __bf16 bf4,
                          __bf16 bf5, __bf16 bf6, __bf16 bf7, __bf16 bf8,
                          __bf16 bf9, __bf16 bf10, __bf16 bf11, __bf16 bf12,
                          __bf16 bf13, __bf16 bf14, __bf16 bf15, __bf16 bf16,
                          __bf16 bf17, __bf16 bf18, __bf16 bf19, __bf16 bf20,
                          __bf16 bf21, __bf16 bf22, __bf16 bf23, __bf16 bf24,
                          __bf16 bf25, __bf16 bf26, __bf16 bf27, __bf16 bf28,
                          __bf16 bf29, __bf16 bf30, __bf16 bf31, __bf16 bf32) {
  // CHECK-LABEL: @test_mm512_set_pbh
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 0
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 1
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 2
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 3
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 4
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 5
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 6
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 7
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 8
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 9
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 10
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 11
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 12
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 13
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 14
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 15
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 16
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 17
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 18
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 19
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 20
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 21
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 22
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 23
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 24
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 25
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 26
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 27
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 28
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 29
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 30
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 31
  return _mm512_set_pbh(bf1, bf2, bf3, bf4, bf5, bf6, bf7, bf8,
                       bf9, bf10, bf11, bf12, bf13, bf14, bf15, bf16,
                       bf17, bf18, bf19, bf20, bf21, bf22, bf23, bf24,
                       bf25, bf26, bf27, bf28, bf29, bf30, bf31, bf32);
}

__m512bh test_mm512_setr_pbh(__bf16 bf1, __bf16 bf2, __bf16 bf3, __bf16 bf4,
                           __bf16 bf5, __bf16 bf6, __bf16 bf7, __bf16 bf8,
                           __bf16 bf9, __bf16 bf10, __bf16 bf11, __bf16 bf12,
                           __bf16 bf13, __bf16 bf14, __bf16 bf15, __bf16 bf16,
                           __bf16 bf17, __bf16 bf18, __bf16 bf19, __bf16 bf20,
                           __bf16 bf21, __bf16 bf22, __bf16 bf23, __bf16 bf24,
                           __bf16 bf25, __bf16 bf26, __bf16 bf27, __bf16 bf28,
                           __bf16 bf29, __bf16 bf30, __bf16 bf31, __bf16 bf32) {
  // CHECK-LABEL: @test_mm512_setr_pbh
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 0
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 1
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 2
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 3
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 4
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 5
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 6
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 7
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 8
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 9
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 10
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 11
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 12
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 13
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 14
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 15
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 16
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 17
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 18
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 19
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 20
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 21
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 22
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 23
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 24
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 25
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 26
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 27
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 28
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 29
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 30
  // CHECK: insertelement <32 x bfloat> {{.*}}, i32 31
  return _mm512_setr_pbh(bf1, bf2, bf3, bf4, bf5, bf6, bf7, bf8,
                        bf9, bf10, bf11, bf12, bf13, bf14, bf15, bf16,
                        bf17, bf18, bf19, bf20, bf21, bf22, bf23, bf24,
                        bf25, bf26, bf27, bf28, bf29, bf30, bf31, bf32);
}

__m512 test_mm512_castbf16_ps(__m512bh A) {
  // CHECK-LABEL: test_mm512_castbf16_ps
  // CHECK: bitcast <32 x bfloat> %{{.*}} to <16 x float>
  return _mm512_castbf16_ps(A);
}

__m512d test_mm512_castbf16_pd(__m512bh A) {
  // CHECK-LABEL: test_mm512_castbf16_pd
  // CHECK: bitcast <32 x bfloat> %{{.*}} to <8 x double>
  return _mm512_castbf16_pd(A);
}

__m512i test_mm512_castbf16_si512(__m512bh A) {
  // CHECK-LABEL: test_mm512_castbf16_si512
  // CHECK: bitcast <32 x bfloat> %{{.*}} to <8 x i64>
  return _mm512_castbf16_si512(A);
}

__m512bh test_mm512_castps_pbh(__m512 A) {
  // CHECK-LABEL: test_mm512_castps_pbh
  // CHECK: bitcast <16 x float> %{{.*}} to <32 x bfloat>
  return _mm512_castps_pbh(A);
}

__m512bh test_mm512_castpd_pbh(__m512d A) {
  // CHECK-LABEL: test_mm512_castpd_pbh
  // CHECK: bitcast <8 x double> %{{.*}} to <32 x bfloat>
  return _mm512_castpd_pbh(A);
}

__m512bh test_mm512_castsi512_pbh(__m512i A) {
  // CHECK-LABEL: test_mm512_castsi512_pbh
  // CHECK: bitcast <8 x i64> %{{.*}} to <32 x bfloat>
  return _mm512_castsi512_pbh(A);
}

__m128bh test_mm512_castbf16512_pbh128(__m512bh __a) {
  // CHECK-LABEL: test_mm512_castbf16512_pbh128
  // CHECK: shufflevector <32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  return _mm512_castbf16512_pbh128(__a);
}

__m256bh test_mm512_castbf16512_pbh256(__m512bh __a) {
  // CHECK-LABEL: test_mm512_castbf16512_pbh256
  // CHECK: shufflevector <32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  return _mm512_castbf16512_pbh256(__a);
}

__m512bh test_mm512_castbf16128_pbh512(__m128bh __a) {
  // CHECK-LABEL: test_mm512_castbf16128_pbh512
  // CHECK: shufflevector <8 x bfloat> %{{.*}}, <8 x bfloat> %{{.*}}, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  return _mm512_castbf16128_pbh512(__a);
}

__m512bh test_mm512_castbf16256_pbh512(__m256bh __a) {
  // CHECK-LABEL: test_mm512_castbf16256_pbh512
  // CHECK: shufflevector <16 x bfloat> %{{.*}}, <16 x bfloat> %{{.*}}, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  return _mm512_castbf16256_pbh512(__a);
}

__m512bh test_mm512_zextbf16128_pbh512(__m128bh __a) {
  // CHECK-LABEL: test_mm512_zextbf16128_pbh512
  // CHECK: shufflevector <8 x bfloat> %{{.*}}, <8 x bfloat> {{.*}}, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  return _mm512_zextbf16128_pbh512(__a);
}

__m512bh test_mm512_zextbf16256_pbh512(__m256bh __a) {
  // CHECK-LABEL: test_mm512_zextbf16256_pbh512
  // CHECK: shufflevector <16 x bfloat> %{{.*}}, <16 x bfloat> {{.*}}, <32 x i32>
  return _mm512_zextbf16256_pbh512(__a);
}

__m512bh test_mm512_abs_pbh(__m512bh a) {
  // CHECK-LABEL: @test_mm512_abs_pbh
  // CHECK: and <16 x i32>
  return _mm512_abs_pbh(a);
}

// VMOVSH

__m512bh test_mm512_load_pbh(void *p) {
  // CHECK-LABEL: @test_mm512_load_pbh
  // CHECK: load <32 x bfloat>, ptr %{{.*}}, align 64
  return _mm512_load_pbh(p);
}

__m512bh test_mm512_loadu_pbh(void *p) {
  // CHECK-LABEL: @test_mm512_loadu_pbh
  // CHECK: load <32 x bfloat>, ptr {{.*}}, align 1{{$}}
  return _mm512_loadu_pbh(p);
}

void test_mm512_store_pbh(void *p, __m512bh a) {
  // CHECK-LABEL: @test_mm512_store_pbh
  // CHECK: store <32 x bfloat> %{{.*}}, ptr %{{.*}}, align 64
  _mm512_store_pbh(p, a);
}

void test_mm512_storeu_pbh(void *p, __m512bh a) {
  // CHECK-LABEL: @test_mm512_storeu_pbh
  // CHECK: store <32 x bfloat> %{{.*}}, ptr %{{.*}}, align 1{{$}}
  // CHECK-NEXT: ret void
  _mm512_storeu_pbh(p, a);
}

__m512bh test_mm512_mask_blend_pbh(__mmask32 __U, __m512bh __A, __m512bh __W) {
  // CHECK-LABEL: @test_mm512_mask_blend_pbh
  // CHECK:  %{{.*}} = bitcast i32 %{{.*}} to <32 x i1>
  // CHECK:  %{{.*}} = select <32 x i1> %{{.*}}, <32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}}
  return _mm512_mask_blend_pbh(__U, __A, __W);
}

__m512bh test_mm512_permutex2var_pbh(__m512bh __A, __m512i __I, __m512bh __B) {
  // CHECK-LABEL: @test_mm512_permutex2var_pbh
  // CHECK:  %{{.*}} = bitcast <32 x bfloat> %{{.*}} to <32 x i16>
  // CHECK:  %{{.*}} = bitcast <8 x i64> %{{.*}} to <32 x i16>
  // CHECK:  %{{.*}} = bitcast <32 x bfloat> %{{.*}} to <32 x i16>
  // CHECK:  %{{.*}} = call <32 x i16> @llvm.x86.avx512.vpermi2var.hi.512(<32 x i16> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}})
  // CHECK:  %{{.*}} = bitcast <32 x i16> %{{.*}} to <32 x bfloat>
  return _mm512_permutex2var_pbh(__A, __I, __B);
}

__m512bh test_mm512_permutexvar_epi16(__m512i __A, __m512bh __B) {
  // CHECK-LABEL: @test_mm512_permutexvar_epi16
  // CHECK:  %{{.*}} = bitcast <32 x bfloat> %{{.*}} to <32 x i16>
  // CHECK:  %{{.*}} = bitcast <8 x i64> %{{.*}} to <32 x i16>
  // CHECK:  %{{.*}} = call <32 x i16> @llvm.x86.avx512.permvar.hi.512(<32 x i16> %{{.*}}, <32 x i16> %{{.*}})
  // CHECK:  %{{.*}} = bitcast <32 x i16> %{{.*}} to <32 x bfloat>
  return _mm512_permutexvar_pbh(__A, __B);
}

__m512bh test_mm512_add_pbh(__m512bh __A, __m512bh __B) {
  // CHECK-LABEL: @test_mm512_add_pbh
  // CHECK: %{{.*}} = fadd <32 x bfloat> %{{.*}}, %{{.*}}
  return _mm512_add_pbh(__A, __B);
}

__m512bh test_mm512_mask_add_pbh(__m512bh __W, __mmask32 __U, __m512bh __A, __m512bh __B) {
  // CHECK: %{{.*}} = fadd <32 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: select <32 x i1> %{{.*}}, <32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}}
  return _mm512_mask_add_pbh(__W, __U, __A, __B);
}

__m512bh test_mm512_maskz_add_pbh(__mmask32 __U, __m512bh __A, __m512bh __B) {
  // CHECK: %{{.*}} = fadd <32 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: select <32 x i1> %{{.*}}, <32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}}
  return _mm512_maskz_add_pbh(__U, __A, __B);
}

__m512bh test_mm512_sub_pbh(__m512bh __A, __m512bh __B) {
  // CHECK-LABEL: @test_mm512_sub_pbh
  // CHECK: %{{.*}} = fsub <32 x bfloat> %{{.*}}, %{{.*}}
  return _mm512_sub_pbh(__A, __B);
}

__m512bh test_mm512_mask_sub_pbh(__m512bh __W, __mmask32 __U, __m512bh __A, __m512bh __B) {
  // CHECK: %{{.*}} = fsub <32 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: select <32 x i1> %{{.*}}, <32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}}
  return _mm512_mask_sub_pbh(__W, __U, __A, __B);
}

__m512bh test_mm512_maskz_sub_pbh(__mmask32 __U, __m512bh __A, __m512bh __B) {
  // CHECK: %{{.*}} = fsub <32 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: select <32 x i1> %{{.*}}, <32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}}
  return _mm512_maskz_sub_pbh(__U, __A, __B);
}

__m512bh test_mm512_mul_pbh(__m512bh __A, __m512bh __B) {
  // CHECK-LABEL: @test_mm512_mul_pbh
  // CHECK: %{{.*}} = fmul <32 x bfloat> %{{.*}}, %{{.*}}
  return _mm512_mul_pbh(__A, __B);
}

__m512bh test_mm512_mask_mul_pbh(__m512bh __W, __mmask32 __U, __m512bh __A, __m512bh __B) {
  // CHECK: %{{.*}} = fmul <32 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: select <32 x i1> %{{.*}}, <32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}}
  return _mm512_mask_mul_pbh(__W, __U, __A, __B);
}

__m512bh test_mm512_maskz_mul_pbh(__mmask32 __U, __m512bh __A, __m512bh __B) {
  // CHECK: %{{.*}} = fmul <32 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: select <32 x i1> %{{.*}}, <32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}}
  return _mm512_maskz_mul_pbh(__U, __A, __B);
}

__m512bh test_mm512_div_pbh(__m512bh __A, __m512bh __B) {
  // CHECK-LABEL: @test_mm512_div_pbh
  // CHECK: %{{.*}} = fdiv <32 x bfloat> %{{.*}}, %{{.*}}
  return _mm512_div_pbh(__A, __B);
}

__m512bh test_mm512_mask_div_pbh(__m512bh __W, __mmask32 __U, __m512bh __A, __m512bh __B) {
  // CHECK: %{{.*}} = fdiv <32 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: select <32 x i1> %{{.*}}, <32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}}
  return _mm512_mask_div_pbh(__W, __U, __A, __B);
}

__m512bh test_mm512_maskz_div_pbh(__mmask32 __U, __m512bh __A, __m512bh __B) {
  // CHECK: %{{.*}} = fdiv <32 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: select <32 x i1> %{{.*}}, <32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}}
  return _mm512_maskz_div_pbh(__U, __A, __B);
}

__m512bh test_mm512_max_pbh(__m512bh __A, __m512bh __B) {
  // CHECK-LABEL: @test_mm512_max_pbh
  // CHECK: @llvm.x86.avx10.vmaxbf16512(
  return _mm512_max_pbh(__A, __B);
}

__m512bh test_mm512_mask_max_pbh(__m512bh __W, __mmask32 __U, __m512bh __A, __m512bh __B) {
  // CHECK: @llvm.x86.avx10.vmaxbf16512
  // CHECK: select <32 x i1> %{{.*}}, <32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}}
  return _mm512_mask_max_pbh(__W, __U, __A, __B);
}

__m512bh test_mm512_maskz_max_pbh(__mmask32 __U, __m512bh __A, __m512bh __B) {
  // CHECK: @llvm.x86.avx10.vmaxbf16512
  // CHECK: select <32 x i1> %{{.*}}, <32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}}
  return _mm512_maskz_max_pbh(__U, __A, __B);
}

__m512bh test_mm512_min_pbh(__m512bh __A, __m512bh __B) {
  // CHECK-LABEL: @test_mm512_min_pbh
  // CHECK: @llvm.x86.avx10.vminbf16512(
  return _mm512_min_pbh(__A, __B);
}

__m512bh test_mm512_mask_min_pbh(__m512bh __W, __mmask32 __U, __m512bh __A, __m512bh __B) {
  // CHECK: @llvm.x86.avx10.vminbf16512
  // CHECK: select <32 x i1> %{{.*}}, <32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}}
  return _mm512_mask_min_pbh(__W, __U, __A, __B);
}

__m512bh test_mm512_maskz_min_pbh(__mmask32 __U, __m512bh __A, __m512bh __B) {
  // CHECK: @llvm.x86.avx10.vminbf16512
  // CHECK: select <32 x i1> %{{.*}}, <32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}}
  return _mm512_maskz_min_pbh(__U, __A, __B);
}

__mmask32 test_mm512_cmp_pbh_mask_eq_oq(__m512bh a, __m512bh b) {
  // CHECK-LABEL: @test_mm512_cmp_pbh_mask_eq_oq
  // CHECK: fcmp oeq <32 x bfloat> %{{.*}}, %{{.*}}
  return _mm512_cmp_pbh_mask(a, b, _CMP_EQ_OQ);
}

__mmask32 test_mm512_cmp_pbh_mask_lt_os(__m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_cmp_pbh_mask_lt_os
  // CHECK: fcmp olt <32 x bfloat> %{{.*}}, %{{.*}}
  return _mm512_cmp_pbh_mask(a, b, _CMP_LT_OS);
}

__mmask32 test_mm512_cmp_pbh_mask_le_os(__m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_cmp_pbh_mask_le_os
  // CHECK: fcmp ole <32 x bfloat> %{{.*}}, %{{.*}}
  return _mm512_cmp_pbh_mask(a, b, _CMP_LE_OS);
}

__mmask32 test_mm512_cmp_pbh_mask_unord_q(__m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_cmp_pbh_mask_unord_q
  // CHECK: fcmp uno <32 x bfloat> %{{.*}}, %{{.*}}
  return _mm512_cmp_pbh_mask(a, b, _CMP_UNORD_Q);
}

__mmask32 test_mm512_cmp_pbh_mask_neq_uq(__m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_cmp_pbh_mask_neq_uq
  // CHECK: fcmp une <32 x bfloat> %{{.*}}, %{{.*}}
  return _mm512_cmp_pbh_mask(a, b, _CMP_NEQ_UQ);
}

__mmask32 test_mm512_cmp_pbh_mask_nlt_us(__m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_cmp_pbh_mask_nlt_us
  // CHECK: fcmp uge <32 x bfloat> %{{.*}}, %{{.*}}
  return _mm512_cmp_pbh_mask(a, b, _CMP_NLT_US);
}

__mmask32 test_mm512_cmp_pbh_mask_nle_us(__m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_cmp_pbh_mask_nle_us
  // CHECK: fcmp ugt <32 x bfloat> %{{.*}}, %{{.*}}
  return _mm512_cmp_pbh_mask(a, b, _CMP_NLE_US);
}

__mmask32 test_mm512_cmp_pbh_mask_ord_q(__m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_cmp_pbh_mask_ord_q
  // CHECK: fcmp ord <32 x bfloat> %{{.*}}, %{{.*}}
  return _mm512_cmp_pbh_mask(a, b, _CMP_ORD_Q);
}

__mmask32 test_mm512_cmp_pbh_mask_eq_uq(__m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_cmp_pbh_mask_eq_uq
  // CHECK: fcmp ueq <32 x bfloat> %{{.*}}, %{{.*}}
  return _mm512_cmp_pbh_mask(a, b, _CMP_EQ_UQ);
}

__mmask32 test_mm512_cmp_pbh_mask_nge_us(__m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_cmp_pbh_mask_nge_us
  // CHECK: fcmp ult <32 x bfloat> %{{.*}}, %{{.*}}
  return _mm512_cmp_pbh_mask(a, b, _CMP_NGE_US);
}

__mmask32 test_mm512_cmp_pbh_mask_ngt_us(__m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_cmp_pbh_mask_ngt_us
  // CHECK: fcmp ule <32 x bfloat> %{{.*}}, %{{.*}}
  return _mm512_cmp_pbh_mask(a, b, _CMP_NGT_US);
}

__mmask32 test_mm512_cmp_pbh_mask_false_oq(__m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_cmp_pbh_mask_false_oq
  // CHECK: fcmp false <32 x bfloat> %{{.*}}, %{{.*}}
  return _mm512_cmp_pbh_mask(a, b, _CMP_FALSE_OQ);
}

__mmask32 test_mm512_cmp_pbh_mask_neq_oq(__m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_cmp_pbh_mask_neq_oq
  // CHECK: fcmp one <32 x bfloat> %{{.*}}, %{{.*}}
  return _mm512_cmp_pbh_mask(a, b, _CMP_NEQ_OQ);
}

__mmask32 test_mm512_cmp_pbh_mask_ge_os(__m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_cmp_pbh_mask_ge_os
  // CHECK: fcmp oge <32 x bfloat> %{{.*}}, %{{.*}}
  return _mm512_cmp_pbh_mask(a, b, _CMP_GE_OS);
}

__mmask32 test_mm512_cmp_pbh_mask_gt_os(__m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_cmp_pbh_mask_gt_os
  // CHECK: fcmp ogt <32 x bfloat> %{{.*}}, %{{.*}}
  return _mm512_cmp_pbh_mask(a, b, _CMP_GT_OS);
}

__mmask32 test_mm512_cmp_pbh_mask_true_uq(__m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_cmp_pbh_mask_true_uq
  // CHECK: fcmp true <32 x bfloat> %{{.*}}, %{{.*}}
  return _mm512_cmp_pbh_mask(a, b, _CMP_TRUE_UQ);
}

__mmask32 test_mm512_cmp_pbh_mask_eq_os(__m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_cmp_pbh_mask_eq_os
  // CHECK: fcmp oeq <32 x bfloat> %{{.*}}, %{{.*}}
  return _mm512_cmp_pbh_mask(a, b, _CMP_EQ_OS);
}

__mmask32 test_mm512_cmp_pbh_mask_lt_oq(__m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_cmp_pbh_mask_lt_oq
  // CHECK: fcmp olt <32 x bfloat> %{{.*}}, %{{.*}}
  return _mm512_cmp_pbh_mask(a, b, _CMP_LT_OQ);
}

__mmask32 test_mm512_cmp_pbh_mask_le_oq(__m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_cmp_pbh_mask_le_oq
  // CHECK: fcmp ole <32 x bfloat> %{{.*}}, %{{.*}}
  return _mm512_cmp_pbh_mask(a, b, _CMP_LE_OQ);
}

__mmask32 test_mm512_cmp_pbh_mask_unord_s(__m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_cmp_pbh_mask_unord_s
  // CHECK: fcmp uno <32 x bfloat> %{{.*}}, %{{.*}}
  return _mm512_cmp_pbh_mask(a, b, _CMP_UNORD_S);
}

__mmask32 test_mm512_cmp_pbh_mask_neq_us(__m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_cmp_pbh_mask_neq_us
  // CHECK: fcmp une <32 x bfloat> %{{.*}}, %{{.*}}
  return _mm512_cmp_pbh_mask(a, b, _CMP_NEQ_US);
}

__mmask32 test_mm512_cmp_pbh_mask_nlt_uq(__m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_cmp_pbh_mask_nlt_uq
  // CHECK: fcmp uge <32 x bfloat> %{{.*}}, %{{.*}}
  return _mm512_cmp_pbh_mask(a, b, _CMP_NLT_UQ);
}

__mmask32 test_mm512_cmp_pbh_mask_nle_uq(__m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_cmp_pbh_mask_nle_uq
  // CHECK: fcmp ugt <32 x bfloat> %{{.*}}, %{{.*}}
  return _mm512_cmp_pbh_mask(a, b, _CMP_NLE_UQ);
}

__mmask32 test_mm512_cmp_pbh_mask_ord_s(__m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_cmp_pbh_mask_ord_s
  // CHECK: fcmp ord <32 x bfloat> %{{.*}}, %{{.*}}
  return _mm512_cmp_pbh_mask(a, b, _CMP_ORD_S);
}

__mmask32 test_mm512_cmp_pbh_mask_eq_us(__m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_cmp_pbh_mask_eq_us
  // CHECK: fcmp ueq <32 x bfloat> %{{.*}}, %{{.*}}
  return _mm512_cmp_pbh_mask(a, b, _CMP_EQ_US);
}

__mmask32 test_mm512_cmp_pbh_mask_nge_uq(__m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_cmp_pbh_mask_nge_uq
  // CHECK: fcmp ult <32 x bfloat> %{{.*}}, %{{.*}}
  return _mm512_cmp_pbh_mask(a, b, _CMP_NGE_UQ);
}

__mmask32 test_mm512_cmp_pbh_mask_ngt_uq(__m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_cmp_pbh_mask_ngt_uq
  // CHECK: fcmp ule <32 x bfloat> %{{.*}}, %{{.*}}
  return _mm512_cmp_pbh_mask(a, b, _CMP_NGT_UQ);
}

__mmask32 test_mm512_cmp_pbh_mask_false_os(__m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_cmp_pbh_mask_false_os
  // CHECK: fcmp false <32 x bfloat> %{{.*}}, %{{.*}}
  return _mm512_cmp_pbh_mask(a, b, _CMP_FALSE_OS);
}

__mmask32 test_mm512_cmp_pbh_mask_neq_os(__m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_cmp_pbh_mask_neq_os
  // CHECK: fcmp one <32 x bfloat> %{{.*}}, %{{.*}}
  return _mm512_cmp_pbh_mask(a, b, _CMP_NEQ_OS);
}

__mmask32 test_mm512_cmp_pbh_mask_ge_oq(__m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_cmp_pbh_mask_ge_oq
  // CHECK: fcmp oge <32 x bfloat> %{{.*}}, %{{.*}}
  return _mm512_cmp_pbh_mask(a, b, _CMP_GE_OQ);
}

__mmask32 test_mm512_cmp_pbh_mask_gt_oq(__m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_cmp_pbh_mask_gt_oq
  // CHECK: fcmp ogt <32 x bfloat> %{{.*}}, %{{.*}}
  return _mm512_cmp_pbh_mask(a, b, _CMP_GT_OQ);
}

__mmask32 test_mm512_cmp_pbh_mask_true_us(__m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_cmp_pbh_mask_true_us
  // CHECK: fcmp true <32 x bfloat> %{{.*}}, %{{.*}}
  return _mm512_cmp_pbh_mask(a, b, _CMP_TRUE_US);
}

__mmask32 test_mm512_mask_cmp_pbh_mask_eq_oq(__mmask32 m, __m512bh a, __m512bh b) {
  // CHECK-LABEL: @test_mm512_mask_cmp_pbh_mask_eq_oq
  // CHECK: fcmp oeq <32 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return _mm512_mask_cmp_pbh_mask(m, a, b, _CMP_EQ_OQ);
}

__mmask32 test_mm512_mask_cmp_pbh_mask_lt_os(__mmask32 m, __m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pbh_mask_lt_os
  // CHECK: fcmp olt <32 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return _mm512_mask_cmp_pbh_mask(m, a, b, _CMP_LT_OS);
}

__mmask32 test_mm512_mask_cmp_pbh_mask_le_os(__mmask32 m, __m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pbh_mask_le_os
  // CHECK: fcmp ole <32 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return _mm512_mask_cmp_pbh_mask(m, a, b, _CMP_LE_OS);
}

__mmask32 test_mm512_mask_cmp_pbh_mask_unord_q(__mmask32 m, __m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pbh_mask_unord_q
  // CHECK: fcmp uno <32 x bfloat> %{{.*}}, %{{.*}}
  return _mm512_mask_cmp_pbh_mask(m, a, b, _CMP_UNORD_Q);
}

__mmask32 test_mm512_mask_cmp_pbh_mask_neq_uq(__mmask32 m, __m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pbh_mask_neq_uq
  // CHECK: fcmp une <32 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return _mm512_mask_cmp_pbh_mask(m, a, b, _CMP_NEQ_UQ);
}

__mmask32 test_mm512_mask_cmp_pbh_mask_nlt_us(__mmask32 m, __m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pbh_mask_nlt_us
  // CHECK: fcmp uge <32 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return _mm512_mask_cmp_pbh_mask(m, a, b, _CMP_NLT_US);
}

__mmask32 test_mm512_mask_cmp_pbh_mask_nle_us(__mmask32 m, __m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pbh_mask_nle_us
  // CHECK: fcmp ugt <32 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return _mm512_mask_cmp_pbh_mask(m, a, b, _CMP_NLE_US);
}

__mmask32 test_mm512_mask_cmp_pbh_mask_ord_q(__mmask32 m, __m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pbh_mask_ord_q
  // CHECK: fcmp ord <32 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return _mm512_mask_cmp_pbh_mask(m, a, b, _CMP_ORD_Q);
}

__mmask32 test_mm512_mask_cmp_pbh_mask_eq_uq(__mmask32 m, __m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pbh_mask_eq_uq
  // CHECK: fcmp ueq <32 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return _mm512_mask_cmp_pbh_mask(m, a, b, _CMP_EQ_UQ);
}

__mmask32 test_mm512_mask_cmp_pbh_mask_nge_us(__mmask32 m, __m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pbh_mask_nge_us
  // CHECK: fcmp ult <32 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return _mm512_mask_cmp_pbh_mask(m, a, b, _CMP_NGE_US);
}

__mmask32 test_mm512_mask_cmp_pbh_mask_ngt_us(__mmask32 m, __m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pbh_mask_ngt_us
  // CHECK: fcmp ule <32 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return _mm512_mask_cmp_pbh_mask(m, a, b, _CMP_NGT_US);
}

__mmask32 test_mm512_mask_cmp_pbh_mask_false_oq(__mmask32 m, __m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pbh_mask_false_oq
  // CHECK: fcmp false <32 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return _mm512_mask_cmp_pbh_mask(m, a, b, _CMP_FALSE_OQ);
}

__mmask32 test_mm512_mask_cmp_pbh_mask_neq_oq(__mmask32 m, __m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pbh_mask_neq_oq
  // CHECK: fcmp one <32 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return _mm512_mask_cmp_pbh_mask(m, a, b, _CMP_NEQ_OQ);
}

__mmask32 test_mm512_mask_cmp_pbh_mask_ge_os(__mmask32 m, __m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pbh_mask_ge_os
  // CHECK: fcmp oge <32 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return _mm512_mask_cmp_pbh_mask(m, a, b, _CMP_GE_OS);
}

__mmask32 test_mm512_mask_cmp_pbh_mask_gt_os(__mmask32 m, __m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pbh_mask_gt_os
  // CHECK: fcmp ogt <32 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return _mm512_mask_cmp_pbh_mask(m, a, b, _CMP_GT_OS);
}

__mmask32 test_mm512_mask_cmp_pbh_mask_true_uq(__mmask32 m, __m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pbh_mask_true_uq
  // CHECK: fcmp true <32 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return _mm512_mask_cmp_pbh_mask(m, a, b, _CMP_TRUE_UQ);
}

__mmask32 test_mm512_mask_cmp_pbh_mask_eq_os(__mmask32 m, __m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pbh_mask_eq_os
  // CHECK: fcmp oeq <32 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return _mm512_mask_cmp_pbh_mask(m, a, b, _CMP_EQ_OS);
}

__mmask32 test_mm512_mask_cmp_pbh_mask_lt_oq(__mmask32 m, __m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pbh_mask_lt_oq
  // CHECK: fcmp olt <32 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return _mm512_mask_cmp_pbh_mask(m, a, b, _CMP_LT_OQ);
}

__mmask32 test_mm512_mask_cmp_pbh_mask_le_oq(__mmask32 m, __m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pbh_mask_le_oq
  // CHECK: fcmp ole <32 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return _mm512_mask_cmp_pbh_mask(m, a, b, _CMP_LE_OQ);
}

__mmask32 test_mm512_mask_cmp_pbh_mask_unord_s(__mmask32 m, __m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pbh_mask_unord_s
  // CHECK: fcmp uno <32 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return _mm512_mask_cmp_pbh_mask(m, a, b, _CMP_UNORD_S);
}

__mmask32 test_mm512_mask_cmp_pbh_mask_neq_us(__mmask32 m, __m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pbh_mask_neq_us
  // CHECK: fcmp une <32 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return _mm512_mask_cmp_pbh_mask(m, a, b, _CMP_NEQ_US);
}

__mmask32 test_mm512_mask_cmp_pbh_mask_nlt_uq(__mmask32 m, __m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pbh_mask_nlt_uq
  // CHECK: fcmp uge <32 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return _mm512_mask_cmp_pbh_mask(m, a, b, _CMP_NLT_UQ);
}

__mmask32 test_mm512_mask_cmp_pbh_mask_nle_uq(__mmask32 m, __m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pbh_mask_nle_uq
  // CHECK: fcmp ugt <32 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return _mm512_mask_cmp_pbh_mask(m, a, b, _CMP_NLE_UQ);
}

__mmask32 test_mm512_mask_cmp_pbh_mask_ord_s(__mmask32 m, __m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pbh_mask_ord_s
  // CHECK: fcmp ord <32 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return _mm512_mask_cmp_pbh_mask(m, a, b, _CMP_ORD_S);
}

__mmask32 test_mm512_mask_cmp_pbh_mask_eq_us(__mmask32 m, __m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pbh_mask_eq_us
  // CHECK: fcmp ueq <32 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return _mm512_mask_cmp_pbh_mask(m, a, b, _CMP_EQ_US);
}

__mmask32 test_mm512_mask_cmp_pbh_mask_nge_uq(__mmask32 m, __m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pbh_mask_nge_uq
  // CHECK: fcmp ult <32 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return _mm512_mask_cmp_pbh_mask(m, a, b, _CMP_NGE_UQ);
}

__mmask32 test_mm512_mask_cmp_pbh_mask_ngt_uq(__mmask32 m, __m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pbh_mask_ngt_uq
  // CHECK: fcmp ule <32 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return _mm512_mask_cmp_pbh_mask(m, a, b, _CMP_NGT_UQ);
}

__mmask32 test_mm512_mask_cmp_pbh_mask_false_os(__mmask32 m, __m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pbh_mask_false_os
  // CHECK: fcmp false <32 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return _mm512_mask_cmp_pbh_mask(m, a, b, _CMP_FALSE_OS);
}

__mmask32 test_mm512_mask_cmp_pbh_mask_neq_os(__mmask32 m, __m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pbh_mask_neq_os
  // CHECK: fcmp one <32 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return _mm512_mask_cmp_pbh_mask(m, a, b, _CMP_NEQ_OS);
}

__mmask32 test_mm512_mask_cmp_pbh_mask_ge_oq(__mmask32 m, __m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pbh_mask_ge_oq
  // CHECK: fcmp oge <32 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return _mm512_mask_cmp_pbh_mask(m, a, b, _CMP_GE_OQ);
}

__mmask32 test_mm512_mask_cmp_pbh_mask_gt_oq(__mmask32 m, __m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pbh_mask_gt_oq
  // CHECK: fcmp ogt <32 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return _mm512_mask_cmp_pbh_mask(m, a, b, _CMP_GT_OQ);
}

__mmask32 test_mm512_mask_cmp_pbh_mask_true_us(__mmask32 m, __m512bh a, __m512bh b) {
  // CHECK-LABEL: test_mm512_mask_cmp_pbh_mask_true_us
  // CHECK: fcmp true <32 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return _mm512_mask_cmp_pbh_mask(m, a, b, _CMP_TRUE_US);
}

__mmask32 test_mm512_mask_fpclass_pbh_mask(__mmask32 __U, __m512bh __A) {
  // CHECK-LABEL: @test_mm512_mask_fpclass_pbh_mask
  // CHECK: @llvm.x86.avx10.fpclass.bf16.512
  return _mm512_mask_fpclass_pbh_mask(__U, __A, 4);
}

__mmask32 test_mm512_fpclass_pbh_mask(__m512bh __A) {
  // CHECK-LABEL: @test_mm512_fpclass_pbh_mask
  // CHECK: @llvm.x86.avx10.fpclass.bf16.512
  return _mm512_fpclass_pbh_mask(__A, 4);
}

__m512bh test_mm512_scalef_pbh(__m512bh __A, __m512bh __B) {
  // CHECK-LABEL: @test_mm512_scalef_pbh
  // CHECK: @llvm.x86.avx10.mask.scalef.bf16.512
  return _mm512_scalef_pbh(__A, __B);
}

__m512bh test_mm512_mask_scalef_pbh(__m512bh __W, __mmask32 __U, __m512bh __A, __m512bh __B) {
  // CHECK-LABEL: @test_mm512_mask_scalef_pbh
  // CHECK: @llvm.x86.avx10.mask.scalef.bf16.512
  return _mm512_mask_scalef_pbh(__W, __U, __A, __B);
}

__m512bh test_mm512_maskz_scalef_pbh(__mmask32 __U, __m512bh __A, __m512bh __B) {
  // CHECK-LABEL: @test_mm512_maskz_scalef_pbh
  // CHECK: @llvm.x86.avx10.mask.scalef.bf16.512
  return _mm512_maskz_scalef_pbh(__U, __A, __B);
}

__m512bh test_mm512_rcp_pbh(__m512bh __A) {
  // CHECK-LABEL: @test_mm512_rcp_pbh
  // CHECK: @llvm.x86.avx10.mask.rcp.bf16.512
  return _mm512_rcp_pbh(__A);
}

__m512bh test_mm512_mask_rcp_pbh(__m512bh __W, __mmask32 __U, __m512bh __A) {
  // CHECK-LABEL: @test_mm512_mask_rcp_pbh
  // CHECK: @llvm.x86.avx10.mask.rcp.bf16.512
  return (__m512bh)_mm512_mask_rcp_pbh(__W, __U, __A);
}

__m512bh test_mm512_maskz_rcp_pbh(__mmask32 __U, __m512bh __A) {
  // CHECK-LABEL: @test_mm512_maskz_rcp_pbh
  // CHECK: @llvm.x86.avx10.mask.rcp.bf16.512
  return _mm512_maskz_rcp_pbh(__U, __A);
}

__m512bh test_mm512_getexp_pbh(__m512bh __A) {
  // CHECK-LABEL: @test_mm512_getexp_pbh
  // CHECK: @llvm.x86.avx10.mask.getexp.bf16.512
  return _mm512_getexp_pbh(__A);
}

__m512bh test_mm512_mask_getexp_pbh(__m512bh __W, __mmask32 __U, __m512bh __A) {
  // CHECK-LABEL: @test_mm512_mask_getexp_pbh
  // CHECK: @llvm.x86.avx10.mask.getexp.bf16.512
  return _mm512_mask_getexp_pbh(__W, __U, __A);
}

__m512bh test_mm512_maskz_getexp_pbh(__mmask32 __U, __m512bh __A) {
  // CHECK-LABEL: @test_mm512_maskz_getexp_pbh
  // CHECK: @llvm.x86.avx10.mask.getexp.bf16.512
  return _mm512_maskz_getexp_pbh(__U, __A);
}

__m512bh test_mm512_rsqrt_pbh(__m512bh __A) {
  // CHECK-LABEL: @test_mm512_rsqrt_pbh
  // CHECK: @llvm.x86.avx10.mask.rsqrt.bf16.512
  return _mm512_rsqrt_pbh(__A);
}

__m512bh test_mm512_mask_rsqrt_pbh(__m512bh __W, __mmask32 __U, __m512bh __A) {
  // CHECK-LABEL: @test_mm512_mask_rsqrt_pbh
  // CHECK: @llvm.x86.avx10.mask.rsqrt.bf16.512
  return (__m512bh)_mm512_mask_rsqrt_pbh(__W, __U, __A);
}

__m512bh test_mm512_maskz_rsqrt_pbh(__mmask32 __U, __m512bh __A) {
  // CHECK-LABEL: @test_mm512_maskz_rsqrt_pbh
  // CHECK: @llvm.x86.avx10.mask.rsqrt.bf16.512
  return _mm512_maskz_rsqrt_pbh(__U, __A);
}

__m512bh test_mm512_reduce_pbh(__m512bh __A) {
  // CHECK-LABEL: @test_mm512_reduce_pbh
  // CHECK: @llvm.x86.avx10.mask.reduce.bf16.512
  return _mm512_reduce_pbh(__A, 3);
}

__m512bh test_mm512_mask_reduce_pbh(__m512bh __W, __mmask16 __U, __m512bh __A) {
  // CHECK-LABEL: @test_mm512_mask_reduce_pbh
  // CHECK: @llvm.x86.avx10.mask.reduce.bf16.512
  return _mm512_mask_reduce_pbh(__W, __U, __A, 1);
}

__m512bh test_mm512_maskz_reduce_pbh(__mmask16 __U, __m512bh __A) {
  // CHECK-LABEL: @test_mm512_maskz_reduce_pbh
  // CHECK: @llvm.x86.avx10.mask.reduce.bf16.512
  return _mm512_maskz_reduce_pbh(__U, __A, 1);
}

__m512bh test_mm512_roundscale_pbh(__m512bh __A) {
  // CHECK-LABEL: @test_mm512_roundscale_pbh
  // CHECK: @llvm.x86.avx10.mask.rndscale.bf16.512
  return _mm512_roundscale_pbh(__A, 3);
}

__m512bh test_mm512_mask_roundscale_pbh(__m512bh __W, __mmask16 __U, __m512bh __A) {
  // CHECK-LABEL: @test_mm512_mask_roundscale_pbh
  // CHECK: @llvm.x86.avx10.mask.rndscale.bf16.512
  return _mm512_mask_roundscale_pbh(__W, __U, __A, 1);
}

__m512bh test_mm512_maskz_roundscale_pbh(__mmask16 __U, __m512bh __A) {
  // CHECK-LABEL: @test_mm512_maskz_roundscale_pbh
  // CHECK: @llvm.x86.avx10.mask.rndscale.bf16.512
  return _mm512_maskz_roundscale_pbh(__U, __A, 1 );
}

__m512bh test_mm512_getmant_pbh(__m512bh __A) {
  // CHECK-LABEL: @test_mm512_getmant_pbh
  // CHECK: @llvm.x86.avx10.mask.getmant.bf16.512
  return _mm512_getmant_pbh(__A, _MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan);
}

__m512bh test_mm512_mask_getmant_pbh(__m512bh __W, __mmask32 __U, __m512bh __A) {
  // CHECK-LABEL: @test_mm512_mask_getmant_pbh
  // CHECK: @llvm.x86.avx10.mask.getmant.bf16.512
  return _mm512_mask_getmant_pbh(__W, __U, __A, _MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan);
}

__m512bh test_mm512_maskz_getmant_pbh(__mmask32 __U, __m512bh __A) {
  // CHECK-LABEL: @test_mm512_maskz_getmant_pbh
  // CHECK: @llvm.x86.avx10.mask.getmant.bf16.512
  return _mm512_maskz_getmant_pbh(__U, __A, _MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan);
}

__m512bh test_mm512_sqrt_pbh(__m512bh __A) {
  // CHECK-LABEL: @test_mm512_sqrt_pbh
  // CHECK: %{{.*}} = call <32 x bfloat> @llvm.sqrt.v32bf16(<32 x bfloat> %{{.*}})
  return _mm512_sqrt_pbh(__A);
}

__m512bh test_mm512_mask_sqrt_pbh(__m512bh __W, __mmask32 __U, __m512bh __A) {
  // CHECK-LABEL: @test_mm512_mask_sqrt_pbh
  // CHECK: %{{.*}} = call <32 x bfloat> @llvm.sqrt.v32bf16(<32 x bfloat> %{{.*}})
  // CHECK: bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: select <32 x i1> %{{.*}}, <32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}}
  return (__m512bh)_mm512_mask_sqrt_pbh(__W, __U, __A);
}

__m512bh test_mm512_maskz_sqrt_pbh(__mmask32 __U, __m512bh __A) {
  // CHECK-LABEL: @test_mm512_maskz_sqrt_pbh
  // CHECK: %{{.*}} = call <32 x bfloat> @llvm.sqrt.v32bf16(<32 x bfloat> %{{.*}})
  // CHECK: bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: select <32 x i1> %{{.*}}, <32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}}
  return _mm512_maskz_sqrt_pbh(__U, __A);
}

__m512bh test_mm512_fmadd_pbh(__m512bh __A, __m512bh __B, __m512bh __C) {
  // CHECK-LABEL: @test_mm512_fmadd_pbh
  // CHECK: call <32 x bfloat> @llvm.fma.v32bf16(<32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}})
  return _mm512_fmadd_pbh(__A, __B, __C);
}

__m512bh test_mm512_mask_fmadd_pbh(__m512bh __A, __mmask32 __U, __m512bh __B, __m512bh __C) {
  // CHECK-LABEL: @test_mm512_mask_fmadd_pbh
  // CHECK: call <32 x bfloat> @llvm.fma.v32bf16(<32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}})
  // CHECK: select <32 x i1> %{{.*}}, <32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}}
  return _mm512_mask_fmadd_pbh(__A, __U, __B, __C);
}

__m512bh test_mm512_mask3_fmadd_pbh(__m512bh __A, __m512bh __B, __m512bh __C, __mmask32 __U) {
  // CHECK-LABEL: @test_mm512_mask3_fmadd_pbh
  // CHECK: call <32 x bfloat> @llvm.fma.v32bf16(<32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}})
  // CHECK: select <32 x i1> %{{.*}}, <32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}}
  return _mm512_mask3_fmadd_pbh(__A, __B, __C, __U);
}

__m512bh test_mm512_maskz_fmadd_pbh(__mmask32 __U, __m512bh __A, __m512bh __B, __m512bh __C) {
  // CHECK-LABEL: @test_mm512_maskz_fmadd_pbh
  // CHECK: call <32 x bfloat> @llvm.fma.v32bf16(<32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}})
  // CHECK: select <32 x i1> %{{.*}}, <32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}}
  return _mm512_maskz_fmadd_pbh(__U, __A, __B, __C);
}

__m512bh test_mm512_fmsub_pbh(__m512bh __A, __m512bh __B, __m512bh __C) {
  // CHECK-LABEL: @test_mm512_fmsub_pbh
  // CHECK: fneg
  // CHECK: call <32 x bfloat> @llvm.fma.v32bf16(<32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}})
  return _mm512_fmsub_pbh(__A, __B, __C);
}

__m512bh test_mm512_mask_fmsub_pbh(__m512bh __A, __mmask32 __U, __m512bh __B, __m512bh __C) {
  // CHECK-LABEL: @test_mm512_mask_fmsub_pbh
  // CHECK: fneg
  // CHECK: call <32 x bfloat> @llvm.fma.v32bf16(<32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}})
  // CHECK: select <32 x i1> %{{.*}}, <32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}}
  return _mm512_mask_fmsub_pbh(__A, __U, __B, __C);
}

__m512bh test_mm512_mask3_fmsub_pbh(__m512bh __A, __m512bh __B, __m512bh __C, __mmask32 __U) {
  // CHECK-LABEL: @test_mm512_mask3_fmsub_pbh
  // CHECK: fneg
  // CHECK: call <32 x bfloat> @llvm.fma.v32bf16(<32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}})
  // CHECK: select <32 x i1> %{{.*}}, <32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}}
  return _mm512_mask3_fmsub_pbh(__A, __B, __C, __U);
}

__m512bh test_mm512_maskz_fmsub_pbh(__mmask32 __U, __m512bh __A, __m512bh __B, __m512bh __C) {
  // CHECK-LABEL: @test_mm512_maskz_fmsub_pbh
  // CHECK: fneg
  // CHECK: call <32 x bfloat> @llvm.fma.v32bf16(<32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}})
  // CHECK: select <32 x i1> %{{.*}}, <32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}}
  return _mm512_maskz_fmsub_pbh(__U, __A, __B, __C);
}

__m512bh test_mm512_fnmadd_pbh(__m512bh __A, __m512bh __B, __m512bh __C) {
  // CHECK-LABEL: @test_mm512_fnmadd_pbh
  // CHECK: fneg
  // CHECK: call <32 x bfloat> @llvm.fma.v32bf16(<32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}})
  return _mm512_fnmadd_pbh(__A, __B, __C);
}

__m512bh test_mm512_mask_fnmadd_pbh(__m512bh __A, __mmask32 __U, __m512bh __B, __m512bh __C) {
  // CHECK-LABEL: @test_mm512_mask_fnmadd_pbh
  // CHECK: fneg
  // CHECK: call <32 x bfloat> @llvm.fma.v32bf16(<32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}})
  // CHECK: select <32 x i1> %{{.*}}, <32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}}
  return _mm512_mask_fnmadd_pbh(__A, __U, __B, __C);
}

__m512bh test_mm512_mask3_fnmadd_pbh(__m512bh __A, __m512bh __B, __m512bh __C, __mmask32 __U) {
  // CHECK-LABEL: @test_mm512_mask3_fnmadd_pbh
  // CHECK: fneg
  // CHECK: call <32 x bfloat> @llvm.fma.v32bf16(<32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}})
  // CHECK: select <32 x i1> %{{.*}}, <32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}}
  return _mm512_mask3_fnmadd_pbh(__A, __B, __C, __U);
}

__m512bh test_mm512_maskz_fnmadd_pbh(__mmask32 __U, __m512bh __A, __m512bh __B, __m512bh __C) {
  // CHECK-LABEL: @test_mm512_maskz_fnmadd_pbh
  // CHECK: fneg
  // CHECK: call <32 x bfloat> @llvm.fma.v32bf16(<32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}})
  // CHECK: select <32 x i1> %{{.*}}, <32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}}
  return _mm512_maskz_fnmadd_pbh(__U, __A, __B, __C);
}

__m512bh test_mm512_fnmsub_pbh(__m512bh __A, __m512bh __B, __m512bh __C) {
  // CHECK-LABEL: @test_mm512_fnmsub_pbh
  // CHECK: fneg
  // CHECK: fneg
  // CHECK: call <32 x bfloat> @llvm.fma.v32bf16(<32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}})
  return _mm512_fnmsub_pbh(__A, __B, __C);
}

__m512bh test_mm512_mask_fnmsub_pbh(__m512bh __A, __mmask32 __U, __m512bh __B, __m512bh __C) {
  // CHECK-LABEL: @test_mm512_mask_fnmsub_pbh
  // CHECK: fneg
  // CHECK: fneg
  // CHECK: call <32 x bfloat> @llvm.fma.v32bf16(<32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}})
  // CHECK: select <32 x i1> %{{.*}}, <32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}}
  return _mm512_mask_fnmsub_pbh(__A, __U, __B, __C);
}

__m512bh test_mm512_mask3_fnmsub_pbh(__m512bh __A, __m512bh __B, __m512bh __C, __mmask32 __U) {
  // CHECK-LABEL: @test_mm512_mask3_fnmsub_pbh
  // CHECK: fneg
  // CHECK: fneg
  // CHECK: call <32 x bfloat> @llvm.fma.v32bf16(<32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}})
  // CHECK: select <32 x i1> %{{.*}}, <32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}}
  return _mm512_mask3_fnmsub_pbh(__A, __B, __C, __U);
}

__m512bh test_mm512_maskz_fnmsub_pbh(__mmask32 __U, __m512bh __A, __m512bh __B, __m512bh __C) {
  // CHECK-LABEL: @test_mm512_maskz_fnmsub_pbh
  // CHECK: fneg
  // CHECK: fneg
  // CHECK: call <32 x bfloat> @llvm.fma.v32bf16(<32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}})
  // CHECK: select <32 x i1> %{{.*}}, <32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}}
  return _mm512_maskz_fnmsub_pbh(__U, __A, __B, __C);
}

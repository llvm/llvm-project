// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-unknown -target-feature +avx512vl -target-feature +avx512fp16 -emit-llvm -o - -Wall -Werror | FileCheck %s

#include <immintrin.h>

_Float16 test_mm_cvtsh_h(__m128h __A) {
  // CHECK-LABEL: @test_mm_cvtsh_h
  // CHECK: extractelement <8 x half> %{{.*}}, i32 0
  return _mm_cvtsh_h(__A);
}

_Float16 test_mm256_cvtsh_h(__m256h __A) {
  // CHECK-LABEL: @test_mm256_cvtsh_h
  // CHECK: extractelement <16 x half> %{{.*}}, i32 0
  return _mm256_cvtsh_h(__A);
}

__m128h test_mm_set_sh(_Float16 __h) {
  // CHECK-LABEL: @test_mm_set_sh
  // CHECK: insertelement <8 x half> {{.*}}, i32 0
  // CHECK: insertelement <8 x half> %{{.*}}, half 0xH0000, i32 1
  // CHECK: insertelement <8 x half> %{{.*}}, half 0xH0000, i32 2
  // CHECK: insertelement <8 x half> %{{.*}}, half 0xH0000, i32 3
  // CHECK: insertelement <8 x half> %{{.*}}, half 0xH0000, i32 4
  // CHECK: insertelement <8 x half> %{{.*}}, half 0xH0000, i32 5
  // CHECK: insertelement <8 x half> %{{.*}}, half 0xH0000, i32 6
  // CHECK: insertelement <8 x half> %{{.*}}, half 0xH0000, i32 7
  return _mm_set_sh(__h);
}

__m128h test_mm_set1_ph(_Float16 h) {
  // CHECK-LABEL: @test_mm_set1_ph
  // CHECK: insertelement <8 x half> {{.*}}, i32 0
  // CHECK: insertelement <8 x half> {{.*}}, i32 1
  // CHECK: insertelement <8 x half> {{.*}}, i32 2
  // CHECK: insertelement <8 x half> {{.*}}, i32 3
  // CHECK: insertelement <8 x half> {{.*}}, i32 4
  // CHECK: insertelement <8 x half> {{.*}}, i32 5
  // CHECK: insertelement <8 x half> {{.*}}, i32 6
  // CHECK: insertelement <8 x half> {{.*}}, i32 7
  return _mm_set1_ph(h);
}

__m256h test_mm256_set1_ph(_Float16 h) {
  // CHECK-LABEL: @test_mm256_set1_ph
  // CHECK: insertelement <16 x half> {{.*}}, i32 0
  // CHECK: insertelement <16 x half> {{.*}}, i32 1
  // CHECK: insertelement <16 x half> {{.*}}, i32 2
  // CHECK: insertelement <16 x half> {{.*}}, i32 3
  // CHECK: insertelement <16 x half> {{.*}}, i32 4
  // CHECK: insertelement <16 x half> {{.*}}, i32 5
  // CHECK: insertelement <16 x half> {{.*}}, i32 6
  // CHECK: insertelement <16 x half> {{.*}}, i32 7
  // CHECK: insertelement <16 x half> {{.*}}, i32 8
  // CHECK: insertelement <16 x half> {{.*}}, i32 9
  // CHECK: insertelement <16 x half> {{.*}}, i32 10
  // CHECK: insertelement <16 x half> {{.*}}, i32 11
  // CHECK: insertelement <16 x half> {{.*}}, i32 12
  // CHECK: insertelement <16 x half> {{.*}}, i32 13
  // CHECK: insertelement <16 x half> {{.*}}, i32 14
  // CHECK: insertelement <16 x half> {{.*}}, i32 15
  return _mm256_set1_ph(h);
}

__m128h test_mm_set_ph(_Float16 __h1, _Float16 __h2, _Float16 __h3, _Float16 __h4,
                       _Float16 __h5, _Float16 __h6, _Float16 __h7, _Float16 __h8) {
  // CHECK-LABEL: @test_mm_set_ph
  // CHECK: insertelement <8 x half> {{.*}}, i32 0
  // CHECK: insertelement <8 x half> {{.*}}, i32 1
  // CHECK: insertelement <8 x half> {{.*}}, i32 2
  // CHECK: insertelement <8 x half> {{.*}}, i32 3
  // CHECK: insertelement <8 x half> {{.*}}, i32 4
  // CHECK: insertelement <8 x half> {{.*}}, i32 5
  // CHECK: insertelement <8 x half> {{.*}}, i32 6
  // CHECK: insertelement <8 x half> {{.*}}, i32 7
  return _mm_set_ph(__h1, __h2, __h3, __h4, __h5, __h6, __h7, __h8);
}

__m256h test_mm256_set_ph(_Float16 __h1, _Float16 __h2, _Float16 __h3, _Float16 __h4,
                          _Float16 __h5, _Float16 __h6, _Float16 __h7, _Float16 __h8,
                          _Float16 __h9, _Float16 __h10, _Float16 __h11, _Float16 __h12,
                          _Float16 __h13, _Float16 __h14, _Float16 __h15, _Float16 __h16) {
  // CHECK-LABEL: @test_mm256_set_ph
  // CHECK: insertelement <16 x half> {{.*}}, i32 0
  // CHECK: insertelement <16 x half> {{.*}}, i32 1
  // CHECK: insertelement <16 x half> {{.*}}, i32 2
  // CHECK: insertelement <16 x half> {{.*}}, i32 3
  // CHECK: insertelement <16 x half> {{.*}}, i32 4
  // CHECK: insertelement <16 x half> {{.*}}, i32 5
  // CHECK: insertelement <16 x half> {{.*}}, i32 6
  // CHECK: insertelement <16 x half> {{.*}}, i32 7
  // CHECK: insertelement <16 x half> {{.*}}, i32 8
  // CHECK: insertelement <16 x half> {{.*}}, i32 9
  // CHECK: insertelement <16 x half> {{.*}}, i32 10
  // CHECK: insertelement <16 x half> {{.*}}, i32 11
  // CHECK: insertelement <16 x half> {{.*}}, i32 12
  // CHECK: insertelement <16 x half> {{.*}}, i32 13
  // CHECK: insertelement <16 x half> {{.*}}, i32 14
  // CHECK: insertelement <16 x half> {{.*}}, i32 15
  return _mm256_set_ph(__h1, __h2, __h3, __h4, __h5, __h6, __h7, __h8,
                       __h9, __h10, __h11, __h12, __h13, __h14, __h15, __h16);
}

__m128h test_mm_setr_ph(_Float16 __h1, _Float16 __h2, _Float16 __h3, _Float16 __h4,
                        _Float16 __h5, _Float16 __h6, _Float16 __h7, _Float16 __h8) {
  // CHECK-LABEL: @test_mm_setr_ph
  // CHECK: insertelement <8 x half> {{.*}}, i32 0
  // CHECK: insertelement <8 x half> {{.*}}, i32 1
  // CHECK: insertelement <8 x half> {{.*}}, i32 2
  // CHECK: insertelement <8 x half> {{.*}}, i32 3
  // CHECK: insertelement <8 x half> {{.*}}, i32 4
  // CHECK: insertelement <8 x half> {{.*}}, i32 5
  // CHECK: insertelement <8 x half> {{.*}}, i32 6
  // CHECK: insertelement <8 x half> {{.*}}, i32 7
  return _mm_setr_ph(__h1, __h2, __h3, __h4, __h5, __h6, __h7, __h8);
}

__m256h test_mm256_setr_ph(_Float16 __h1, _Float16 __h2, _Float16 __h3, _Float16 __h4,
                           _Float16 __h5, _Float16 __h6, _Float16 __h7, _Float16 __h8,
                           _Float16 __h9, _Float16 __h10, _Float16 __h11, _Float16 __h12,
                           _Float16 __h13, _Float16 __h14, _Float16 __h15, _Float16 __h16) {
  // CHECK-LABEL: @test_mm256_setr_ph
  // CHECK: insertelement <16 x half> {{.*}}, i32 0
  // CHECK: insertelement <16 x half> {{.*}}, i32 1
  // CHECK: insertelement <16 x half> {{.*}}, i32 2
  // CHECK: insertelement <16 x half> {{.*}}, i32 3
  // CHECK: insertelement <16 x half> {{.*}}, i32 4
  // CHECK: insertelement <16 x half> {{.*}}, i32 5
  // CHECK: insertelement <16 x half> {{.*}}, i32 6
  // CHECK: insertelement <16 x half> {{.*}}, i32 7
  // CHECK: insertelement <16 x half> {{.*}}, i32 8
  // CHECK: insertelement <16 x half> {{.*}}, i32 9
  // CHECK: insertelement <16 x half> {{.*}}, i32 10
  // CHECK: insertelement <16 x half> {{.*}}, i32 11
  // CHECK: insertelement <16 x half> {{.*}}, i32 12
  // CHECK: insertelement <16 x half> {{.*}}, i32 13
  // CHECK: insertelement <16 x half> {{.*}}, i32 14
  // CHECK: insertelement <16 x half> {{.*}}, i32 15
  return _mm256_setr_ph(__h1, __h2, __h3, __h4, __h5, __h6, __h7, __h8,
                        __h9, __h10, __h11, __h12, __h13, __h14, __h15, __h16);
}

__m128h test_mm_abs_ph(__m128h a) {
  // CHECK-LABEL: @test_mm_abs_ph
  // CHECK: and <4 x i32>
  return _mm_abs_ph(a);
}

__m256h test_mm256_abs_ph(__m256h a) {
  // CHECK-LABEL: @test_mm256_abs_ph
  // CHECK: and <8 x i32>
  return _mm256_abs_ph(a);
}

__m128h test_mm_mask_blend_ph(__mmask8 __U, __m128h __A, __m128h __W) {
  // CHECK-LABEL: @test_mm_mask_blend_ph
  // CHECK:  %{{.*}} = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK:  %{{.*}} = select <8 x i1> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}}
  return _mm_mask_blend_ph(__U, __A, __W);
}

__m256h test_mm256_mask_blend_ph(__mmask16 __U, __m256h __A, __m256h __W) {
  // CHECK-LABEL: @test_mm256_mask_blend_ph
  // CHECK:  %{{.*}} = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK:  %{{.*}} = select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return _mm256_mask_blend_ph(__U, __A, __W);
}

__m128h test_mm_permutex2var_ph(__m128h __A, __m128i __I, __m128h __B) {
  // CHECK-LABEL: @test_mm_permutex2var_ph
  // CHECK:  %{{.*}} = bitcast <8 x half> %{{.*}} to <8 x i16>
  // CHECK:  %{{.*}} = bitcast <2 x i64> %{{.*}} to <8 x i16>
  // CHECK:  %{{.*}} = bitcast <8 x half> %{{.*}} to <8 x i16>
  // CHECK:  %{{.*}} = call <8 x i16> @llvm.x86.avx512.vpermi2var.hi.128(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK:  %{{.*}} = bitcast <8 x i16> %{{.*}} to <8 x half>
  return _mm_permutex2var_ph(__A, __I, __B);
}

__m256h test_mm256_permutex2var_ph(__m256h __A, __m256i __I, __m256h __B) {
  // CHECK-LABEL: @test_mm256_permutex2var_ph
  // CHECK:  %{{.*}} = bitcast <16 x half> %{{.*}} to <16 x i16>
  // CHECK:  %{{.*}} = bitcast <4 x i64> %{{.*}} to <16 x i16>
  // CHECK:  %{{.*}} = bitcast <16 x half> %{{.*}} to <16 x i16>
  // CHECK:  %{{.*}} = call <16 x i16> @llvm.x86.avx512.vpermi2var.hi.256(<16 x i16> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  // CHECK:  %{{.*}} = bitcast <16 x i16> %{{.*}} to <16 x half>
  return _mm256_permutex2var_ph(__A, __I, __B);
}

__m128h test_mm_permutexvar_ph(__m128i __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_permutexvar_ph
  // CHECK:  %{{.*}} = bitcast <8 x half> %{{.*}} to <8 x i16>
  // CHECK:  %{{.*}} = bitcast <2 x i64> %{{.*}} to <8 x i16>
  // CHECK:  %{{.*}} = call <8 x i16> @llvm.x86.avx512.permvar.hi.128(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK:  %{{.*}} = bitcast <8 x i16> %{{.*}} to <8 x half>
  return _mm_permutexvar_ph(__A, __B);
}

__m256h test_mm256_permutexvar_ph(__m256i __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_permutexvar_ph
  // CHECK:  %{{.*}} = bitcast <16 x half> %{{.*}} to <16 x i16>
  // CHECK:  %{{.*}} = bitcast <4 x i64> %{{.*}} to <16 x i16>
  // CHECK:  %{{.*}} = call <16 x i16> @llvm.x86.avx512.permvar.hi.256(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  // CHECK:  %{{.*}} = bitcast <16 x i16> %{{.*}} to <16 x half>
  return _mm256_permutexvar_ph(__A, __B);
}

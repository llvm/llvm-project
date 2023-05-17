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

__m128h test_mm_set1_pch(_Float16 _Complex h) {
  // CHECK-LABEL: @test_mm_set1_pch
  // CHECK: insertelement <4 x float> {{.*}}, i32 0
  // CHECK: insertelement <4 x float> {{.*}}, i32 1
  // CHECK: insertelement <4 x float> {{.*}}, i32 2
  // CHECK: insertelement <4 x float> {{.*}}, i32 3
  return _mm_set1_pch(h);
}

__m256h test_mm256_set1_pch(_Float16 _Complex h) {
  // CHECK-LABEL: @test_mm256_set1_pch
  // CHECK: insertelement <8 x float> {{.*}}, i32 0
  // CHECK: insertelement <8 x float> {{.*}}, i32 1
  // CHECK: insertelement <8 x float> {{.*}}, i32 2
  // CHECK: insertelement <8 x float> {{.*}}, i32 3
  // CHECK: insertelement <8 x float> {{.*}}, i32 4
  // CHECK: insertelement <8 x float> {{.*}}, i32 5
  // CHECK: insertelement <8 x float> {{.*}}, i32 6
  // CHECK: insertelement <8 x float> {{.*}}, i32 7
  return _mm256_set1_pch(h);
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

__m256h test_mm256_add_ph(__m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_add_ph
  // CHECK: %{{.*}} = fadd <16 x half> %{{.*}}, %{{.*}}
  return _mm256_add_ph(__A, __B);
}

__m256h test_mm256_mask_add_ph(__m256h __W, __mmask32 __U, __m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_mask_add_ph
  // CHECK: %{{.*}} = fadd <16 x half> %{{.*}}, %{{.*}}
  // CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return (__m256h)_mm256_mask_add_ph(__W, __U, __A, __B);
}

__m256h test_mm256_maskz_add_ph(__mmask32 __U, __m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_maskz_add_ph
  // CHECK: %{{.*}} = fadd <16 x half> %{{.*}}, %{{.*}}
  // CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return _mm256_maskz_add_ph(__U, __A, __B);
}

__m128h test_mm_add_ph(__m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_add_ph
  // CHECK: %{{.*}} = fadd <8 x half> %{{.*}}, %{{.*}}
  return _mm_add_ph(__A, __B);
}

__m128h test_mm_mask_add_ph(__m128h __W, __mmask32 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_mask_add_ph
  // CHECK: %{{.*}} = fadd <8 x half> %{{.*}}, %{{.*}}
  // CHECK: select <8 x i1> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}}
  return (__m128h)_mm_mask_add_ph(__W, __U, __A, __B);
}

__m128h test_mm_maskz_add_ph(__mmask32 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_maskz_add_ph
  // CHECK: %{{.*}} = fadd <8 x half> %{{.*}}, %{{.*}}
  // CHECK: select <8 x i1> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}}
  return _mm_maskz_add_ph(__U, __A, __B);
}

__m256h test_mm256_sub_ph(__m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_sub_ph
  // CHECK: %{{.*}} = fsub <16 x half> %{{.*}}, %{{.*}}
  return _mm256_sub_ph(__A, __B);
}

__m256h test_mm256_mask_sub_ph(__m256h __W, __mmask32 __U, __m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_mask_sub_ph
  // CHECK: %{{.*}} = fsub <16 x half> %{{.*}}, %{{.*}}
  // CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return (__m256h)_mm256_mask_sub_ph(__W, __U, __A, __B);
}

__m256h test_mm256_maskz_sub_ph(__mmask32 __U, __m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_maskz_sub_ph
  // CHECK: %{{.*}} = fsub <16 x half> %{{.*}}, %{{.*}}
  // CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return _mm256_maskz_sub_ph(__U, __A, __B);
}

__m128h test_mm_sub_ph(__m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_sub_ph
  // CHECK: %{{.*}} = fsub <8 x half> %{{.*}}, %{{.*}}
  return _mm_sub_ph(__A, __B);
}

__m128h test_mm_mask_sub_ph(__m128h __W, __mmask32 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_mask_sub_ph
  // CHECK: %{{.*}} = fsub <8 x half> %{{.*}}, %{{.*}}
  // CHECK: select <8 x i1> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}}
  return (__m128h)_mm_mask_sub_ph(__W, __U, __A, __B);
}

__m128h test_mm_maskz_sub_ph(__mmask32 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_maskz_sub_ph
  // CHECK: %{{.*}} = fsub <8 x half> %{{.*}}, %{{.*}}
  // CHECK: select <8 x i1> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}}
  return _mm_maskz_sub_ph(__U, __A, __B);
}

__m256h test_mm256_mul_ph(__m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_mul_ph
  // CHECK: %{{.*}} = fmul <16 x half> %{{.*}}, %{{.*}}
  return _mm256_mul_ph(__A, __B);
}

__m256h test_mm256_mask_mul_ph(__m256h __W, __mmask32 __U, __m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_mask_mul_ph
  // CHECK: %{{.*}} = fmul <16 x half> %{{.*}}, %{{.*}}
  // CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return (__m256h)_mm256_mask_mul_ph(__W, __U, __A, __B);
}

__m256h test_mm256_maskz_mul_ph(__mmask32 __U, __m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_maskz_mul_ph
  // CHECK: %{{.*}} = fmul <16 x half> %{{.*}}, %{{.*}}
  // CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return _mm256_maskz_mul_ph(__U, __A, __B);
}

__m128h test_mm_mul_ph(__m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_mul_ph
  // CHECK: %{{.*}} = fmul <8 x half> %{{.*}}, %{{.*}}
  return _mm_mul_ph(__A, __B);
}

__m128h test_mm_mask_mul_ph(__m128h __W, __mmask32 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_mask_mul_ph
  // CHECK: %{{.*}} = fmul <8 x half> %{{.*}}, %{{.*}}
  // CHECK: select <8 x i1> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}}
  return (__m128h)_mm_mask_mul_ph(__W, __U, __A, __B);
}

__m128h test_mm_maskz_mul_ph(__mmask32 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_maskz_mul_ph
  // CHECK: %{{.*}} = fmul <8 x half> %{{.*}}, %{{.*}}
  // CHECK: select <8 x i1> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}}
  return _mm_maskz_mul_ph(__U, __A, __B);
}

__m256h test_mm256_div_ph(__m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_div_ph
  // CHECK: %{{.*}} = fdiv <16 x half> %{{.*}}, %{{.*}}
  return _mm256_div_ph(__A, __B);
}

__m256h test_mm256_mask_div_ph(__m256h __W, __mmask32 __U, __m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_mask_div_ph
  // CHECK: %{{.*}} = fdiv <16 x half> %{{.*}}, %{{.*}}
  // CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return (__m256h)_mm256_mask_div_ph(__W, __U, __A, __B);
}

__m256h test_mm256_maskz_div_ph(__mmask32 __U, __m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_maskz_div_ph
  // CHECK: %{{.*}} = fdiv <16 x half> %{{.*}}, %{{.*}}
  // CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return _mm256_maskz_div_ph(__U, __A, __B);
}

__m128h test_mm_div_ph(__m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_div_ph
  // CHECK: %{{.*}} = fdiv <8 x half> %{{.*}}, %{{.*}}
  return _mm_div_ph(__A, __B);
}

__m128h test_mm_mask_div_ph(__m128h __W, __mmask32 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_mask_div_ph
  // CHECK: %{{.*}} = fdiv <8 x half> %{{.*}}, %{{.*}}
  // CHECK: select <8 x i1> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}}
  return (__m128h)_mm_mask_div_ph(__W, __U, __A, __B);
}

__m128h test_mm_maskz_div_ph(__mmask32 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_maskz_div_ph
  // CHECK: %{{.*}} = fdiv <8 x half> %{{.*}}, %{{.*}}
  // CHECK: select <8 x i1> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}}
  return _mm_maskz_div_ph(__U, __A, __B);
}

__m256h test_mm256_min_ph(__m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_min_ph
  // CHECK: @llvm.x86.avx512fp16.min.ph.256
  return _mm256_min_ph(__A, __B);
}

__m256h test_mm256_mask_min_ph(__m256h __W, __mmask32 __U, __m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_mask_min_ph
  // CHECK: @llvm.x86.avx512fp16.min.ph.256
  return (__m256h)_mm256_mask_min_ph(__W, __U, __A, __B);
}

__m256h test_mm256_maskz_min_ph(__mmask32 __U, __m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_maskz_min_ph
  // CHECK: @llvm.x86.avx512fp16.min.ph.256
  return _mm256_maskz_min_ph(__U, __A, __B);
}

__m128h test_mm_min_ph(__m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_min_ph
  // CHECK: @llvm.x86.avx512fp16.min.ph.128
  return _mm_min_ph(__A, __B);
}

__m128h test_mm_mask_min_ph(__m128h __W, __mmask32 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_mask_min_ph
  // CHECK: @llvm.x86.avx512fp16.min.ph.128
  return (__m128h)_mm_mask_min_ph(__W, __U, __A, __B);
}

__m128h test_mm_maskz_min_ph(__mmask32 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_maskz_min_ph
  // CHECK: @llvm.x86.avx512fp16.min.ph.128
  return _mm_maskz_min_ph(__U, __A, __B);
}

__m256h test_mm256_max_ph(__m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_max_ph
  // CHECK: @llvm.x86.avx512fp16.max.ph.256
  return _mm256_max_ph(__A, __B);
}

__m256h test_mm256_mask_max_ph(__m256h __W, __mmask32 __U, __m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_mask_max_ph
  // CHECK: @llvm.x86.avx512fp16.max.ph.256
  return (__m256h)_mm256_mask_max_ph(__W, __U, __A, __B);
}

__m256h test_mm256_maskz_max_ph(__mmask32 __U, __m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_maskz_max_ph
  // CHECK: @llvm.x86.avx512fp16.max.ph.256
  return _mm256_maskz_max_ph(__U, __A, __B);
}

__m128h test_mm_max_ph(__m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_max_ph
  // CHECK: @llvm.x86.avx512fp16.max.ph.128
  return _mm_max_ph(__A, __B);
}

__m128h test_mm_mask_max_ph(__m128h __W, __mmask32 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_mask_max_ph
  // CHECK: @llvm.x86.avx512fp16.max.ph.128
  return (__m128h)_mm_mask_max_ph(__W, __U, __A, __B);
}

__m128h test_mm_maskz_max_ph(__mmask32 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_maskz_max_ph
  // CHECK: @llvm.x86.avx512fp16.max.ph.128
  return _mm_maskz_max_ph(__U, __A, __B);
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

__m256h test_mm256_conj_pch(__m256h __A) {
  // CHECK-LABEL: @test_mm256_conj_pch
  // CHECK:  %{{.*}} = bitcast <16 x half> %{{.*}} to <8 x float>
  // CHECK:  %{{.*}} = bitcast <8 x float> %{{.*}} to <8 x i32>
  // CHECK:  %{{.*}} = bitcast <8 x float> %{{.*}} to <8 x i32>
  // CHECK:  %{{.*}} = xor <8 x i32> %{{.*}}, %{{.*}}
  // CHECK:  %{{.*}} = bitcast <8 x i32> %{{.*}} to <8 x float>
  // CHECK:  %{{.*}} = bitcast <8 x float> %{{.*}} to <16 x half>
  return _mm256_conj_pch(__A);
}

__m256h test_mm256_mask_conj_pch(__m256h __W, __mmask32 __U, __m256h __A) {
  // CHECK-LABEL: @test_mm256_mask_conj_pch
  // CHECK:  %{{.*}} = trunc i32 %{{.*}} to i8
  // CHECK:  %{{.*}} = bitcast <16 x half> %{{.*}} to <8 x float>
  // CHECK:  %{{.*}} = bitcast <8 x float> %{{.*}} to <8 x i32>
  // CHECK:  %{{.*}} = bitcast <8 x float> %{{.*}} to <8 x i32>
  // CHECK:  %{{.*}} = xor <8 x i32> %{{.*}}, %{{.*}}
  // CHECK:  %{{.*}} = bitcast <8 x i32> %{{.*}} to <8 x float>
  // CHECK:  %{{.*}} = bitcast <8 x float> %{{.*}} to <16 x half>
  // CHECK:  %{{.*}} = bitcast <16 x half> %{{.*}} to <8 x float>
  // CHECK:  %{{.*}} = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK:  %{{.*}} = select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  // CHECK:  %{{.*}} = bitcast <8 x float> %{{.*}} to <16 x half>
  return _mm256_mask_conj_pch(__W, __U, __A);
}

__m256h test_mm256_maskz_conj_pch(__mmask32 __U, __m256h __A) {
  // CHECK-LABEL: @test_mm256_maskz_conj_pch
  // CHECK:  %{{.*}} = trunc i32 %{{.*}} to i8
  // CHECK:  %{{.*}} = bitcast <16 x half> %{{.*}} to <8 x float>
  // CHECK:  %{{.*}} = bitcast <8 x float> %{{.*}} to <8 x i32>
  // CHECK:  %{{.*}} = bitcast <8 x float> %{{.*}} to <8 x i32>
  // CHECK:  %{{.*}} = xor <8 x i32> %{{.*}}, %{{.*}}
  // CHECK:  %{{.*}} = bitcast <8 x i32> %{{.*}} to <8 x float>
  // CHECK:  %{{.*}} = bitcast <8 x float> %{{.*}} to <16 x half>
  // CHECK:  %{{.*}} = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK:  %{{.*}} = select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  // CHECK:  %{{.*}} = bitcast <8 x float> %{{.*}} to <16 x half>
  return _mm256_maskz_conj_pch(__U, __A);
}

__m128h test_mm_conj_pch(__m128h __A) {
  // CHECK-LABEL: @test_mm_conj_pch
  // CHECK:  %{{.*}} = bitcast <8 x half> %{{.*}} to <4 x float>
  // CHECK:  %{{.*}} = bitcast <4 x float> %{{.*}} to <4 x i32>
  // CHECK:  %{{.*}} = bitcast <4 x float> %{{.*}} to <4 x i32>
  // CHECK:  %{{.*}} = xor <4 x i32> %{{.*}}, %{{.*}}
  // CHECK:  %{{.*}} = bitcast <4 x i32> %{{.*}} to <4 x float>
  // CHECK:  %{{.*}} = bitcast <4 x float> %{{.*}} to <8 x half>
  return _mm_conj_pch(__A);
}

__m128h test_mm_mask_conj_pch(__m128h __W, __mmask32 __U, __m128h __A) {
  // CHECK-LABEL: @test_mm_mask_conj_pch
  // CHECK:  %{{.*}} = trunc i32 %{{.*}} to i8
  // CHECK:  %{{.*}} = bitcast <8 x half> %{{.*}} to <4 x float>
  // CHECK:  %{{.*}} = bitcast <4 x float> %{{.*}} to <4 x i32>
  // CHECK:  %{{.*}} = bitcast <4 x float> %{{.*}} to <4 x i32>
  // CHECK:  %{{.*}} = xor <4 x i32> %{{.*}}, %{{.*}}
  // CHECK:  %{{.*}} = bitcast <4 x i32> %{{.*}} to <4 x float>
  // CHECK:  %{{.*}} = bitcast <4 x float> %{{.*}} to <8 x half>
  // CHECK:  %{{.*}} = bitcast <8 x half> %{{.*}} to <4 x float>
  // CHECK:  %{{.*}} = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK:  %{{.*}} = shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK:  %{{.*}} = select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  // CHECK:  %{{.*}} = bitcast <4 x float> %{{.*}} to <8 x half>
  return _mm_mask_conj_pch(__W, __U, __A);
}

__m128h test_mm_maskz_conj_pch(__mmask32 __U, __m128h __A) {
  // CHECK-LABEL: @test_mm_maskz_conj_pch
  // CHECK:  %{{.*}} = trunc i32 %{{.*}} to i8
  // CHECK:  %{{.*}} = bitcast <8 x half> %{{.*}} to <4 x float>
  // CHECK:  %{{.*}} = bitcast <4 x float> %{{.*}} to <4 x i32>
  // CHECK:  %{{.*}} = bitcast <4 x float> %{{.*}} to <4 x i32>
  // CHECK:  %{{.*}} = xor <4 x i32> %{{.*}}, %{{.*}}
  // CHECK:  %{{.*}} = bitcast <4 x i32> %{{.*}} to <4 x float>
  // CHECK:  %{{.*}} = bitcast <4 x float> %{{.*}} to <8 x half>
  // CHECK:  %{{.*}} = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK:  %{{.*}} = shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK:  %{{.*}} = select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  // CHECK:  %{{.*}} = bitcast <4 x float> %{{.*}} to <8 x half>
  return _mm_maskz_conj_pch(__U, __A);
}

__mmask16 test_mm256_cmp_ph_mask_eq_oq(__m256h a, __m256h b) {
  // CHECK-LABEL: @test_mm256_cmp_ph_mask_eq_oq
  // CHECK: fcmp oeq <16 x half> %{{.*}}, %{{.*}}
  return _mm256_cmp_ph_mask(a, b, _CMP_EQ_OQ);
}

__mmask16 test_mm256_cmp_ph_mask_lt_os(__m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_cmp_ph_mask_lt_os
  // CHECK: fcmp olt <16 x half> %{{.*}}, %{{.*}}
  return _mm256_cmp_ph_mask(a, b, _CMP_LT_OS);
}

__mmask16 test_mm256_cmp_ph_mask_le_os(__m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_cmp_ph_mask_le_os
  // CHECK: fcmp ole <16 x half> %{{.*}}, %{{.*}}
  return _mm256_cmp_ph_mask(a, b, _CMP_LE_OS);
}

__mmask16 test_mm256_cmp_ph_mask_unord_q(__m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_cmp_ph_mask_unord_q
  // CHECK: fcmp uno <16 x half> %{{.*}}, %{{.*}}
  return _mm256_cmp_ph_mask(a, b, _CMP_UNORD_Q);
}

__mmask16 test_mm256_cmp_ph_mask_neq_uq(__m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_cmp_ph_mask_neq_uq
  // CHECK: fcmp une <16 x half> %{{.*}}, %{{.*}}
  return _mm256_cmp_ph_mask(a, b, _CMP_NEQ_UQ);
}

__mmask16 test_mm256_cmp_ph_mask_nlt_us(__m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_cmp_ph_mask_nlt_us
  // CHECK: fcmp uge <16 x half> %{{.*}}, %{{.*}}
  return _mm256_cmp_ph_mask(a, b, _CMP_NLT_US);
}

__mmask16 test_mm256_cmp_ph_mask_nle_us(__m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_cmp_ph_mask_nle_us
  // CHECK: fcmp ugt <16 x half> %{{.*}}, %{{.*}}
  return _mm256_cmp_ph_mask(a, b, _CMP_NLE_US);
}

__mmask16 test_mm256_cmp_ph_mask_ord_q(__m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_cmp_ph_mask_ord_q
  // CHECK: fcmp ord <16 x half> %{{.*}}, %{{.*}}
  return _mm256_cmp_ph_mask(a, b, _CMP_ORD_Q);
}

__mmask16 test_mm256_cmp_ph_mask_eq_uq(__m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_cmp_ph_mask_eq_uq
  // CHECK: fcmp ueq <16 x half> %{{.*}}, %{{.*}}
  return _mm256_cmp_ph_mask(a, b, _CMP_EQ_UQ);
}

__mmask16 test_mm256_cmp_ph_mask_nge_us(__m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_cmp_ph_mask_nge_us
  // CHECK: fcmp ult <16 x half> %{{.*}}, %{{.*}}
  return _mm256_cmp_ph_mask(a, b, _CMP_NGE_US);
}

__mmask16 test_mm256_cmp_ph_mask_ngt_us(__m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_cmp_ph_mask_ngt_us
  // CHECK: fcmp ule <16 x half> %{{.*}}, %{{.*}}
  return _mm256_cmp_ph_mask(a, b, _CMP_NGT_US);
}

__mmask16 test_mm256_cmp_ph_mask_false_oq(__m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_cmp_ph_mask_false_oq
  // CHECK: fcmp false <16 x half> %{{.*}}, %{{.*}}
  return _mm256_cmp_ph_mask(a, b, _CMP_FALSE_OQ);
}

__mmask16 test_mm256_cmp_ph_mask_neq_oq(__m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_cmp_ph_mask_neq_oq
  // CHECK: fcmp one <16 x half> %{{.*}}, %{{.*}}
  return _mm256_cmp_ph_mask(a, b, _CMP_NEQ_OQ);
}

__mmask16 test_mm256_cmp_ph_mask_ge_os(__m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_cmp_ph_mask_ge_os
  // CHECK: fcmp oge <16 x half> %{{.*}}, %{{.*}}
  return _mm256_cmp_ph_mask(a, b, _CMP_GE_OS);
}

__mmask16 test_mm256_cmp_ph_mask_gt_os(__m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_cmp_ph_mask_gt_os
  // CHECK: fcmp ogt <16 x half> %{{.*}}, %{{.*}}
  return _mm256_cmp_ph_mask(a, b, _CMP_GT_OS);
}

__mmask16 test_mm256_cmp_ph_mask_true_uq(__m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_cmp_ph_mask_true_uq
  // CHECK: fcmp true <16 x half> %{{.*}}, %{{.*}}
  return _mm256_cmp_ph_mask(a, b, _CMP_TRUE_UQ);
}

__mmask16 test_mm256_cmp_ph_mask_eq_os(__m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_cmp_ph_mask_eq_os
  // CHECK: fcmp oeq <16 x half> %{{.*}}, %{{.*}}
  return _mm256_cmp_ph_mask(a, b, _CMP_EQ_OS);
}

__mmask16 test_mm256_cmp_ph_mask_lt_oq(__m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_cmp_ph_mask_lt_oq
  // CHECK: fcmp olt <16 x half> %{{.*}}, %{{.*}}
  return _mm256_cmp_ph_mask(a, b, _CMP_LT_OQ);
}

__mmask16 test_mm256_cmp_ph_mask_le_oq(__m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_cmp_ph_mask_le_oq
  // CHECK: fcmp ole <16 x half> %{{.*}}, %{{.*}}
  return _mm256_cmp_ph_mask(a, b, _CMP_LE_OQ);
}

__mmask16 test_mm256_cmp_ph_mask_unord_s(__m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_cmp_ph_mask_unord_s
  // CHECK: fcmp uno <16 x half> %{{.*}}, %{{.*}}
  return _mm256_cmp_ph_mask(a, b, _CMP_UNORD_S);
}

__mmask16 test_mm256_cmp_ph_mask_neq_us(__m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_cmp_ph_mask_neq_us
  // CHECK: fcmp une <16 x half> %{{.*}}, %{{.*}}
  return _mm256_cmp_ph_mask(a, b, _CMP_NEQ_US);
}

__mmask16 test_mm256_cmp_ph_mask_nlt_uq(__m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_cmp_ph_mask_nlt_uq
  // CHECK: fcmp uge <16 x half> %{{.*}}, %{{.*}}
  return _mm256_cmp_ph_mask(a, b, _CMP_NLT_UQ);
}

__mmask16 test_mm256_cmp_ph_mask_nle_uq(__m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_cmp_ph_mask_nle_uq
  // CHECK: fcmp ugt <16 x half> %{{.*}}, %{{.*}}
  return _mm256_cmp_ph_mask(a, b, _CMP_NLE_UQ);
}

__mmask16 test_mm256_cmp_ph_mask_ord_s(__m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_cmp_ph_mask_ord_s
  // CHECK: fcmp ord <16 x half> %{{.*}}, %{{.*}}
  return _mm256_cmp_ph_mask(a, b, _CMP_ORD_S);
}

__mmask16 test_mm256_cmp_ph_mask_eq_us(__m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_cmp_ph_mask_eq_us
  // CHECK: fcmp ueq <16 x half> %{{.*}}, %{{.*}}
  return _mm256_cmp_ph_mask(a, b, _CMP_EQ_US);
}

__mmask16 test_mm256_cmp_ph_mask_nge_uq(__m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_cmp_ph_mask_nge_uq
  // CHECK: fcmp ult <16 x half> %{{.*}}, %{{.*}}
  return _mm256_cmp_ph_mask(a, b, _CMP_NGE_UQ);
}

__mmask16 test_mm256_cmp_ph_mask_ngt_uq(__m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_cmp_ph_mask_ngt_uq
  // CHECK: fcmp ule <16 x half> %{{.*}}, %{{.*}}
  return _mm256_cmp_ph_mask(a, b, _CMP_NGT_UQ);
}

__mmask16 test_mm256_cmp_ph_mask_false_os(__m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_cmp_ph_mask_false_os
  // CHECK: fcmp false <16 x half> %{{.*}}, %{{.*}}
  return _mm256_cmp_ph_mask(a, b, _CMP_FALSE_OS);
}

__mmask16 test_mm256_cmp_ph_mask_neq_os(__m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_cmp_ph_mask_neq_os
  // CHECK: fcmp one <16 x half> %{{.*}}, %{{.*}}
  return _mm256_cmp_ph_mask(a, b, _CMP_NEQ_OS);
}

__mmask16 test_mm256_cmp_ph_mask_ge_oq(__m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_cmp_ph_mask_ge_oq
  // CHECK: fcmp oge <16 x half> %{{.*}}, %{{.*}}
  return _mm256_cmp_ph_mask(a, b, _CMP_GE_OQ);
}

__mmask16 test_mm256_cmp_ph_mask_gt_oq(__m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_cmp_ph_mask_gt_oq
  // CHECK: fcmp ogt <16 x half> %{{.*}}, %{{.*}}
  return _mm256_cmp_ph_mask(a, b, _CMP_GT_OQ);
}

__mmask16 test_mm256_cmp_ph_mask_true_us(__m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_cmp_ph_mask_true_us
  // CHECK: fcmp true <16 x half> %{{.*}}, %{{.*}}
  return _mm256_cmp_ph_mask(a, b, _CMP_TRUE_US);
}

__mmask16 test_mm256_mask_cmp_ph_mask_eq_oq(__mmask16 m, __m256h a, __m256h b) {
  // CHECK-LABEL: @test_mm256_mask_cmp_ph_mask_eq_oq
  // CHECK: [[CMP:%.*]] = fcmp oeq <16 x half> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ph_mask(m, a, b, _CMP_EQ_OQ);
}

__mmask16 test_mm256_mask_cmp_ph_mask_lt_os(__mmask16 m, __m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ph_mask_lt_os
  // CHECK: [[CMP:%.*]] = fcmp olt <16 x half> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ph_mask(m, a, b, _CMP_LT_OS);
}

__mmask16 test_mm256_mask_cmp_ph_mask_le_os(__mmask16 m, __m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ph_mask_le_os
  // CHECK: [[CMP:%.*]] = fcmp ole <16 x half> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ph_mask(m, a, b, _CMP_LE_OS);
}

__mmask16 test_mm256_mask_cmp_ph_mask_unord_q(__mmask16 m, __m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ph_mask_unord_q
  // CHECK: [[CMP:%.*]] = fcmp uno <16 x half> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ph_mask(m, a, b, _CMP_UNORD_Q);
}

__mmask16 test_mm256_mask_cmp_ph_mask_neq_uq(__mmask16 m, __m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ph_mask_neq_uq
  // CHECK: [[CMP:%.*]] = fcmp une <16 x half> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ph_mask(m, a, b, _CMP_NEQ_UQ);
}

__mmask16 test_mm256_mask_cmp_ph_mask_nlt_us(__mmask16 m, __m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ph_mask_nlt_us
  // CHECK: [[CMP:%.*]] = fcmp uge <16 x half> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ph_mask(m, a, b, _CMP_NLT_US);
}

__mmask16 test_mm256_mask_cmp_ph_mask_nle_us(__mmask16 m, __m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ph_mask_nle_us
  // CHECK: [[CMP:%.*]] = fcmp ugt <16 x half> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ph_mask(m, a, b, _CMP_NLE_US);
}

__mmask16 test_mm256_mask_cmp_ph_mask_ord_q(__mmask16 m, __m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ph_mask_ord_q
  // CHECK: [[CMP:%.*]] = fcmp ord <16 x half> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ph_mask(m, a, b, _CMP_ORD_Q);
}

__mmask16 test_mm256_mask_cmp_ph_mask_eq_uq(__mmask16 m, __m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ph_mask_eq_uq
  // CHECK: [[CMP:%.*]] = fcmp ueq <16 x half> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ph_mask(m, a, b, _CMP_EQ_UQ);
}

__mmask16 test_mm256_mask_cmp_ph_mask_nge_us(__mmask16 m, __m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ph_mask_nge_us
  // CHECK: [[CMP:%.*]] = fcmp ult <16 x half> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ph_mask(m, a, b, _CMP_NGE_US);
}

__mmask16 test_mm256_mask_cmp_ph_mask_ngt_us(__mmask16 m, __m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ph_mask_ngt_us
  // CHECK: [[CMP:%.*]] = fcmp ule <16 x half> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ph_mask(m, a, b, _CMP_NGT_US);
}

__mmask16 test_mm256_mask_cmp_ph_mask_false_oq(__mmask16 m, __m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ph_mask_false_oq
  // CHECK: [[CMP:%.*]] = fcmp false <16 x half> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ph_mask(m, a, b, _CMP_FALSE_OQ);
}

__mmask16 test_mm256_mask_cmp_ph_mask_neq_oq(__mmask16 m, __m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ph_mask_neq_oq
  // CHECK: [[CMP:%.*]] = fcmp one <16 x half> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ph_mask(m, a, b, _CMP_NEQ_OQ);
}

__mmask16 test_mm256_mask_cmp_ph_mask_ge_os(__mmask16 m, __m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ph_mask_ge_os
  // CHECK: [[CMP:%.*]] = fcmp oge <16 x half> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ph_mask(m, a, b, _CMP_GE_OS);
}

__mmask16 test_mm256_mask_cmp_ph_mask_gt_os(__mmask16 m, __m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ph_mask_gt_os
  // CHECK: [[CMP:%.*]] = fcmp ogt <16 x half> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ph_mask(m, a, b, _CMP_GT_OS);
}

__mmask16 test_mm256_mask_cmp_ph_mask_true_uq(__mmask16 m, __m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ph_mask_true_uq
  // CHECK: [[CMP:%.*]] = fcmp true <16 x half> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ph_mask(m, a, b, _CMP_TRUE_UQ);
}

__mmask16 test_mm256_mask_cmp_ph_mask_eq_os(__mmask16 m, __m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ph_mask_eq_os
  // CHECK: [[CMP:%.*]] = fcmp oeq <16 x half> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ph_mask(m, a, b, _CMP_EQ_OS);
}

__mmask16 test_mm256_mask_cmp_ph_mask_lt_oq(__mmask16 m, __m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ph_mask_lt_oq
  // CHECK: [[CMP:%.*]] = fcmp olt <16 x half> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ph_mask(m, a, b, _CMP_LT_OQ);
}

__mmask16 test_mm256_mask_cmp_ph_mask_le_oq(__mmask16 m, __m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ph_mask_le_oq
  // CHECK: [[CMP:%.*]] = fcmp ole <16 x half> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ph_mask(m, a, b, _CMP_LE_OQ);
}

__mmask16 test_mm256_mask_cmp_ph_mask_unord_s(__mmask16 m, __m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ph_mask_unord_s
  // CHECK: [[CMP:%.*]] = fcmp uno <16 x half> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ph_mask(m, a, b, _CMP_UNORD_S);
}

__mmask16 test_mm256_mask_cmp_ph_mask_neq_us(__mmask16 m, __m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ph_mask_neq_us
  // CHECK: [[CMP:%.*]] = fcmp une <16 x half> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ph_mask(m, a, b, _CMP_NEQ_US);
}

__mmask16 test_mm256_mask_cmp_ph_mask_nlt_uq(__mmask16 m, __m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ph_mask_nlt_uq
  // CHECK: [[CMP:%.*]] = fcmp uge <16 x half> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ph_mask(m, a, b, _CMP_NLT_UQ);
}

__mmask16 test_mm256_mask_cmp_ph_mask_nle_uq(__mmask16 m, __m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ph_mask_nle_uq
  // CHECK: [[CMP:%.*]] = fcmp ugt <16 x half> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ph_mask(m, a, b, _CMP_NLE_UQ);
}

__mmask16 test_mm256_mask_cmp_ph_mask_ord_s(__mmask16 m, __m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ph_mask_ord_s
  // CHECK: [[CMP:%.*]] = fcmp ord <16 x half> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ph_mask(m, a, b, _CMP_ORD_S);
}

__mmask16 test_mm256_mask_cmp_ph_mask_eq_us(__mmask16 m, __m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ph_mask_eq_us
  // CHECK: [[CMP:%.*]] = fcmp ueq <16 x half> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ph_mask(m, a, b, _CMP_EQ_US);
}

__mmask16 test_mm256_mask_cmp_ph_mask_nge_uq(__mmask16 m, __m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ph_mask_nge_uq
  // CHECK: [[CMP:%.*]] = fcmp ult <16 x half> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ph_mask(m, a, b, _CMP_NGE_UQ);
}

__mmask16 test_mm256_mask_cmp_ph_mask_ngt_uq(__mmask16 m, __m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ph_mask_ngt_uq
  // CHECK: [[CMP:%.*]] = fcmp ule <16 x half> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ph_mask(m, a, b, _CMP_NGT_UQ);
}

__mmask16 test_mm256_mask_cmp_ph_mask_false_os(__mmask16 m, __m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ph_mask_false_os
  // CHECK: [[CMP:%.*]] = fcmp false <16 x half> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ph_mask(m, a, b, _CMP_FALSE_OS);
}

__mmask16 test_mm256_mask_cmp_ph_mask_neq_os(__mmask16 m, __m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ph_mask_neq_os
  // CHECK: [[CMP:%.*]] = fcmp one <16 x half> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ph_mask(m, a, b, _CMP_NEQ_OS);
}

__mmask16 test_mm256_mask_cmp_ph_mask_ge_oq(__mmask16 m, __m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ph_mask_ge_oq
  // CHECK: [[CMP:%.*]] = fcmp oge <16 x half> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ph_mask(m, a, b, _CMP_GE_OQ);
}

__mmask16 test_mm256_mask_cmp_ph_mask_gt_oq(__mmask16 m, __m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ph_mask_gt_oq
  // CHECK: [[CMP:%.*]] = fcmp ogt <16 x half> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ph_mask(m, a, b, _CMP_GT_OQ);
}

__mmask16 test_mm256_mask_cmp_ph_mask_true_us(__mmask16 m, __m256h a, __m256h b) {
  // CHECK-LABEL: test_mm256_mask_cmp_ph_mask_true_us
  // CHECK: [[CMP:%.*]] = fcmp true <16 x half> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> [[CMP]], {{.*}}
  return _mm256_mask_cmp_ph_mask(m, a, b, _CMP_TRUE_US);
}

__mmask8 test_mm_cmp_ph_mask_eq_oq(__m128h a, __m128h b) {
  // CHECK-LABEL: @test_mm_cmp_ph_mask_eq_oq
  // CHECK: fcmp oeq <8 x half> %{{.*}}, %{{.*}}
  return _mm_cmp_ph_mask(a, b, _CMP_EQ_OQ);
}

__mmask8 test_mm_cmp_ph_mask_lt_os(__m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_cmp_ph_mask_lt_os
  // CHECK: fcmp olt <8 x half> %{{.*}}, %{{.*}}
  return _mm_cmp_ph_mask(a, b, _CMP_LT_OS);
}

__mmask8 test_mm_cmp_ph_mask_le_os(__m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_cmp_ph_mask_le_os
  // CHECK: fcmp ole <8 x half> %{{.*}}, %{{.*}}
  return _mm_cmp_ph_mask(a, b, _CMP_LE_OS);
}

__mmask8 test_mm_cmp_ph_mask_unord_q(__m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_cmp_ph_mask_unord_q
  // CHECK: fcmp uno <8 x half> %{{.*}}, %{{.*}}
  return _mm_cmp_ph_mask(a, b, _CMP_UNORD_Q);
}

__mmask8 test_mm_cmp_ph_mask_neq_uq(__m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_cmp_ph_mask_neq_uq
  // CHECK: fcmp une <8 x half> %{{.*}}, %{{.*}}
  return _mm_cmp_ph_mask(a, b, _CMP_NEQ_UQ);
}

__mmask8 test_mm_cmp_ph_mask_nlt_us(__m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_cmp_ph_mask_nlt_us
  // CHECK: fcmp uge <8 x half> %{{.*}}, %{{.*}}
  return _mm_cmp_ph_mask(a, b, _CMP_NLT_US);
}

__mmask8 test_mm_cmp_ph_mask_nle_us(__m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_cmp_ph_mask_nle_us
  // CHECK: fcmp ugt <8 x half> %{{.*}}, %{{.*}}
  return _mm_cmp_ph_mask(a, b, _CMP_NLE_US);
}

__mmask8 test_mm_cmp_ph_mask_ord_q(__m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_cmp_ph_mask_ord_q
  // CHECK: fcmp ord <8 x half> %{{.*}}, %{{.*}}
  return _mm_cmp_ph_mask(a, b, _CMP_ORD_Q);
}

__mmask8 test_mm_cmp_ph_mask_eq_uq(__m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_cmp_ph_mask_eq_uq
  // CHECK: fcmp ueq <8 x half> %{{.*}}, %{{.*}}
  return _mm_cmp_ph_mask(a, b, _CMP_EQ_UQ);
}

__mmask8 test_mm_cmp_ph_mask_nge_us(__m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_cmp_ph_mask_nge_us
  // CHECK: fcmp ult <8 x half> %{{.*}}, %{{.*}}
  return _mm_cmp_ph_mask(a, b, _CMP_NGE_US);
}

__mmask8 test_mm_cmp_ph_mask_ngt_us(__m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_cmp_ph_mask_ngt_us
  // CHECK: fcmp ule <8 x half> %{{.*}}, %{{.*}}
  return _mm_cmp_ph_mask(a, b, _CMP_NGT_US);
}

__mmask8 test_mm_cmp_ph_mask_false_oq(__m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_cmp_ph_mask_false_oq
  // CHECK: fcmp false <8 x half> %{{.*}}, %{{.*}}
  return _mm_cmp_ph_mask(a, b, _CMP_FALSE_OQ);
}

__mmask8 test_mm_cmp_ph_mask_neq_oq(__m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_cmp_ph_mask_neq_oq
  // CHECK: fcmp one <8 x half> %{{.*}}, %{{.*}}
  return _mm_cmp_ph_mask(a, b, _CMP_NEQ_OQ);
}

__mmask8 test_mm_cmp_ph_mask_ge_os(__m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_cmp_ph_mask_ge_os
  // CHECK: fcmp oge <8 x half> %{{.*}}, %{{.*}}
  return _mm_cmp_ph_mask(a, b, _CMP_GE_OS);
}

__mmask8 test_mm_cmp_ph_mask_gt_os(__m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_cmp_ph_mask_gt_os
  // CHECK: fcmp ogt <8 x half> %{{.*}}, %{{.*}}
  return _mm_cmp_ph_mask(a, b, _CMP_GT_OS);
}

__mmask8 test_mm_cmp_ph_mask_true_uq(__m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_cmp_ph_mask_true_uq
  // CHECK: fcmp true <8 x half> %{{.*}}, %{{.*}}
  return _mm_cmp_ph_mask(a, b, _CMP_TRUE_UQ);
}

__mmask8 test_mm_cmp_ph_mask_eq_os(__m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_cmp_ph_mask_eq_os
  // CHECK: fcmp oeq <8 x half> %{{.*}}, %{{.*}}
  return _mm_cmp_ph_mask(a, b, _CMP_EQ_OS);
}

__mmask8 test_mm_cmp_ph_mask_lt_oq(__m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_cmp_ph_mask_lt_oq
  // CHECK: fcmp olt <8 x half> %{{.*}}, %{{.*}}
  return _mm_cmp_ph_mask(a, b, _CMP_LT_OQ);
}

__mmask8 test_mm_cmp_ph_mask_le_oq(__m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_cmp_ph_mask_le_oq
  // CHECK: fcmp ole <8 x half> %{{.*}}, %{{.*}}
  return _mm_cmp_ph_mask(a, b, _CMP_LE_OQ);
}

__mmask8 test_mm_cmp_ph_mask_unord_s(__m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_cmp_ph_mask_unord_s
  // CHECK: fcmp uno <8 x half> %{{.*}}, %{{.*}}
  return _mm_cmp_ph_mask(a, b, _CMP_UNORD_S);
}

__mmask8 test_mm_cmp_ph_mask_neq_us(__m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_cmp_ph_mask_neq_us
  // CHECK: fcmp une <8 x half> %{{.*}}, %{{.*}}
  return _mm_cmp_ph_mask(a, b, _CMP_NEQ_US);
}

__mmask8 test_mm_cmp_ph_mask_nlt_uq(__m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_cmp_ph_mask_nlt_uq
  // CHECK: fcmp uge <8 x half> %{{.*}}, %{{.*}}
  return _mm_cmp_ph_mask(a, b, _CMP_NLT_UQ);
}

__mmask8 test_mm_cmp_ph_mask_nle_uq(__m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_cmp_ph_mask_nle_uq
  // CHECK: fcmp ugt <8 x half> %{{.*}}, %{{.*}}
  return _mm_cmp_ph_mask(a, b, _CMP_NLE_UQ);
}

__mmask8 test_mm_cmp_ph_mask_ord_s(__m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_cmp_ph_mask_ord_s
  // CHECK: fcmp ord <8 x half> %{{.*}}, %{{.*}}
  return _mm_cmp_ph_mask(a, b, _CMP_ORD_S);
}

__mmask8 test_mm_cmp_ph_mask_eq_us(__m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_cmp_ph_mask_eq_us
  // CHECK: fcmp ueq <8 x half> %{{.*}}, %{{.*}}
  return _mm_cmp_ph_mask(a, b, _CMP_EQ_US);
}

__mmask8 test_mm_cmp_ph_mask_nge_uq(__m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_cmp_ph_mask_nge_uq
  // CHECK: fcmp ult <8 x half> %{{.*}}, %{{.*}}
  return _mm_cmp_ph_mask(a, b, _CMP_NGE_UQ);
}

__mmask8 test_mm_cmp_ph_mask_ngt_uq(__m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_cmp_ph_mask_ngt_uq
  // CHECK: fcmp ule <8 x half> %{{.*}}, %{{.*}}
  return _mm_cmp_ph_mask(a, b, _CMP_NGT_UQ);
}

__mmask8 test_mm_cmp_ph_mask_false_os(__m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_cmp_ph_mask_false_os
  // CHECK: fcmp false <8 x half> %{{.*}}, %{{.*}}
  return _mm_cmp_ph_mask(a, b, _CMP_FALSE_OS);
}

__mmask8 test_mm_cmp_ph_mask_neq_os(__m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_cmp_ph_mask_neq_os
  // CHECK: fcmp one <8 x half> %{{.*}}, %{{.*}}
  return _mm_cmp_ph_mask(a, b, _CMP_NEQ_OS);
}

__mmask8 test_mm_cmp_ph_mask_ge_oq(__m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_cmp_ph_mask_ge_oq
  // CHECK: fcmp oge <8 x half> %{{.*}}, %{{.*}}
  return _mm_cmp_ph_mask(a, b, _CMP_GE_OQ);
}

__mmask8 test_mm_cmp_ph_mask_gt_oq(__m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_cmp_ph_mask_gt_oq
  // CHECK: fcmp ogt <8 x half> %{{.*}}, %{{.*}}
  return _mm_cmp_ph_mask(a, b, _CMP_GT_OQ);
}

__mmask8 test_mm_cmp_ph_mask_true_us(__m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_cmp_ph_mask_true_us
  // CHECK: fcmp true <8 x half> %{{.*}}, %{{.*}}
  return _mm_cmp_ph_mask(a, b, _CMP_TRUE_US);
}

__mmask8 test_mm_mask_cmp_ph_mask_eq_oq(__mmask8 m, __m128h a, __m128h b) {
  // CHECK-LABEL: @test_mm_mask_cmp_ph_mask_eq_oq
  // CHECK: [[CMP:%.*]] = fcmp oeq <8 x half> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ph_mask(m, a, b, _CMP_EQ_OQ);
}

__mmask8 test_mm_mask_cmp_ph_mask_lt_os(__mmask8 m, __m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_mask_cmp_ph_mask_lt_os
  // CHECK: [[CMP:%.*]] = fcmp olt <8 x half> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ph_mask(m, a, b, _CMP_LT_OS);
}

__mmask8 test_mm_mask_cmp_ph_mask_le_os(__mmask8 m, __m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_mask_cmp_ph_mask_le_os
  // CHECK: [[CMP:%.*]] = fcmp ole <8 x half> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ph_mask(m, a, b, _CMP_LE_OS);
}

__mmask8 test_mm_mask_cmp_ph_mask_unord_q(__mmask8 m, __m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_mask_cmp_ph_mask_unord_q
  // CHECK: [[CMP:%.*]] = fcmp uno <8 x half> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ph_mask(m, a, b, _CMP_UNORD_Q);
}

__mmask8 test_mm_mask_cmp_ph_mask_neq_uq(__mmask8 m, __m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_mask_cmp_ph_mask_neq_uq
  // CHECK: [[CMP:%.*]] = fcmp une <8 x half> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ph_mask(m, a, b, _CMP_NEQ_UQ);
}

__mmask8 test_mm_mask_cmp_ph_mask_nlt_us(__mmask8 m, __m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_mask_cmp_ph_mask_nlt_us
  // CHECK: [[CMP:%.*]] = fcmp uge <8 x half> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ph_mask(m, a, b, _CMP_NLT_US);
}

__mmask8 test_mm_mask_cmp_ph_mask_nle_us(__mmask8 m, __m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_mask_cmp_ph_mask_nle_us
  // CHECK: [[CMP:%.*]] = fcmp ugt <8 x half> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ph_mask(m, a, b, _CMP_NLE_US);
}

__mmask8 test_mm_mask_cmp_ph_mask_ord_q(__mmask8 m, __m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_mask_cmp_ph_mask_ord_q
  // CHECK: [[CMP:%.*]] = fcmp ord <8 x half> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ph_mask(m, a, b, _CMP_ORD_Q);
}

__mmask8 test_mm_mask_cmp_ph_mask_eq_uq(__mmask8 m, __m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_mask_cmp_ph_mask_eq_uq
  // CHECK: [[CMP:%.*]] = fcmp ueq <8 x half> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ph_mask(m, a, b, _CMP_EQ_UQ);
}

__mmask8 test_mm_mask_cmp_ph_mask_nge_us(__mmask8 m, __m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_mask_cmp_ph_mask_nge_us
  // CHECK: [[CMP:%.*]] = fcmp ult <8 x half> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ph_mask(m, a, b, _CMP_NGE_US);
}

__mmask8 test_mm_mask_cmp_ph_mask_ngt_us(__mmask8 m, __m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_mask_cmp_ph_mask_ngt_us
  // CHECK: [[CMP:%.*]] = fcmp ule <8 x half> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ph_mask(m, a, b, _CMP_NGT_US);
}

__mmask8 test_mm_mask_cmp_ph_mask_false_oq(__mmask8 m, __m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_mask_cmp_ph_mask_false_oq
  // CHECK: [[CMP:%.*]] = fcmp false <8 x half> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ph_mask(m, a, b, _CMP_FALSE_OQ);
}

__mmask8 test_mm_mask_cmp_ph_mask_neq_oq(__mmask8 m, __m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_mask_cmp_ph_mask_neq_oq
  // CHECK: [[CMP:%.*]] = fcmp one <8 x half> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ph_mask(m, a, b, _CMP_NEQ_OQ);
}

__mmask8 test_mm_mask_cmp_ph_mask_ge_os(__mmask8 m, __m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_mask_cmp_ph_mask_ge_os
  // CHECK: [[CMP:%.*]] = fcmp oge <8 x half> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ph_mask(m, a, b, _CMP_GE_OS);
}

__mmask8 test_mm_mask_cmp_ph_mask_gt_os(__mmask8 m, __m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_mask_cmp_ph_mask_gt_os
  // CHECK: [[CMP:%.*]] = fcmp ogt <8 x half> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ph_mask(m, a, b, _CMP_GT_OS);
}

__mmask8 test_mm_mask_cmp_ph_mask_true_uq(__mmask8 m, __m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_mask_cmp_ph_mask_true_uq
  // CHECK: [[CMP:%.*]] = fcmp true <8 x half> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ph_mask(m, a, b, _CMP_TRUE_UQ);
}

__mmask8 test_mm_mask_cmp_ph_mask_eq_os(__mmask8 m, __m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_mask_cmp_ph_mask_eq_os
  // CHECK: [[CMP:%.*]] = fcmp oeq <8 x half> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ph_mask(m, a, b, _CMP_EQ_OS);
}

__mmask8 test_mm_mask_cmp_ph_mask_lt_oq(__mmask8 m, __m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_mask_cmp_ph_mask_lt_oq
  // CHECK: [[CMP:%.*]] = fcmp olt <8 x half> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ph_mask(m, a, b, _CMP_LT_OQ);
}

__mmask8 test_mm_mask_cmp_ph_mask_le_oq(__mmask8 m, __m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_mask_cmp_ph_mask_le_oq
  // CHECK: [[CMP:%.*]] = fcmp ole <8 x half> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ph_mask(m, a, b, _CMP_LE_OQ);
}

__mmask8 test_mm_mask_cmp_ph_mask_unord_s(__mmask8 m, __m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_mask_cmp_ph_mask_unord_s
  // CHECK: [[CMP:%.*]] = fcmp uno <8 x half> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ph_mask(m, a, b, _CMP_UNORD_S);
}

__mmask8 test_mm_mask_cmp_ph_mask_neq_us(__mmask8 m, __m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_mask_cmp_ph_mask_neq_us
  // CHECK: [[CMP:%.*]] = fcmp une <8 x half> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ph_mask(m, a, b, _CMP_NEQ_US);
}

__mmask8 test_mm_mask_cmp_ph_mask_nlt_uq(__mmask8 m, __m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_mask_cmp_ph_mask_nlt_uq
  // CHECK: [[CMP:%.*]] = fcmp uge <8 x half> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ph_mask(m, a, b, _CMP_NLT_UQ);
}

__mmask8 test_mm_mask_cmp_ph_mask_nle_uq(__mmask8 m, __m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_mask_cmp_ph_mask_nle_uq
  // CHECK: [[CMP:%.*]] = fcmp ugt <8 x half> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ph_mask(m, a, b, _CMP_NLE_UQ);
}

__mmask8 test_mm_mask_cmp_ph_mask_ord_s(__mmask8 m, __m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_mask_cmp_ph_mask_ord_s
  // CHECK: [[CMP:%.*]] = fcmp ord <8 x half> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ph_mask(m, a, b, _CMP_ORD_S);
}

__mmask8 test_mm_mask_cmp_ph_mask_eq_us(__mmask8 m, __m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_mask_cmp_ph_mask_eq_us
  // CHECK: [[CMP:%.*]] = fcmp ueq <8 x half> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ph_mask(m, a, b, _CMP_EQ_US);
}

__mmask8 test_mm_mask_cmp_ph_mask_nge_uq(__mmask8 m, __m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_mask_cmp_ph_mask_nge_uq
  // CHECK: [[CMP:%.*]] = fcmp ult <8 x half> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ph_mask(m, a, b, _CMP_NGE_UQ);
}

__mmask8 test_mm_mask_cmp_ph_mask_ngt_uq(__mmask8 m, __m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_mask_cmp_ph_mask_ngt_uq
  // CHECK: [[CMP:%.*]] = fcmp ule <8 x half> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ph_mask(m, a, b, _CMP_NGT_UQ);
}

__mmask8 test_mm_mask_cmp_ph_mask_false_os(__mmask8 m, __m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_mask_cmp_ph_mask_false_os
  // CHECK: [[CMP:%.*]] = fcmp false <8 x half> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ph_mask(m, a, b, _CMP_FALSE_OS);
}

__mmask8 test_mm_mask_cmp_ph_mask_neq_os(__mmask8 m, __m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_mask_cmp_ph_mask_neq_os
  // CHECK: [[CMP:%.*]] = fcmp one <8 x half> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ph_mask(m, a, b, _CMP_NEQ_OS);
}

__mmask8 test_mm_mask_cmp_ph_mask_ge_oq(__mmask8 m, __m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_mask_cmp_ph_mask_ge_oq
  // CHECK: [[CMP:%.*]] = fcmp oge <8 x half> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ph_mask(m, a, b, _CMP_GE_OQ);
}

__mmask8 test_mm_mask_cmp_ph_mask_gt_oq(__mmask8 m, __m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_mask_cmp_ph_mask_gt_oq
  // CHECK: [[CMP:%.*]] = fcmp ogt <8 x half> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ph_mask(m, a, b, _CMP_GT_OQ);
}

__mmask8 test_mm_mask_cmp_ph_mask_true_us(__mmask8 m, __m128h a, __m128h b) {
  // CHECK-LABEL: test_mm_mask_cmp_ph_mask_true_us
  // CHECK: [[CMP:%.*]] = fcmp true <8 x half> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> [[CMP]], {{.*}}
  return _mm_mask_cmp_ph_mask(m, a, b, _CMP_TRUE_US);
}

__m256h test_mm256_rcp_ph(__m256h __A) {
  // CHECK-LABEL: @test_mm256_rcp_ph
  // CHECK: @llvm.x86.avx512fp16.mask.rcp.ph.256
  return _mm256_rcp_ph(__A);
}

__m256h test_mm256_mask_rcp_ph(__m256h __W, __mmask32 __U, __m256h __A) {
  // CHECK-LABEL: @test_mm256_mask_rcp_ph
  // CHECK: @llvm.x86.avx512fp16.mask.rcp.ph.256
  return (__m256h)_mm256_mask_rcp_ph(__W, __U, __A);
}

__m256h test_mm256_maskz_rcp_ph(__mmask32 __U, __m256h __A) {
  // CHECK-LABEL: @test_mm256_maskz_rcp_ph
  // CHECK: @llvm.x86.avx512fp16.mask.rcp.ph.256
  return _mm256_maskz_rcp_ph(__U, __A);
}

__m128h test_mm_rcp_ph(__m128h __A) {
  // CHECK-LABEL: @test_mm_rcp_ph
  // CHECK: @llvm.x86.avx512fp16.mask.rcp.ph.128
  return _mm_rcp_ph(__A);
}

__m128h test_mm_mask_rcp_ph(__m128h __W, __mmask32 __U, __m128h __A) {
  // CHECK-LABEL: @test_mm_mask_rcp_ph
  // CHECK: @llvm.x86.avx512fp16.mask.rcp.ph.128
  return (__m128h)_mm_mask_rcp_ph(__W, __U, __A);
}

__m128h test_mm_maskz_rcp_ph(__mmask32 __U, __m128h __A) {
  // CHECK-LABEL: @test_mm_maskz_rcp_ph
  // CHECK: @llvm.x86.avx512fp16.mask.rcp.ph.128
  return _mm_maskz_rcp_ph(__U, __A);
}

__m256h test_mm256_rsqrt_ph(__m256h __A) {
  // CHECK-LABEL: @test_mm256_rsqrt_ph
  // CHECK: @llvm.x86.avx512fp16.mask.rsqrt.ph.256
  return _mm256_rsqrt_ph(__A);
}

__m256h test_mm256_mask_rsqrt_ph(__m256h __W, __mmask16 __U, __m256h __A) {
  // CHECK-LABEL: @test_mm256_mask_rsqrt_ph
  // CHECK: @llvm.x86.avx512fp16.mask.rsqrt.ph.256
  return (__m256h)_mm256_mask_rsqrt_ph(__W, __U, __A);
}

__m256h test_mm256_maskz_rsqrt_ph(__mmask16 __U, __m256h __A) {
  // CHECK-LABEL: @test_mm256_maskz_rsqrt_ph
  // CHECK: @llvm.x86.avx512fp16.mask.rsqrt.ph.256
  return _mm256_maskz_rsqrt_ph(__U, __A);
}

__m128h test_mm_rsqrt_ph(__m128h __A) {
  // CHECK-LABEL: @test_mm_rsqrt_ph
  // CHECK: @llvm.x86.avx512fp16.mask.rsqrt.ph.128
  return _mm_rsqrt_ph(__A);
}

__m128h test_mm_mask_rsqrt_ph(__m128h __W, __mmask32 __U, __m128h __A) {
  // CHECK-LABEL: @test_mm_mask_rsqrt_ph
  // CHECK: @llvm.x86.avx512fp16.mask.rsqrt.ph.128
  return (__m128h)_mm_mask_rsqrt_ph(__W, __U, __A);
}

__m128h test_mm_maskz_rsqrt_ph(__mmask32 __U, __m128h __A) {
  // CHECK-LABEL: @test_mm_maskz_rsqrt_ph
  // CHECK: @llvm.x86.avx512fp16.mask.rsqrt.ph.128
  return _mm_maskz_rsqrt_ph(__U, __A);
}

__m128h test_mm_getmant_ph(__m128h __A) {
  // CHECK-LABEL: @test_mm_getmant_ph
  // CHECK: @llvm.x86.avx512fp16.mask.getmant.ph.128
  return _mm_getmant_ph(__A, _MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan);
}

__m128h test_mm_mask_getmant_ph(__m128h __W, __mmask8 __U, __m128h __A) {
  // CHECK-LABEL: @test_mm_mask_getmant_ph
  // CHECK: @llvm.x86.avx512fp16.mask.getmant.ph.128
  return _mm_mask_getmant_ph(__W, __U, __A, _MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan);
}

__m128h test_mm_maskz_getmant_ph(__mmask8 __U, __m128h __A) {
  // CHECK-LABEL: @test_mm_maskz_getmant_ph
  // CHECK: @llvm.x86.avx512fp16.mask.getmant.ph.128
  return _mm_maskz_getmant_ph(__U, __A, _MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan);
}

__m256h test_mm256_getmant_ph(__m256h __A) {
  // CHECK-LABEL: @test_mm256_getmant_ph
  // CHECK: @llvm.x86.avx512fp16.mask.getmant.ph.256
  return _mm256_getmant_ph(__A, _MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan);
}

__m256h test_mm256_mask_getmant_ph(__m256h __W, __mmask16 __U, __m256h __A) {
  // CHECK-LABEL: @test_mm256_mask_getmant_ph
  // CHECK: @llvm.x86.avx512fp16.mask.getmant.ph.256
  return _mm256_mask_getmant_ph(__W, __U, __A, _MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan);
}

__m256h test_mm256_maskz_getmant_ph(__mmask16 __U, __m256h __A) {
  // CHECK-LABEL: @test_mm256_maskz_getmant_ph
  // CHECK: @llvm.x86.avx512fp16.mask.getmant.ph.256
  return _mm256_maskz_getmant_ph(__U, __A, _MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan);
}

__m128h test_mm_getexp_ph(__m128h __A) {
  // CHECK-LABEL: @test_mm_getexp_ph
  // CHECK: @llvm.x86.avx512fp16.mask.getexp.ph.128
  return _mm_getexp_ph(__A);
}

__m128h test_mm_mask_getexp_ph(__m128h __W, __mmask8 __U, __m128h __A) {
  // CHECK-LABEL: @test_mm_mask_getexp_ph
  // CHECK: @llvm.x86.avx512fp16.mask.getexp.ph.128
  return _mm_mask_getexp_ph(__W, __U, __A);
}

__m128h test_mm_maskz_getexp_ph(__mmask8 __U, __m128h __A) {
  // CHECK-LABEL: @test_mm_maskz_getexp_ph
  // CHECK: @llvm.x86.avx512fp16.mask.getexp.ph.128
  return _mm_maskz_getexp_ph(__U, __A);
}

__m256h test_mm256_getexp_ph(__m256h __A) {
  // CHECK-LABEL: @test_mm256_getexp_ph
  // CHECK: @llvm.x86.avx512fp16.mask.getexp.ph.256
  return _mm256_getexp_ph(__A);
}

__m256h test_mm256_mask_getexp_ph(__m256h __W, __mmask16 __U, __m256h __A) {
  // CHECK-LABEL: @test_mm256_mask_getexp_ph
  // CHECK: @llvm.x86.avx512fp16.mask.getexp.ph.256
  return _mm256_mask_getexp_ph(__W, __U, __A);
}

__m256h test_mm256_maskz_getexp_ph(__mmask16 __U, __m256h __A) {
  // CHECK-LABEL: @test_mm256_maskz_getexp_ph
  // CHECK: @llvm.x86.avx512fp16.mask.getexp.ph.256
  return _mm256_maskz_getexp_ph(__U, __A);
}

__m128h test_mm_scalef_ph(__m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_scalef_ph
  // CHECK: @llvm.x86.avx512fp16.mask.scalef.ph.128
  return _mm_scalef_ph(__A, __B);
}

__m128h test_mm_mask_scalef_ph(__m128h __W, __mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_mask_scalef_ph
  // CHECK: @llvm.x86.avx512fp16.mask.scalef.ph.128
  return _mm_mask_scalef_ph(__W, __U, __A, __B);
}

__m128h test_mm_maskz_scalef_ph(__mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_maskz_scalef_ph
  // CHECK: @llvm.x86.avx512fp16.mask.scalef.ph.128
  return _mm_maskz_scalef_ph(__U, __A, __B);
}

__m256h test_mm256_scalef_ph(__m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_scalef_ph
  // CHECK: @llvm.x86.avx512fp16.mask.scalef.ph.256
  return _mm256_scalef_ph(__A, __B);
}

__m256h test_mm256_mask_scalef_ph(__m256h __W, __mmask16 __U, __m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_mask_scalef_ph
  // CHECK: @llvm.x86.avx512fp16.mask.scalef.ph.256
  return _mm256_mask_scalef_ph(__W, __U, __A, __B);
}

__m256h test_mm256_maskz_scalef_ph(__mmask16 __U, __m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_maskz_scalef_ph
  // CHECK: @llvm.x86.avx512fp16.mask.scalef.ph.256
  return _mm256_maskz_scalef_ph(__U, __A, __B);
}

__m128h test_mm_roundscale_ph(__m128h __A) {
  // CHECK-LABEL: @test_mm_roundscale_ph
  // CHECK: @llvm.x86.avx512fp16.mask.rndscale.ph.128
  return _mm_roundscale_ph(__A, 4);
}

__m128h test_mm_mask_roundscale_ph(__m128h __W, __mmask8 __U, __m128h __A) {
  // CHECK-LABEL: @test_mm_mask_roundscale_ph
  // CHECK: @llvm.x86.avx512fp16.mask.rndscale.ph.128
  return _mm_mask_roundscale_ph(__W, __U, __A, 4);
}

__m128h test_mm_maskz_roundscale_ph(__mmask8 __U, __m128h __A) {
  // CHECK-LABEL: @test_mm_maskz_roundscale_ph
  // CHECK: @llvm.x86.avx512fp16.mask.rndscale.ph.128
  return _mm_maskz_roundscale_ph(__U, __A, 4);
}

__m256h test_mm256_roundscale_ph(__m256h __A) {
  // CHECK-LABEL: @test_mm256_roundscale_ph
  // CHECK: @llvm.x86.avx512fp16.mask.rndscale.ph.256
  return _mm256_roundscale_ph(__A, 4);
}

__m256h test_mm256_mask_roundscale_ph(__m256h __W, __mmask16 __U, __m256h __A) {
  // CHECK-LABEL: @test_mm256_mask_roundscale_ph
  // CHECK: @llvm.x86.avx512fp16.mask.rndscale.ph.256
  return _mm256_mask_roundscale_ph(__W, __U, __A, 4);
}

__m256h test_mm256_maskz_roundscale_ph(__mmask16 __U, __m256h __A) {
  // CHECK-LABEL: @test_mm256_maskz_roundscale_ph
  // CHECK: @llvm.x86.avx512fp16.mask.rndscale.ph.256
  return _mm256_maskz_roundscale_ph(__U, __A, 4);
}

__m128h test_mm_reduce_ph(__m128h __A) {
  // CHECK-LABEL: @test_mm_reduce_ph
  // CHECK: @llvm.x86.avx512fp16.mask.reduce.ph.128
  return _mm_reduce_ph(__A, 4);
}

__m128h test_mm_mask_reduce_ph(__m128h __W, __mmask8 __U, __m128h __A) {
  // CHECK-LABEL: @test_mm_mask_reduce_ph
  // CHECK: @llvm.x86.avx512fp16.mask.reduce.ph.128
  return _mm_mask_reduce_ph(__W, __U, __A, 4);
}

__m128h test_mm_maskz_reduce_ph(__mmask8 __U, __m128h __A) {
  // CHECK-LABEL: @test_mm_maskz_reduce_ph
  // CHECK: @llvm.x86.avx512fp16.mask.reduce.ph.128
  return _mm_maskz_reduce_ph(__U, __A, 4);
}

__m256h test_mm256_reduce_ph(__m256h __A) {
  // CHECK-LABEL: @test_mm256_reduce_ph
  // CHECK: @llvm.x86.avx512fp16.mask.reduce.ph.256
  return _mm256_reduce_ph(__A, 4);
}

__m256h test_mm256_mask_reduce_ph(__m256h __W, __mmask16 __U, __m256h __A) {
  // CHECK-LABEL: @test_mm256_mask_reduce_ph
  // CHECK: @llvm.x86.avx512fp16.mask.reduce.ph.256
  return _mm256_mask_reduce_ph(__W, __U, __A, 4);
}

__m256h test_mm256_maskz_reduce_ph(__mmask16 __U, __m256h __A) {
  // CHECK-LABEL: @test_mm256_maskz_reduce_ph
  // CHECK: @llvm.x86.avx512fp16.mask.reduce.ph.256
  return _mm256_maskz_reduce_ph(__U, __A, 4);
}
__m128h test_mm_sqrt_ph(__m128h x) {
  // CHECK-LABEL: test_mm_sqrt_ph
  // CHECK: call <8 x half> @llvm.sqrt.v8f16(<8 x half> {{.*}})
  return _mm_sqrt_ph(x);
}

__m256h test_mm256_sqrt_ph(__m256h A) {
  // CHECK-LABEL: test_mm256_sqrt_ph
  // CHECK: call <16 x half> @llvm.sqrt.v16f16(<16 x half> %{{.*}})
  return _mm256_sqrt_ph(A);
}

__m128h test_mm_mask_sqrt_ph(__m128h __W, __mmask8 __U, __m128h __A) {
  // CHECK-LABEL: @test_mm_mask_sqrt_ph
  // CHECK: @llvm.sqrt.v8f16
  // CHECK: select <8 x i1> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}}
  return _mm_mask_sqrt_ph(__W, __U, __A);
}

__m128h test_mm_maskz_sqrt_ph(__mmask8 __U, __m128h __A) {
  // CHECK-LABEL: @test_mm_maskz_sqrt_ph
  // CHECK: @llvm.sqrt.v8f16
  // CHECK: select <8 x i1> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}}
  return _mm_maskz_sqrt_ph(__U, __A);
}

__m256h test_mm256_mask_sqrt_ph(__m256h __W, __mmask16 __U, __m256h __A) {
  // CHECK-LABEL: @test_mm256_mask_sqrt_ph
  // CHECK: @llvm.sqrt.v16f16
  // CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return _mm256_mask_sqrt_ph(__W, __U, __A);
}

__m256h test_mm256_maskz_sqrt_ph(__mmask16 __U, __m256h __A) {
  // CHECK-LABEL: @test_mm256_maskz_sqrt_ph
  // CHECK: @llvm.sqrt.v16f16
  // CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return _mm256_maskz_sqrt_ph(__U, __A);
}
__mmask8 test_mm_mask_fpclass_ph_mask(__mmask8 __U, __m128h __A) {
  // CHECK-LABEL: @test_mm_mask_fpclass_ph_mask
  // CHECK: @llvm.x86.avx512fp16.fpclass.ph.128
  return _mm_mask_fpclass_ph_mask(__U, __A, 2);
}

__mmask8 test_mm_fpclass_ph_mask(__m128h __A) {
  // CHECK-LABEL: @test_mm_fpclass_ph_mask
  // CHECK: @llvm.x86.avx512fp16.fpclass.ph.128
  return _mm_fpclass_ph_mask(__A, 2);
}

__mmask16 test_mm256_mask_fpclass_ph_mask(__mmask16 __U, __m256h __A) {
  // CHECK-LABEL: @test_mm256_mask_fpclass_ph_mask
  // CHECK: @llvm.x86.avx512fp16.fpclass.ph.256
  return _mm256_mask_fpclass_ph_mask(__U, __A, 2);
}

__mmask16 test_mm256_fpclass_ph_mask(__m256h __A) {
  // CHECK-LABEL: @test_mm256_fpclass_ph_mask
  // CHECK: @llvm.x86.avx512fp16.fpclass.ph.256
  return _mm256_fpclass_ph_mask(__A, 2);
}

__m128h test_mm_cvtpd_ph(__m128d A) {
  // CHECK-LABEL: test_mm_cvtpd_ph
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtpd2ph.128
  return _mm_cvtpd_ph(A);
}

__m128h test_mm_mask_cvtpd_ph(__m128h A, __mmask8 B, __m128d C) {
  // CHECK-LABEL: test_mm_mask_cvtpd_ph
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtpd2ph.128
  return _mm_mask_cvtpd_ph(A, B, C);
}

__m128h test_mm_maskz_cvtpd_ph(__mmask8 A, __m128d B) {
  // CHECK-LABEL: test_mm_maskz_cvtpd_ph
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtpd2ph.128
  return _mm_maskz_cvtpd_ph(A, B);
}

__m128h test_mm256_cvtpd_ph(__m256d A) {
  // CHECK-LABEL: test_mm256_cvtpd_ph
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtpd2ph.256
  return _mm256_cvtpd_ph(A);
}

__m128h test_mm256_mask_cvtpd_ph(__m128h A, __mmask8 B, __m256d C) {
  // CHECK-LABEL: test_mm256_mask_cvtpd_ph
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtpd2ph.256
  return _mm256_mask_cvtpd_ph(A, B, C);
}

__m128h test_mm256_maskz_cvtpd_ph(__mmask8 A, __m256d B) {
  // CHECK-LABEL: test_mm256_maskz_cvtpd_ph
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtpd2ph.256
  return _mm256_maskz_cvtpd_ph(A, B);
}

__m128d test_mm_cvtph_pd(__m128h A) {
  // CHECK-LABEL: test_mm_cvtph_pd
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2pd.128
  return _mm_cvtph_pd(A);
}

__m128d test_mm_mask_cvtph_pd(__m128d A, __mmask8 B, __m128h C) {
  // CHECK-LABEL: test_mm_mask_cvtph_pd
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2pd.128
  return _mm_mask_cvtph_pd(A, B, C);
}

__m128d test_mm_maskz_cvtph_pd(__mmask8 A, __m128h B) {
  // CHECK-LABEL: test_mm_maskz_cvtph_pd
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2pd.128
  return _mm_maskz_cvtph_pd(A, B);
}

__m256d test_mm256_cvtph_pd(__m128h A) {
  // CHECK-LABEL: test_mm256_cvtph_pd
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2pd.256
  return _mm256_cvtph_pd(A);
}

__m256d test_mm256_mask_cvtph_pd(__m256d A, __mmask8 B, __m128h C) {
  // CHECK-LABEL: test_mm256_mask_cvtph_pd
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2pd.256
  return _mm256_mask_cvtph_pd(A, B, C);
}

__m256d test_mm256_maskz_cvtph_pd(__mmask8 A, __m128h B) {
  // CHECK-LABEL: test_mm256_maskz_cvtph_pd
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2pd.256
  return _mm256_maskz_cvtph_pd(A, B);
}

__m128i test_mm_cvtph_epi16(__m128h A) {
  // CHECK-LABEL: test_mm_cvtph_epi16
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2w.128
  return _mm_cvtph_epi16(A);
}

__m128i test_mm_mask_cvtph_epi16(__m128i A, __mmask8 B, __m128h C) {
  // CHECK-LABEL: test_mm_mask_cvtph_epi16
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2w.128
  return _mm_mask_cvtph_epi16(A, B, C);
}

__m128i test_mm_maskz_cvtph_epi16(__mmask8 A, __m128h B) {
  // CHECK-LABEL: test_mm_maskz_cvtph_epi16
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2w.128
  return _mm_maskz_cvtph_epi16(A, B);
}

__m256i test_mm256_cvtph_epi16(__m256h A) {
  // CHECK-LABEL: test_mm256_cvtph_epi16
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2w.256
  return _mm256_cvtph_epi16(A);
}

__m256i test_mm256_mask_cvtph_epi16(__m256i A, __mmask16 B, __m256h C) {
  // CHECK-LABEL: test_mm256_mask_cvtph_epi16
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2w.256
  return _mm256_mask_cvtph_epi16(A, B, C);
}

__m256i test_mm256_maskz_cvtph_epi16(__mmask16 A, __m256h B) {
  // CHECK-LABEL: test_mm256_maskz_cvtph_epi16
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2w.256
  return _mm256_maskz_cvtph_epi16(A, B);
}

__m128i test_mm_cvttph_epi16(__m128h A) {
  // CHECK-LABEL: test_mm_cvttph_epi16
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2w.128
  return _mm_cvttph_epi16(A);
}

__m128i test_mm_mask_cvttph_epi16(__m128i A, __mmask8 B, __m128h C) {
  // CHECK-LABEL: test_mm_mask_cvttph_epi16
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2w.128
  return _mm_mask_cvttph_epi16(A, B, C);
}

__m128i test_mm_maskz_cvttph_epi16(__mmask8 A, __m128h B) {
  // CHECK-LABEL: test_mm_maskz_cvttph_epi16
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2w.128
  return _mm_maskz_cvttph_epi16(A, B);
}

__m256i test_mm256_cvttph_epi16(__m256h A) {
  // CHECK-LABEL: test_mm256_cvttph_epi16
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2w.256
  return _mm256_cvttph_epi16(A);
}

__m256i test_mm256_mask_cvttph_epi16(__m256i A, __mmask16 B, __m256h C) {
  // CHECK-LABEL: test_mm256_mask_cvttph_epi16
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2w.256
  return _mm256_mask_cvttph_epi16(A, B, C);
}

__m256i test_mm256_maskz_cvttph_epi16(__mmask16 A, __m256h B) {
  // CHECK-LABEL: test_mm256_maskz_cvttph_epi16
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2w.256
  return _mm256_maskz_cvttph_epi16(A, B);
}

__m128h test_mm_cvtepi16_ph(__m128i A) {
  // CHECK-LABEL: test_mm_cvtepi16_ph
  // CHECK: %{{.*}} = sitofp <8 x i16> %{{.*}} to <8 x half>
  return _mm_cvtepi16_ph(A);
}

__m128h test_mm_mask_cvtepi16_ph(__m128h A, __mmask8 B, __m128i C) {
  // CHECK-LABEL: test_mm_mask_cvtepi16_ph
  // CHECK: %{{.*}} = sitofp <8 x i16> %{{.*}} to <8 x half>
  return _mm_mask_cvtepi16_ph(A, B, C);
}

__m128h test_mm_maskz_cvtepi16_ph(__mmask8 A, __m128i B) {
  // CHECK-LABEL: test_mm_maskz_cvtepi16_ph
  // CHECK: %{{.*}} = sitofp <8 x i16> %{{.*}} to <8 x half>
  return _mm_maskz_cvtepi16_ph(A, B);
}

__m256h test_mm256_cvtepi16_ph(__m256i A) {
  // CHECK-LABEL: test_mm256_cvtepi16_ph
  // CHECK: %{{.*}} = sitofp <16 x i16> %{{.*}} to <16 x half>
  return _mm256_cvtepi16_ph(A);
}

__m256h test_mm256_mask_cvtepi16_ph(__m256h A, __mmask16 B, __m256i C) {
  // CHECK-LABEL: test_mm256_mask_cvtepi16_ph
  // CHECK: %{{.*}} = sitofp <16 x i16> %{{.*}} to <16 x half>
  return _mm256_mask_cvtepi16_ph(A, B, C);
}

__m256h test_mm256_maskz_cvtepi16_ph(__mmask16 A, __m256i B) {
  // CHECK-LABEL: test_mm256_maskz_cvtepi16_ph
  // CHECK: %{{.*}} = sitofp <16 x i16> %{{.*}} to <16 x half>
  return _mm256_maskz_cvtepi16_ph(A, B);
}

__m128i test_mm_cvtph_epu16(__m128h A) {
  // CHECK-LABEL: test_mm_cvtph_epu16
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2uw.128
  return _mm_cvtph_epu16(A);
}

__m128i test_mm_mask_cvtph_epu16(__m128i A, __mmask8 B, __m128h C) {
  // CHECK-LABEL: test_mm_mask_cvtph_epu16
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2uw.128
  return _mm_mask_cvtph_epu16(A, B, C);
}

__m128i test_mm_maskz_cvtph_epu16(__mmask8 A, __m128h B) {
  // CHECK-LABEL: test_mm_maskz_cvtph_epu16
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2uw.128
  return _mm_maskz_cvtph_epu16(A, B);
}

__m256i test_mm256_cvtph_epu16(__m256h A) {
  // CHECK-LABEL: test_mm256_cvtph_epu16
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2uw.256
  return _mm256_cvtph_epu16(A);
}

__m256i test_mm256_mask_cvtph_epu16(__m256i A, __mmask16 B, __m256h C) {
  // CHECK-LABEL: test_mm256_mask_cvtph_epu16
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2uw.256
  return _mm256_mask_cvtph_epu16(A, B, C);
}

__m256i test_mm256_maskz_cvtph_epu16(__mmask16 A, __m256h B) {
  // CHECK-LABEL: test_mm256_maskz_cvtph_epu16
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2uw.256
  return _mm256_maskz_cvtph_epu16(A, B);
}

__m128i test_mm_cvttph_epu16(__m128h A) {
  // CHECK-LABEL: test_mm_cvttph_epu16
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2uw.128
  return _mm_cvttph_epu16(A);
}

__m128i test_mm_mask_cvttph_epu16(__m128i A, __mmask8 B, __m128h C) {
  // CHECK-LABEL: test_mm_mask_cvttph_epu16
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2uw.128
  return _mm_mask_cvttph_epu16(A, B, C);
}

__m128i test_mm_maskz_cvttph_epu16(__mmask8 A, __m128h B) {
  // CHECK-LABEL: test_mm_maskz_cvttph_epu16
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2uw.128
  return _mm_maskz_cvttph_epu16(A, B);
}

__m256i test_mm256_cvttph_epu16(__m256h A) {
  // CHECK-LABEL: test_mm256_cvttph_epu16
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2uw.256
  return _mm256_cvttph_epu16(A);
}

__m256i test_mm256_mask_cvttph_epu16(__m256i A, __mmask16 B, __m256h C) {
  // CHECK-LABEL: test_mm256_mask_cvttph_epu16
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2uw.256
  return _mm256_mask_cvttph_epu16(A, B, C);
}

__m256i test_mm256_maskz_cvttph_epu16(__mmask16 A, __m256h B) {
  // CHECK-LABEL: test_mm256_maskz_cvttph_epu16
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2uw.256
  return _mm256_maskz_cvttph_epu16(A, B);
}

__m128h test_mm_cvtepu16_ph(__m128i A) {
  // CHECK-LABEL: test_mm_cvtepu16_ph
  // CHECK: %{{.*}} = uitofp <8 x i16> %{{.*}} to <8 x half>
  return _mm_cvtepu16_ph(A);
}

__m128h test_mm_mask_cvtepu16_ph(__m128h A, __mmask8 B, __m128i C) {
  // CHECK-LABEL: test_mm_mask_cvtepu16_ph
  // CHECK: %{{.*}} = uitofp <8 x i16> %{{.*}} to <8 x half>
  return _mm_mask_cvtepu16_ph(A, B, C);
}

__m128h test_mm_maskz_cvtepu16_ph(__mmask8 A, __m128i B) {
  // CHECK-LABEL: test_mm_maskz_cvtepu16_ph
  // CHECK: %{{.*}} = uitofp <8 x i16> %{{.*}} to <8 x half>
  return _mm_maskz_cvtepu16_ph(A, B);
}

__m256h test_mm256_cvtepu16_ph(__m256i A) {
  // CHECK-LABEL: test_mm256_cvtepu16_ph
  // CHECK: %{{.*}} = uitofp <16 x i16> %{{.*}} to <16 x half>
  return _mm256_cvtepu16_ph(A);
}

__m256h test_mm256_mask_cvtepu16_ph(__m256h A, __mmask16 B, __m256i C) {
  // CHECK-LABEL: test_mm256_mask_cvtepu16_ph
  // CHECK: %{{.*}} = uitofp <16 x i16> %{{.*}} to <16 x half>
  return _mm256_mask_cvtepu16_ph(A, B, C);
}

__m256h test_mm256_maskz_cvtepu16_ph(__mmask16 A, __m256i B) {
  // CHECK-LABEL: test_mm256_maskz_cvtepu16_ph
  // CHECK: %{{.*}} = uitofp <16 x i16> %{{.*}} to <16 x half>
  return _mm256_maskz_cvtepu16_ph(A, B);
}

__m128i test_mm_cvtph_epi32(__m128h A) {
  // CHECK-LABEL: test_mm_cvtph_epi32
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2dq.128
  return _mm_cvtph_epi32(A);
}

__m128i test_mm_mask_cvtph_epi32(__m128i A, __mmask8 B, __m128h C) {
  // CHECK-LABEL: test_mm_mask_cvtph_epi32
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2dq.128
  return _mm_mask_cvtph_epi32(A, B, C);
}

__m128i test_mm_maskz_cvtph_epi32(__mmask8 A, __m128h B) {
  // CHECK-LABEL: test_mm_maskz_cvtph_epi32
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2dq.128
  return _mm_maskz_cvtph_epi32(A, B);
}

__m256i test_mm256_cvtph_epi32(__m128h A) {
  // CHECK-LABEL: test_mm256_cvtph_epi32
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2dq.256
  return _mm256_cvtph_epi32(A);
}

__m256i test_mm256_mask_cvtph_epi32(__m256i A, __mmask8 B, __m128h C) {
  // CHECK-LABEL: test_mm256_mask_cvtph_epi32
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2dq.256
  return _mm256_mask_cvtph_epi32(A, B, C);
}

__m256i test_mm256_maskz_cvtph_epi32(__mmask8 A, __m128h B) {
  // CHECK-LABEL: test_mm256_maskz_cvtph_epi32
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2dq.256
  return _mm256_maskz_cvtph_epi32(A, B);
}

__m128i test_mm_cvtph_epu32(__m128h A) {
  // CHECK-LABEL: test_mm_cvtph_epu32
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2udq.128
  return _mm_cvtph_epu32(A);
}

__m128i test_mm_mask_cvtph_epu32(__m128i A, __mmask8 B, __m128h C) {
  // CHECK-LABEL: test_mm_mask_cvtph_epu32
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2udq.128
  return _mm_mask_cvtph_epu32(A, B, C);
}

__m128i test_mm_maskz_cvtph_epu32(__mmask8 A, __m128h B) {
  // CHECK-LABEL: test_mm_maskz_cvtph_epu32
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2udq.128
  return _mm_maskz_cvtph_epu32(A, B);
}

__m256i test_mm256_cvtph_epu32(__m128h A) {
  // CHECK-LABEL: test_mm256_cvtph_epu32
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2udq.256
  return _mm256_cvtph_epu32(A);
}

__m256i test_mm256_mask_cvtph_epu32(__m256i A, __mmask8 B, __m128h C) {
  // CHECK-LABEL: test_mm256_mask_cvtph_epu32
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2udq.256
  return _mm256_mask_cvtph_epu32(A, B, C);
}

__m256i test_mm256_maskz_cvtph_epu32(__mmask8 A, __m128h B) {
  // CHECK-LABEL: test_mm256_maskz_cvtph_epu32
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2udq.256
  return _mm256_maskz_cvtph_epu32(A, B);
}

__m128h test_mm_cvtepi32_ph(__m128i A) {
  // CHECK-LABEL: test_mm_cvtepi32_ph
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtdq2ph.128
  return _mm_cvtepi32_ph(A);
}

__m128h test_mm_mask_cvtepi32_ph(__m128h A, __mmask8 B, __m128i C) {
  // CHECK-LABEL: test_mm_mask_cvtepi32_ph
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtdq2ph.128
  return _mm_mask_cvtepi32_ph(A, B, C);
}

__m128h test_mm_maskz_cvtepi32_ph(__mmask8 A, __m128i B) {
  // CHECK-LABEL: test_mm_maskz_cvtepi32_ph
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtdq2ph.128
  return _mm_maskz_cvtepi32_ph(A, B);
}

__m128h test_mm256_cvtepi32_ph(__m256i A) {
  // CHECK-LABEL: test_mm256_cvtepi32_ph
  // CHECK: %{{.*}} = sitofp <8 x i32> %{{.*}} to <8 x half>
  return _mm256_cvtepi32_ph(A);
}

__m128h test_mm256_mask_cvtepi32_ph(__m128h A, __mmask8 B, __m256i C) {
  // CHECK-LABEL: test_mm256_mask_cvtepi32_ph
  // CHECK: %{{.*}} = sitofp <8 x i32> %{{.*}} to <8 x half>
  return _mm256_mask_cvtepi32_ph(A, B, C);
}

__m128h test_mm256_maskz_cvtepi32_ph(__mmask8 A, __m256i B) {
  // CHECK-LABEL: test_mm256_maskz_cvtepi32_ph
  // CHECK: %{{.*}} = sitofp <8 x i32> %{{.*}} to <8 x half>
  return _mm256_maskz_cvtepi32_ph(A, B);
}

__m128h test_mm_cvtepu32_ph(__m128i A) {
  // CHECK-LABEL: test_mm_cvtepu32_ph
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtudq2ph.128
  return _mm_cvtepu32_ph(A);
}

__m128h test_mm_mask_cvtepu32_ph(__m128h A, __mmask8 B, __m128i C) {
  // CHECK-LABEL: test_mm_mask_cvtepu32_ph
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtudq2ph.128
  return _mm_mask_cvtepu32_ph(A, B, C);
}

__m128h test_mm_maskz_cvtepu32_ph(__mmask8 A, __m128i B) {
  // CHECK-LABEL: test_mm_maskz_cvtepu32_ph
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtudq2ph.128
  return _mm_maskz_cvtepu32_ph(A, B);
}

__m128h test_mm256_cvtepu32_ph(__m256i A) {
  // CHECK-LABEL: test_mm256_cvtepu32_ph
  // CHECK: %{{.*}} = uitofp <8 x i32> %{{.*}} to <8 x half>
  return _mm256_cvtepu32_ph(A);
}

__m128h test_mm256_mask_cvtepu32_ph(__m128h A, __mmask8 B, __m256i C) {
  // CHECK-LABEL: test_mm256_mask_cvtepu32_ph
  // CHECK: %{{.*}} = uitofp <8 x i32> %{{.*}} to <8 x half>
  return _mm256_mask_cvtepu32_ph(A, B, C);
}

__m128h test_mm256_maskz_cvtepu32_ph(__mmask8 A, __m256i B) {
  // CHECK-LABEL: test_mm256_maskz_cvtepu32_ph
  // CHECK: %{{.*}} = uitofp <8 x i32> %{{.*}} to <8 x half>
  return _mm256_maskz_cvtepu32_ph(A, B);
}

__m128i test_mm_cvttph_epi32(__m128h A) {
  // CHECK-LABEL: test_mm_cvttph_epi32
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2dq.128
  return _mm_cvttph_epi32(A);
}

__m128i test_mm_mask_cvttph_epi32(__m128i A, __mmask8 B, __m128h C) {
  // CHECK-LABEL: test_mm_mask_cvttph_epi32
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2dq.128
  return _mm_mask_cvttph_epi32(A, B, C);
}

__m128i test_mm_maskz_cvttph_epi32(__mmask8 A, __m128h B) {
  // CHECK-LABEL: test_mm_maskz_cvttph_epi32
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2dq.128
  return _mm_maskz_cvttph_epi32(A, B);
}

__m256i test_mm256_cvttph_epi32(__m128h A) {
  // CHECK-LABEL: test_mm256_cvttph_epi32
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2dq.256
  return _mm256_cvttph_epi32(A);
}

__m256i test_mm256_mask_cvttph_epi32(__m256i A, __mmask8 B, __m128h C) {
  // CHECK-LABEL: test_mm256_mask_cvttph_epi32
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2dq.256
  return _mm256_mask_cvttph_epi32(A, B, C);
}

__m256i test_mm256_maskz_cvttph_epi32(__mmask8 A, __m128h B) {
  // CHECK-LABEL: test_mm256_maskz_cvttph_epi32
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2dq.256
  return _mm256_maskz_cvttph_epi32(A, B);
}

__m128i test_mm_cvttph_epu32(__m128h A) {
  // CHECK-LABEL: test_mm_cvttph_epu32
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2udq.128
  return _mm_cvttph_epu32(A);
}

__m128i test_mm_mask_cvttph_epu32(__m128i A, __mmask8 B, __m128h C) {
  // CHECK-LABEL: test_mm_mask_cvttph_epu32
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2udq.128
  return _mm_mask_cvttph_epu32(A, B, C);
}

__m128i test_mm_maskz_cvttph_epu32(__mmask8 A, __m128h B) {
  // CHECK-LABEL: test_mm_maskz_cvttph_epu32
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2udq.128
  return _mm_maskz_cvttph_epu32(A, B);
}

__m256i test_mm256_cvttph_epu32(__m128h A) {
  // CHECK-LABEL: test_mm256_cvttph_epu32
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2udq.256
  return _mm256_cvttph_epu32(A);
}

__m256i test_mm256_mask_cvttph_epu32(__m256i A, __mmask8 B, __m128h C) {
  // CHECK-LABEL: test_mm256_mask_cvttph_epu32
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2udq.256
  return _mm256_mask_cvttph_epu32(A, B, C);
}

__m256i test_mm256_maskz_cvttph_epu32(__mmask8 A, __m128h B) {
  // CHECK-LABEL: test_mm256_maskz_cvttph_epu32
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2udq.256
  return _mm256_maskz_cvttph_epu32(A, B);
}

__m128h test_mm_cvtepi64_ph(__m128i A) {
  // CHECK-LABEL: test_mm_cvtepi64_ph
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtqq2ph.128
  return _mm_cvtepi64_ph(A);
}

__m128h test_mm_mask_cvtepi64_ph(__m128h A, __mmask8 B, __m128i C) {
  // CHECK-LABEL: test_mm_mask_cvtepi64_ph
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtqq2ph.128
  return _mm_mask_cvtepi64_ph(A, B, C);
}

__m128h test_mm_maskz_cvtepi64_ph(__mmask8 A, __m128i B) {
  // CHECK-LABEL: test_mm_maskz_cvtepi64_ph
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtqq2ph.128
  return _mm_maskz_cvtepi64_ph(A, B);
}

__m128h test_mm256_cvtepi64_ph(__m256i A) {
  // CHECK-LABEL: test_mm256_cvtepi64_ph
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtqq2ph.256
  return _mm256_cvtepi64_ph(A);
}

__m128h test_mm256_mask_cvtepi64_ph(__m128h A, __mmask8 B, __m256i C) {
  // CHECK-LABEL: test_mm256_mask_cvtepi64_ph
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtqq2ph.256
  return _mm256_mask_cvtepi64_ph(A, B, C);
}

__m128h test_mm256_maskz_cvtepi64_ph(__mmask8 A, __m256i B) {
  // CHECK-LABEL: test_mm256_maskz_cvtepi64_ph
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtqq2ph.256
  return _mm256_maskz_cvtepi64_ph(A, B);
}

__m128i test_mm_cvtph_epi64(__m128h A) {
  // CHECK-LABEL: test_mm_cvtph_epi64
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2qq.128
  return _mm_cvtph_epi64(A);
}

__m128i test_mm_mask_cvtph_epi64(__m128i A, __mmask8 B, __m128h C) {
  // CHECK-LABEL: test_mm_mask_cvtph_epi64
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2qq.128
  return _mm_mask_cvtph_epi64(A, B, C);
}

__m128i test_mm_maskz_cvtph_epi64(__mmask8 A, __m128h B) {
  // CHECK-LABEL: test_mm_maskz_cvtph_epi64
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2qq.128
  return _mm_maskz_cvtph_epi64(A, B);
}

__m256i test_mm256_cvtph_epi64(__m128h A) {
  // CHECK-LABEL: test_mm256_cvtph_epi64
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2qq.256
  return _mm256_cvtph_epi64(A);
}

__m256i test_mm256_mask_cvtph_epi64(__m256i A, __mmask8 B, __m128h C) {
  // CHECK-LABEL: test_mm256_mask_cvtph_epi64
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2qq.256
  return _mm256_mask_cvtph_epi64(A, B, C);
}

__m256i test_mm256_maskz_cvtph_epi64(__mmask8 A, __m128h B) {
  // CHECK-LABEL: test_mm256_maskz_cvtph_epi64
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2qq.256
  return _mm256_maskz_cvtph_epi64(A, B);
}

__m128h test_mm_cvtepu64_ph(__m128i A) {
  // CHECK-LABEL: test_mm_cvtepu64_ph
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtuqq2ph.128
  return _mm_cvtepu64_ph(A);
}

__m128h test_mm_mask_cvtepu64_ph(__m128h A, __mmask8 B, __m128i C) {
  // CHECK-LABEL: test_mm_mask_cvtepu64_ph
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtuqq2ph.128
  return _mm_mask_cvtepu64_ph(A, B, C);
}

__m128h test_mm_maskz_cvtepu64_ph(__mmask8 A, __m128i B) {
  // CHECK-LABEL: test_mm_maskz_cvtepu64_ph
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtuqq2ph.128
  return _mm_maskz_cvtepu64_ph(A, B);
}

__m128h test_mm256_cvtepu64_ph(__m256i A) {
  // CHECK-LABEL: test_mm256_cvtepu64_ph
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtuqq2ph.256
  return _mm256_cvtepu64_ph(A);
}

__m128h test_mm256_mask_cvtepu64_ph(__m128h A, __mmask8 B, __m256i C) {
  // CHECK-LABEL: test_mm256_mask_cvtepu64_ph
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtuqq2ph.256
  return _mm256_mask_cvtepu64_ph(A, B, C);
}

__m128h test_mm256_maskz_cvtepu64_ph(__mmask8 A, __m256i B) {
  // CHECK-LABEL: test_mm256_maskz_cvtepu64_ph
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtuqq2ph.256
  return _mm256_maskz_cvtepu64_ph(A, B);
}

__m128i test_mm_cvtph_epu64(__m128h A) {
  // CHECK-LABEL: test_mm_cvtph_epu64
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2uqq.128
  return _mm_cvtph_epu64(A);
}

__m128i test_mm_mask_cvtph_epu64(__m128i A, __mmask8 B, __m128h C) {
  // CHECK-LABEL: test_mm_mask_cvtph_epu64
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2uqq.128
  return _mm_mask_cvtph_epu64(A, B, C);
}

__m128i test_mm_maskz_cvtph_epu64(__mmask8 A, __m128h B) {
  // CHECK-LABEL: test_mm_maskz_cvtph_epu64
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2uqq.128
  return _mm_maskz_cvtph_epu64(A, B);
}

__m256i test_mm256_cvtph_epu64(__m128h A) {
  // CHECK-LABEL: test_mm256_cvtph_epu64
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2uqq.256
  return _mm256_cvtph_epu64(A);
}

__m256i test_mm256_mask_cvtph_epu64(__m256i A, __mmask8 B, __m128h C) {
  // CHECK-LABEL: test_mm256_mask_cvtph_epu64
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2uqq.256
  return _mm256_mask_cvtph_epu64(A, B, C);
}

__m256i test_mm256_maskz_cvtph_epu64(__mmask8 A, __m128h B) {
  // CHECK-LABEL: test_mm256_maskz_cvtph_epu64
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2uqq.256
  return _mm256_maskz_cvtph_epu64(A, B);
}

__m128i test_mm_cvttph_epi64(__m128h A) {
  // CHECK-LABEL: test_mm_cvttph_epi64
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2qq.128
  return _mm_cvttph_epi64(A);
}

__m128i test_mm_mask_cvttph_epi64(__m128i A, __mmask8 B, __m128h C) {
  // CHECK-LABEL: test_mm_mask_cvttph_epi64
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2qq.128
  return _mm_mask_cvttph_epi64(A, B, C);
}

__m128i test_mm_maskz_cvttph_epi64(__mmask8 A, __m128h B) {
  // CHECK-LABEL: test_mm_maskz_cvttph_epi64
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2qq.128
  return _mm_maskz_cvttph_epi64(A, B);
}

__m256i test_mm256_cvttph_epi64(__m128h A) {
  // CHECK-LABEL: test_mm256_cvttph_epi64
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2qq.256
  return _mm256_cvttph_epi64(A);
}

__m256i test_mm256_mask_cvttph_epi64(__m256i A, __mmask8 B, __m128h C) {
  // CHECK-LABEL: test_mm256_mask_cvttph_epi64
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2qq.256
  return _mm256_mask_cvttph_epi64(A, B, C);
}

__m256i test_mm256_maskz_cvttph_epi64(__mmask8 A, __m128h B) {
  // CHECK-LABEL: test_mm256_maskz_cvttph_epi64
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2qq.256
  return _mm256_maskz_cvttph_epi64(A, B);
}

__m128i test_mm_cvttph_epu64(__m128h A) {
  // CHECK-LABEL: test_mm_cvttph_epu64
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2uqq.128
  return _mm_cvttph_epu64(A);
}

__m128i test_mm_mask_cvttph_epu64(__m128i A, __mmask8 B, __m128h C) {
  // CHECK-LABEL: test_mm_mask_cvttph_epu64
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2uqq.128
  return _mm_mask_cvttph_epu64(A, B, C);
}

__m128i test_mm_maskz_cvttph_epu64(__mmask8 A, __m128h B) {
  // CHECK-LABEL: test_mm_maskz_cvttph_epu64
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2uqq.128
  return _mm_maskz_cvttph_epu64(A, B);
}

__m256i test_mm256_cvttph_epu64(__m128h A) {
  // CHECK-LABEL: test_mm256_cvttph_epu64
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2uqq.256
  return _mm256_cvttph_epu64(A);
}

__m256i test_mm256_mask_cvttph_epu64(__m256i A, __mmask8 B, __m128h C) {
  // CHECK-LABEL: test_mm256_mask_cvttph_epu64
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2uqq.256
  return _mm256_mask_cvttph_epu64(A, B, C);
}

__m256i test_mm256_maskz_cvttph_epu64(__mmask8 A, __m128h B) {
  // CHECK-LABEL: test_mm256_maskz_cvttph_epu64
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2uqq.256
  return _mm256_maskz_cvttph_epu64(A, B);
}

__m128 test_mm_cvtxph_ps(__m128h A) {
  // CHECK-LABEL: test_mm_cvtxph_ps
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2psx.128
  return _mm_cvtxph_ps(A);
}

__m128 test_mm_mask_cvtxph_ps(__m128 A, __mmask8 B, __m128h C) {
  // CHECK-LABEL: test_mm_mask_cvtxph_ps
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2psx.128
  return _mm_mask_cvtxph_ps(A, B, C);
}

__m128 test_mm_maskz_cvtxph_ps(__mmask8 A, __m128h B) {
  // CHECK-LABEL: test_mm_maskz_cvtxph_ps
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2psx.128
  return _mm_maskz_cvtxph_ps(A, B);
}

__m256 test_mm256_cvtxph_ps(__m128h A) {
  // CHECK-LABEL: test_mm256_cvtxph_ps
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2psx.256
  return _mm256_cvtxph_ps(A);
}

__m256 test_mm256_mask_cvtxph_ps(__m256 A, __mmask8 B, __m128h C) {
  // CHECK-LABEL: test_mm256_mask_cvtxph_ps
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2psx.256
  return _mm256_mask_cvtxph_ps(A, B, C);
}

__m256 test_mm256_maskz_cvtxph_ps(__mmask8 A, __m128h B) {
  // CHECK-LABEL: test_mm256_maskz_cvtxph_ps
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2psx.256
  return _mm256_maskz_cvtxph_ps(A, B);
}

__m128h test_mm_cvtxps_ph(__m128 A) {
  // CHECK-LABEL: test_mm_cvtxps_ph
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtps2phx.128
  return _mm_cvtxps_ph(A);
}

__m128h test_mm_mask_cvtxps_ph(__m128h A, __mmask8 B, __m128 C) {
  // CHECK-LABEL: test_mm_mask_cvtxps_ph
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtps2phx.128
  return _mm_mask_cvtxps_ph(A, B, C);
}

__m128h test_mm_maskz_cvtxps_ph(__mmask8 A, __m128 B) {
  // CHECK-LABEL: test_mm_maskz_cvtxps_ph
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtps2phx.128
  return _mm_maskz_cvtxps_ph(A, B);
}

__m128h test_mm256_cvtxps_ph(__m256 A) {
  // CHECK-LABEL: test_mm256_cvtxps_ph
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtps2phx.256
  return _mm256_cvtxps_ph(A);
}

__m128h test_mm256_mask_cvtxps_ph(__m128h A, __mmask8 B, __m256 C) {
  // CHECK-LABEL: test_mm256_mask_cvtxps_ph
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtps2phx.256
  return _mm256_mask_cvtxps_ph(A, B, C);
}

__m128h test_mm256_maskz_cvtxps_ph(__mmask8 A, __m256 B) {
  // CHECK-LABEL: test_mm256_maskz_cvtxps_ph
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtps2phx.256
  return _mm256_maskz_cvtxps_ph(A, B);
}

__m128h test_mm_fmadd_ph(__m128h __A, __m128h __B, __m128h __C) {
  // CHECK-LABEL: @test_mm_fmadd_ph
  // CHECK: call <8 x half> @llvm.fma.v8f16(<8 x half> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}})
  return _mm_fmadd_ph(__A, __B, __C);
}

__m128h test_mm_mask_fmadd_ph(__m128h __A, __mmask8 __U, __m128h __B, __m128h __C) {
  // CHECK-LABEL: @test_mm_mask_fmadd_ph
  // CHECK: call <8 x half> @llvm.fma.v8f16(<8 x half> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}}
  return _mm_mask_fmadd_ph(__A, __U, __B, __C);
}

__m128h test_mm_fmsub_ph(__m128h __A, __m128h __B, __m128h __C) {
  // CHECK-LABEL: @test_mm_fmsub_ph
  // CHECK: fneg
  // CHECK: call <8 x half> @llvm.fma.v8f16(<8 x half> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}})
  return _mm_fmsub_ph(__A, __B, __C);
}

__m128h test_mm_mask_fmsub_ph(__m128h __A, __mmask8 __U, __m128h __B, __m128h __C) {
  // CHECK-LABEL: @test_mm_mask_fmsub_ph
  // CHECK: fneg
  // CHECK: call <8 x half> @llvm.fma.v8f16(<8 x half> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}}
  return _mm_mask_fmsub_ph(__A, __U, __B, __C);
}

__m128h test_mm_mask3_fmadd_ph(__m128h __A, __m128h __B, __m128h __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm_mask3_fmadd_ph
  // CHECK: call <8 x half> @llvm.fma.v8f16(<8 x half> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}}
  return _mm_mask3_fmadd_ph(__A, __B, __C, __U);
}

__m128h test_mm_mask3_fnmadd_ph(__m128h __A, __m128h __B, __m128h __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm_mask3_fnmadd_ph
  // CHECK: fneg
  // CHECK: call <8 x half> @llvm.fma.v8f16(<8 x half> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}}
  return _mm_mask3_fnmadd_ph(__A, __B, __C, __U);
}

__m128h test_mm_maskz_fmadd_ph(__mmask8 __U, __m128h __A, __m128h __B, __m128h __C) {
  // CHECK-LABEL: @test_mm_maskz_fmadd_ph
  // CHECK: call <8 x half> @llvm.fma.v8f16(<8 x half> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}}
  return _mm_maskz_fmadd_ph(__U, __A, __B, __C);
}

__m128h test_mm_maskz_fmsub_ph(__mmask8 __U, __m128h __A, __m128h __B, __m128h __C) {
  // CHECK-LABEL: @test_mm_maskz_fmsub_ph
  // CHECK: fneg
  // CHECK: call <8 x half> @llvm.fma.v8f16(<8 x half> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}}
  return _mm_maskz_fmsub_ph(__U, __A, __B, __C);
}

__m128h test_mm_maskz_fnmadd_ph(__mmask8 __U, __m128h __A, __m128h __B, __m128h __C) {
  // CHECK-LABEL: @test_mm_maskz_fnmadd_ph
  // CHECK: fneg
  // CHECK: call <8 x half> @llvm.fma.v8f16(<8 x half> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}}
  return _mm_maskz_fnmadd_ph(__U, __A, __B, __C);
}

__m128h test_mm_maskz_fnmsub_ph(__mmask8 __U, __m128h __A, __m128h __B, __m128h __C) {
  // CHECK-LABEL: @test_mm_maskz_fnmsub_ph
  // CHECK: fneg
  // CHECK: fneg
  // CHECK: call <8 x half> @llvm.fma.v8f16(<8 x half> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}}
  return _mm_maskz_fnmsub_ph(__U, __A, __B, __C);
}

__m256h test_mm256_fmadd_ph(__m256h __A, __m256h __B, __m256h __C) {
  // CHECK-LABEL: @test_mm256_fmadd_ph
  // CHECK: call <16 x half> @llvm.fma.v16f16(<16 x half> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}})
  return _mm256_fmadd_ph(__A, __B, __C);
}

__m256h test_mm256_mask_fmadd_ph(__m256h __A, __mmask8 __U, __m256h __B, __m256h __C) {
  // CHECK-LABEL: @test_mm256_mask_fmadd_ph
  // CHECK: call <16 x half> @llvm.fma.v16f16(<16 x half> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return _mm256_mask_fmadd_ph(__A, __U, __B, __C);
}

__m256h test_mm256_fmsub_ph(__m256h __A, __m256h __B, __m256h __C) {
  // CHECK-LABEL: @test_mm256_fmsub_ph
  // CHECK: fneg
  // CHECK: call <16 x half> @llvm.fma.v16f16(<16 x half> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}})
  return _mm256_fmsub_ph(__A, __B, __C);
}

__m256h test_mm256_mask_fmsub_ph(__m256h __A, __mmask16 __U, __m256h __B, __m256h __C) {
  // CHECK-LABEL: @test_mm256_mask_fmsub_ph
  // CHECK: fneg
  // CHECK: call <16 x half> @llvm.fma.v16f16(<16 x half> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return _mm256_mask_fmsub_ph(__A, __U, __B, __C);
}

__m256h test_mm256_mask3_fmadd_ph(__m256h __A, __m256h __B, __m256h __C, __mmask16 __U) {
  // CHECK-LABEL: @test_mm256_mask3_fmadd_ph
  // CHECK: call <16 x half> @llvm.fma.v16f16(<16 x half> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return _mm256_mask3_fmadd_ph(__A, __B, __C, __U);
}

__m256h test_mm256_mask3_fnmadd_ph(__m256h __A, __m256h __B, __m256h __C, __mmask16 __U) {
  // CHECK-LABEL: @test_mm256_mask3_fnmadd_ph
  // CHECK: fneg
  // CHECK: call <16 x half> @llvm.fma.v16f16(<16 x half> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return _mm256_mask3_fnmadd_ph(__A, __B, __C, __U);
}

__m256h test_mm256_maskz_fmadd_ph(__mmask16 __U, __m256h __A, __m256h __B, __m256h __C) {
  // CHECK-LABEL: @test_mm256_maskz_fmadd_ph
  // CHECK: call <16 x half> @llvm.fma.v16f16(<16 x half> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return _mm256_maskz_fmadd_ph(__U, __A, __B, __C);
}

__m256h test_mm256_maskz_fmsub_ph(__mmask16 __U, __m256h __A, __m256h __B, __m256h __C) {
  // CHECK-LABEL: @test_mm256_maskz_fmsub_ph
  // CHECK: fneg
  // CHECK: call <16 x half> @llvm.fma.v16f16(<16 x half> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return _mm256_maskz_fmsub_ph(__U, __A, __B, __C);
}

__m256h test_mm256_maskz_fnmadd_ph(__mmask16 __U, __m256h __A, __m256h __B, __m256h __C) {
  // CHECK-LABEL: @test_mm256_maskz_fnmadd_ph
  // CHECK: fneg
  // CHECK: call <16 x half> @llvm.fma.v16f16(<16 x half> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return _mm256_maskz_fnmadd_ph(__U, __A, __B, __C);
}

__m256h test_mm256_maskz_fnmsub_ph(__mmask16 __U, __m256h __A, __m256h __B, __m256h __C) {
  // CHECK-LABEL: @test_mm256_maskz_fnmsub_ph
  // CHECK: fneg
  // CHECK: fneg
  // CHECK: call <16 x half> @llvm.fma.v16f16(<16 x half> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return _mm256_maskz_fnmsub_ph(__U, __A, __B, __C);
}

__m128h test_mm_fmaddsub_ph(__m128h __A, __m128h __B, __m128h __C) {
  // CHECK-LABEL: @test_mm_fmaddsub_ph
  // CHECK-NOT: fneg
  // CHECK: call <8 x half> @llvm.x86.avx512fp16.vfmaddsub.ph.128(<8 x half> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}})
  return _mm_fmaddsub_ph(__A, __B, __C);
}

__m128h test_mm_mask_fmaddsub_ph(__m128h __A, __mmask8 __U, __m128h __B, __m128h __C) {
  // CHECK-LABEL: @test_mm_mask_fmaddsub_ph
  // CHECK-NOT: fneg
  // CHECK: call <8 x half> @llvm.x86.avx512fp16.vfmaddsub.ph.128(<8 x half> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}}
  return _mm_mask_fmaddsub_ph(__A, __U, __B, __C);
}

__m128h test_mm_fmsubadd_ph(__m128h __A, __m128h __B, __m128h __C) {
  // CHECK-LABEL: @test_mm_fmsubadd_ph
  // CHECK: [[NEG:%.+]] = fneg
  // CHECK: call <8 x half> @llvm.x86.avx512fp16.vfmaddsub.ph.128(<8 x half> %{{.*}}, <8 x half> %{{.*}}, <8 x half> [[NEG]])
  return _mm_fmsubadd_ph(__A, __B, __C);
}

__m128h test_mm_mask_fmsubadd_ph(__m128h __A, __mmask8 __U, __m128h __B, __m128h __C) {
  // CHECK-LABEL: @test_mm_mask_fmsubadd_ph
  // CHECK: [[NEG:%.+]] = fneg
  // CHECK: call <8 x half> @llvm.x86.avx512fp16.vfmaddsub.ph.128(<8 x half> %{{.*}}, <8 x half> %{{.*}}, <8 x half> [[NEG]])
  // CHECK: select <8 x i1> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}}
  return _mm_mask_fmsubadd_ph(__A, __U, __B, __C);
}

__m128h test_mm_mask3_fmaddsub_ph(__m128h __A, __m128h __B, __m128h __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm_mask3_fmaddsub_ph
  // CHECK-NOT: fneg
  // CHECK: call <8 x half> @llvm.x86.avx512fp16.vfmaddsub.ph.128(<8 x half> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}}
  return _mm_mask3_fmaddsub_ph(__A, __B, __C, __U);
}

__m128h test_mm_maskz_fmaddsub_ph(__mmask8 __U, __m128h __A, __m128h __B, __m128h __C) {
  // CHECK-LABEL: @test_mm_maskz_fmaddsub_ph
  // CHECK-NOT: fneg
  // CHECK: call <8 x half> @llvm.x86.avx512fp16.vfmaddsub.ph.128(<8 x half> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}}
  return _mm_maskz_fmaddsub_ph(__U, __A, __B, __C);
}

__m128h test_mm_maskz_fmsubadd_ph(__mmask8 __U, __m128h __A, __m128h __B, __m128h __C) {
  // CHECK-LABEL: @test_mm_maskz_fmsubadd_ph
  // CHECK: [[NEG:%.+]] = fneg
  // CHECK: call <8 x half> @llvm.x86.avx512fp16.vfmaddsub.ph.128(<8 x half> %{{.*}}, <8 x half> %{{.*}}, <8 x half> [[NEG]])
  // CHECK: select <8 x i1> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}}
  return _mm_maskz_fmsubadd_ph(__U, __A, __B, __C);
}

__m256h test_mm256_fmaddsub_ph(__m256h __A, __m256h __B, __m256h __C) {
  // CHECK-LABEL: @test_mm256_fmaddsub_ph
  // CHECK-NOT: fneg
  // CHECK: call <16 x half> @llvm.x86.avx512fp16.vfmaddsub.ph.256(<16 x half> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}})
  return _mm256_fmaddsub_ph(__A, __B, __C);
}

__m256h test_mm256_mask_fmaddsub_ph(__m256h __A, __mmask16 __U, __m256h __B, __m256h __C) {
  // CHECK-LABEL: @test_mm256_mask_fmaddsub_ph
  // CHECK-NOT: fneg
  // CHECK: call <16 x half> @llvm.x86.avx512fp16.vfmaddsub.ph.256(<16 x half> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return _mm256_mask_fmaddsub_ph(__A, __U, __B, __C);
}

__m256h test_mm256_fmsubadd_ph(__m256h __A, __m256h __B, __m256h __C) {
  // CHECK-LABEL: @test_mm256_fmsubadd_ph
  // CHECK: [[NEG:%.+]] = fneg
  // CHECK: call <16 x half> @llvm.x86.avx512fp16.vfmaddsub.ph.256(<16 x half> %{{.*}}, <16 x half> %{{.*}}, <16 x half> [[NEG]])
  return _mm256_fmsubadd_ph(__A, __B, __C);
}

__m256h test_mm256_mask_fmsubadd_ph(__m256h __A, __mmask16 __U, __m256h __B, __m256h __C) {
  // CHECK-LABEL: @test_mm256_mask_fmsubadd_ph
  // CHECK: [[NEG:%.+]] = fneg
  // CHECK: call <16 x half> @llvm.x86.avx512fp16.vfmaddsub.ph.256(<16 x half> %{{.*}}, <16 x half> %{{.*}}, <16 x half> [[NEG]])
  // CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return _mm256_mask_fmsubadd_ph(__A, __U, __B, __C);
}

__m256h test_mm256_mask3_fmaddsub_ph(__m256h __A, __m256h __B, __m256h __C, __mmask16 __U) {
  // CHECK-LABEL: @test_mm256_mask3_fmaddsub_ph
  // CHECK-NOT: fneg
  // CHECK: call <16 x half> @llvm.x86.avx512fp16.vfmaddsub.ph.256(<16 x half> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return _mm256_mask3_fmaddsub_ph(__A, __B, __C, __U);
}

__m256h test_mm256_maskz_fmaddsub_ph(__mmask16 __U, __m256h __A, __m256h __B, __m256h __C) {
  // CHECK-LABEL: @test_mm256_maskz_fmaddsub_ph
  // CHECK-NOT: fneg
  // CHECK: call <16 x half> @llvm.x86.avx512fp16.vfmaddsub.ph.256(<16 x half> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return _mm256_maskz_fmaddsub_ph(__U, __A, __B, __C);
}

__m256h test_mm256_maskz_fmsubadd_ph(__mmask16 __U, __m256h __A, __m256h __B, __m256h __C) {
  // CHECK-LABEL: @test_mm256_maskz_fmsubadd_ph
  // CHECK: [[NEG:%.+]] = fneg
  // CHECK: call <16 x half> @llvm.x86.avx512fp16.vfmaddsub.ph.256(<16 x half> %{{.*}}, <16 x half> %{{.*}}, <16 x half> [[NEG]])
  // CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return _mm256_maskz_fmsubadd_ph(__U, __A, __B, __C);
}

__m128h test_mm_mask3_fmsub_ph(__m128h __A, __m128h __B, __m128h __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm_mask3_fmsub_ph
  // CHECK: fneg
  // CHECK: call <8 x half> @llvm.fma.v8f16(<8 x half> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}}
  return _mm_mask3_fmsub_ph(__A, __B, __C, __U);
}

__m256h test_mm256_mask3_fmsub_ph(__m256h __A, __m256h __B, __m256h __C, __mmask16 __U) {
  // CHECK-LABEL: @test_mm256_mask3_fmsub_ph
  // CHECK: fneg
  // CHECK: call <16 x half> @llvm.fma.v16f16(<16 x half> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return _mm256_mask3_fmsub_ph(__A, __B, __C, __U);
}

__m128h test_mm_mask3_fmsubadd_ph(__m128h __A, __m128h __B, __m128h __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm_mask3_fmsubadd_ph
  // CHECK: [[NEG:%.+]] = fneg
  // CHECK: call <8 x half> @llvm.x86.avx512fp16.vfmaddsub.ph.128(<8 x half> %{{.*}}, <8 x half> %{{.*}}, <8 x half> [[NEG]])
  // CHECK: select <8 x i1> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}}
  return _mm_mask3_fmsubadd_ph(__A, __B, __C, __U);
}

__m256h test_mm256_mask3_fmsubadd_ph(__m256h __A, __m256h __B, __m256h __C, __mmask16 __U) {
  // CHECK-LABEL: @test_mm256_mask3_fmsubadd_ph
  // CHECK: [[NEG:%.+]] = fneg
  // CHECK: call <16 x half> @llvm.x86.avx512fp16.vfmaddsub.ph.256(<16 x half> %{{.*}}, <16 x half> %{{.*}}, <16 x half> [[NEG]])
  // CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return _mm256_mask3_fmsubadd_ph(__A, __B, __C, __U);
}

__m128h test_mm_fnmadd_ph(__m128h __A, __m128h __B, __m128h __C) {
  // CHECK-LABEL: @test_mm_fnmadd_ph
  // CHECK: fneg
  // CHECK: call <8 x half> @llvm.fma.v8f16(<8 x half> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}})
  return _mm_fnmadd_ph(__A, __B, __C);
}

__m128h test_mm_mask_fnmadd_ph(__m128h __A, __mmask8 __U, __m128h __B, __m128h __C) {
  // CHECK-LABEL: @test_mm_mask_fnmadd_ph
  // CHECK: fneg
  // CHECK: call <8 x half> @llvm.fma.v8f16(<8 x half> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}}
  return _mm_mask_fnmadd_ph(__A, __U, __B, __C);
}

__m256h test_mm256_fnmadd_ph(__m256h __A, __m256h __B, __m256h __C) {
  // CHECK-LABEL: @test_mm256_fnmadd_ph
  // CHECK: fneg
  // CHECK: call <16 x half> @llvm.fma.v16f16(<16 x half> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}})
  return _mm256_fnmadd_ph(__A, __B, __C);
}

__m256h test_mm256_mask_fnmadd_ph(__m256h __A, __mmask16 __U, __m256h __B, __m256h __C) {
  // CHECK-LABEL: @test_mm256_mask_fnmadd_ph
  // CHECK: fneg
  // CHECK: call <16 x half> @llvm.fma.v16f16(<16 x half> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return _mm256_mask_fnmadd_ph(__A, __U, __B, __C);
}

__m128h test_mm_fnmsub_ph(__m128h __A, __m128h __B, __m128h __C) {
  // CHECK-LABEL: @test_mm_fnmsub_ph
  // CHECK: fneg
  // CHECK: fneg
  // CHECK: call <8 x half> @llvm.fma.v8f16(<8 x half> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}})
  return _mm_fnmsub_ph(__A, __B, __C);
}

__m128h test_mm_mask_fnmsub_ph(__m128h __A, __mmask8 __U, __m128h __B, __m128h __C) {
  // CHECK-LABEL: @test_mm_mask_fnmsub_ph
  // CHECK: fneg
  // CHECK: fneg
  // CHECK: call <8 x half> @llvm.fma.v8f16(<8 x half> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}}
  return _mm_mask_fnmsub_ph(__A, __U, __B, __C);
}

__m128h test_mm_mask3_fnmsub_ph(__m128h __A, __m128h __B, __m128h __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm_mask3_fnmsub_ph
  // CHECK: fneg
  // CHECK: fneg
  // CHECK: call <8 x half> @llvm.fma.v8f16(<8 x half> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}}
  return _mm_mask3_fnmsub_ph(__A, __B, __C, __U);
}

__m256h test_mm256_fnmsub_ph(__m256h __A, __m256h __B, __m256h __C) {
  // CHECK-LABEL: @test_mm256_fnmsub_ph
  // CHECK: fneg
  // CHECK: fneg
  // CHECK: call <16 x half> @llvm.fma.v16f16(<16 x half> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}})
  return _mm256_fnmsub_ph(__A, __B, __C);
}

__m256h test_mm256_mask_fnmsub_ph(__m256h __A, __mmask16 __U, __m256h __B, __m256h __C) {
  // CHECK-LABEL: @test_mm256_mask_fnmsub_ph
  // CHECK: fneg
  // CHECK: fneg
  // CHECK: call <16 x half> @llvm.fma.v16f16(<16 x half> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}}
  return _mm256_mask_fnmsub_ph(__A, __U, __B, __C);
}

__m256h test_mm256_mask3_fnmsub_ph(__m256h __A, __m256h __B, __m256h __C, __mmask16 __U) {
  // CHECK-LABEL: @test_mm256_mask3_fnmsub_ph
  // CHECK: fneg
  // CHECK: fneg
  // CHECK: call <16 x half> @llvm.fma.v16f16(<16 x half> %{{.*}}, <16 x half> %{{.*}}, <16 x half> %{{.*}})
  return _mm256_mask3_fnmsub_ph(__A, __B, __C, __U);
}

__m128h test_mm_fcmul_pch(__m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_fcmul_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfcmul.cph.128
  return _mm_fcmul_pch(__A, __B);
}

__m128h test_mm_mask_fcmul_pch(__m128h __W, __mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_mask_fcmul_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfcmul.cph.128
  return _mm_mask_fcmul_pch(__W, __U, __A, __B);
}

__m128h test_mm_maskz_fcmul_pch(__mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_maskz_fcmul_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfcmul.cph.128
  return _mm_maskz_fcmul_pch(__U, __A, __B);
}

__m256h test_mm256_fcmul_pch(__m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_fcmul_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfcmul.cph.256
  return _mm256_fcmul_pch(__A, __B);
}

__m256h test_mm256_mask_fcmul_pch(__m256h __W, __mmask8 __U, __m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_mask_fcmul_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfcmul.cph.256
  return _mm256_mask_fcmul_pch(__W, __U, __A, __B);
}

__m256h test_mm256_maskz_fcmul_pch(__mmask8 __U, __m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_maskz_fcmul_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfcmul.cph.256
  return _mm256_maskz_fcmul_pch(__U, __A, __B);
}

__m128h test_mm_fcmadd_pch(__m128h __A, __m128h __B, __m128h __C) {
  // CHECK-LABEL: @test_mm_fcmadd_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfcmadd.cph.128
  return _mm_fcmadd_pch(__A, __B, __C);
}

__m128h test_mm_mask_fcmadd_pch(__m128h __A, __mmask8 __U, __m128h __B, __m128h __C) {
  // CHECK-LABEL: @test_mm_mask_fcmadd_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfcmadd.cph.128
  // CHECK:  %{{.*}} = shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK:  %{{.*}} = select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_mask_fcmadd_pch(__A, __U, __B, __C);
}

__m128h test_mm_mask3_fcmadd_pch(__m128h __A, __m128h __B, __m128h __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm_mask3_fcmadd_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfcmadd.cph.128
  // CHECK-NOT:  %{{.*}} = select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_mask3_fcmadd_pch(__A, __B, __C, __U);
}

__m128h test_mm_maskz_fcmadd_pch(__mmask8 __U, __m128h __A, __m128h __B, __m128h __C) {
  // CHECK-LABEL: @test_mm_maskz_fcmadd_pch
  // CHECK: @llvm.x86.avx512fp16.maskz.vfcmadd.cph.128
  return _mm_maskz_fcmadd_pch(__U, __A, __B, __C);
}

__m256h test_mm256_fcmadd_pch(__m256h __A, __m256h __B, __m256h __C) {
  // CHECK-LABEL: @test_mm256_fcmadd_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfcmadd.cph.256
  return _mm256_fcmadd_pch(__A, __B, __C);
}

__m256h test_mm256_mask_fcmadd_pch(__m256h __A, __mmask8 __U, __m256h __B, __m256h __C) {
  // CHECK-LABEL: @test_mm256_mask_fcmadd_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfcmadd.cph.256
  // CHECK:  %{{.*}} = select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_fcmadd_pch(__A, __U, __B, __C);
}

__m256h test_mm256_mask3_fcmadd_pch(__m256h __A, __m256h __B, __m256h __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm256_mask3_fcmadd_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfcmadd.cph.256
  // CHECK-NOT:  %{{.*}} = select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask3_fcmadd_pch(__A, __B, __C, __U);
}

__m256h test_mm256_maskz_fcmadd_pch(__mmask8 __U, __m256h __A, __m256h __B, __m256h __C) {
  // CHECK-LABEL: @test_mm256_maskz_fcmadd_pch
  // CHECK: @llvm.x86.avx512fp16.maskz.vfcmadd.cph.256
  return _mm256_maskz_fcmadd_pch(__U, __A, __B, __C);
}

__m128h test_mm_fmul_pch(__m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_fmul_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfmul.cph.128
  return _mm_fmul_pch(__A, __B);
}

__m128h test_mm_mask_fmul_pch(__m128h __W, __mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_mask_fmul_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfmul.cph.128
  return _mm_mask_fmul_pch(__W, __U, __A, __B);
}

__m128h test_mm_maskz_fmul_pch(__mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_maskz_fmul_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfmul.cph.128
  return _mm_maskz_fmul_pch(__U, __A, __B);
}

__m256h test_mm256_fmul_pch(__m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_fmul_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfmul.cph.256
  return _mm256_fmul_pch(__A, __B);
}

__m256h test_mm256_mask_fmul_pch(__m256h __W, __mmask8 __U, __m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_mask_fmul_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfmul.cph.256
  return _mm256_mask_fmul_pch(__W, __U, __A, __B);
}

__m256h test_mm256_maskz_fmul_pch(__mmask8 __U, __m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_maskz_fmul_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfmul.cph.256
  return _mm256_maskz_fmul_pch(__U, __A, __B);
}

__m128h test_mm_fmadd_pch(__m128h __A, __m128h __B, __m128h __C) {
  // CHECK-LABEL: @test_mm_fmadd_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfmadd.cph.128
  return _mm_fmadd_pch(__A, __B, __C);
}

__m128h test_mm_mask_fmadd_pch(__m128h __A, __mmask8 __U, __m128h __B, __m128h __C) {
  // CHECK-LABEL: @test_mm_mask_fmadd_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfmadd.cph.128
  // CHECK:  %{{.*}} = shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK:  %{{.*}} = select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_mask_fmadd_pch(__A, __U, __B, __C);
}

__m128h test_mm_mask3_fmadd_pch(__m128h __A, __m128h __B, __m128h __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm_mask3_fmadd_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfmadd.cph.128
  return _mm_mask3_fmadd_pch(__A, __B, __C, __U);
}

__m128h test_mm_maskz_fmadd_pch(__mmask8 __U, __m128h __A, __m128h __B, __m128h __C) {
  // CHECK-LABEL: @test_mm_maskz_fmadd_pch
  // CHECK: @llvm.x86.avx512fp16.maskz.vfmadd.cph.128
  return _mm_maskz_fmadd_pch(__U, __A, __B, __C);
}

__m256h test_mm256_fmadd_pch(__m256h __A, __m256h __B, __m256h __C) {
  // CHECK-LABEL: @test_mm256_fmadd_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfmadd.cph.256
  return _mm256_fmadd_pch(__A, __B, __C);
}

__m256h test_mm256_mask_fmadd_pch(__m256h __A, __mmask8 __U, __m256h __B, __m256h __C) {
  // CHECK-LABEL: @test_mm256_mask_fmadd_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfmadd.cph.256
  // CHECK:  %{{.*}} = select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_fmadd_pch(__A, __U, __B, __C);
}

__m256h test_mm256_mask3_fmadd_pch(__m256h __A, __m256h __B, __m256h __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm256_mask3_fmadd_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfmadd.cph.256
  return _mm256_mask3_fmadd_pch(__A, __B, __C, __U);
}

__m256h test_mm256_maskz_fmadd_pch(__mmask8 __U, __m256h __A, __m256h __B, __m256h __C) {
  // CHECK-LABEL: @test_mm256_maskz_fmadd_pch
  // CHECK: @llvm.x86.avx512fp16.maskz.vfmadd.cph.256
  return _mm256_maskz_fmadd_pch(__U, __A, __B, __C);
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

_Float16 test_mm256_reduce_add_ph(__m256h __W) {
  // CHECK-LABEL: @test_mm256_reduce_add_ph
  // CHECK: call reassoc half @llvm.vector.reduce.fadd.v16f16(half 0xH8000, <16 x half> %{{.*}})
  return _mm256_reduce_add_ph(__W);
}

_Float16 test_mm256_reduce_mul_ph(__m256h __W) {
  // CHECK-LABEL: @test_mm256_reduce_mul_ph
  // CHECK: call reassoc half @llvm.vector.reduce.fmul.v16f16(half 0xH3C00, <16 x half> %{{.*}})
  return _mm256_reduce_mul_ph(__W);
}

_Float16 test_mm256_reduce_max_ph(__m256h __W) {
  // CHECK-LABEL: @test_mm256_reduce_max_ph
  // CHECK: call nnan half @llvm.vector.reduce.fmax.v16f16(<16 x half> %{{.*}})
  return _mm256_reduce_max_ph(__W);
}

_Float16 test_mm256_reduce_min_ph(__m256h __W) {
  // CHECK-LABEL: @test_mm256_reduce_min_ph
  // CHECK: call nnan half @llvm.vector.reduce.fmin.v16f16(<16 x half> %{{.*}})
  return _mm256_reduce_min_ph(__W);
}

_Float16 test_mm_reduce_add_ph(__m128h __W) {
  // CHECK-LABEL: @test_mm_reduce_add_ph
  // CHECK: call reassoc half @llvm.vector.reduce.fadd.v8f16(half 0xH8000, <8 x half> %{{.*}})
  return _mm_reduce_add_ph(__W);
}

_Float16 test_mm_reduce_mul_ph(__m128h __W) {
  // CHECK-LABEL: @test_mm_reduce_mul_ph
  // CHECK: call reassoc half @llvm.vector.reduce.fmul.v8f16(half 0xH3C00, <8 x half> %{{.*}})
  return _mm_reduce_mul_ph(__W);
}

_Float16 test_mm_reduce_min_ph(__m128h __W) {
  // CHECK-LABEL: @test_mm_reduce_min_ph
  // CHECK: call nnan half @llvm.vector.reduce.fmin.v8f16(<8 x half> %{{.*}})
  return _mm_reduce_min_ph(__W);
}

_Float16 test_mm_reduce_max_ph(__m128h __W) {
  // CHECK-LABEL: @test_mm_reduce_max_ph
  // CHECK: call nnan half @llvm.vector.reduce.fmax.v8f16(<8 x half> %{{.*}})
  return _mm_reduce_max_ph(__W);
}

// tests below are for alias intrinsics.
__m128h test_mm_mul_pch(__m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_mul_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfmul.cph.128
  return _mm_mul_pch(__A, __B);
}

__m128h test_mm_mask_mul_pch(__m128h __W, __mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_mask_mul_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfmul.cph.128
  return _mm_mask_mul_pch(__W, __U, __A, __B);
}

__m128h test_mm_maskz_mul_pch(__mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_maskz_mul_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfmul.cph.128
  return _mm_maskz_mul_pch(__U, __A, __B);
}

__m256h test_mm256_mul_pch(__m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_mul_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfmul.cph.256
  return _mm256_mul_pch(__A, __B);
}

__m256h test_mm256_mask_mul_pch(__m256h __W, __mmask8 __U, __m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_mask_mul_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfmul.cph.256
  return _mm256_mask_mul_pch(__W, __U, __A, __B);
}

__m256h test_mm256_maskz_mul_pch(__mmask8 __U, __m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_maskz_mul_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfmul.cph.256
  return _mm256_maskz_mul_pch(__U, __A, __B);
}

__m128h test_mm_cmul_pch(__m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_cmul_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfcmul.cph.128
  return _mm_cmul_pch(__A, __B);
}

__m128h test_mm_mask_cmul_pch(__m128h __W, __mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_mask_cmul_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfcmul.cph.128
  return _mm_mask_fcmul_pch(__W, __U, __A, __B);
}

__m128h test_mm_maskz_cmul_pch(__mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: @test_mm_maskz_cmul_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfcmul.cph.128
  return _mm_maskz_cmul_pch(__U, __A, __B);
}

__m256h test_mm256_cmul_pch(__m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_cmul_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfcmul.cph.256
  return _mm256_cmul_pch(__A, __B);
}

__m256h test_mm256_mask_cmul_pch(__m256h __W, __mmask8 __U, __m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_mask_cmul_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfcmul.cph.256
  return _mm256_mask_cmul_pch(__W, __U, __A, __B);
}

__m256h test_mm256_maskz_cmul_pch(__mmask8 __U, __m256h __A, __m256h __B) {
  // CHECK-LABEL: @test_mm256_maskz_cmul_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfcmul.cph.256
  return _mm256_maskz_cmul_pch(__U, __A, __B);
}

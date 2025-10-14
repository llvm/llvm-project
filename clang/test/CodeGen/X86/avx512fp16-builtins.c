// RUN: %clang_cc1 -x c -ffreestanding -flax-vector-conversions=none %s -triple=x86_64-unknown-unknown -target-feature +avx512fp16 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK,X64
// RUN: %clang_cc1 -x c -ffreestanding -flax-vector-conversions=none %s -triple=i686-unknown-unknown -target-feature +avx512fp16 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -x c++ -ffreestanding -flax-vector-conversions=none %s -triple=x86_64-unknown-unknown -target-feature +avx512fp16 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK,X64
// RUN: %clang_cc1 -x c++ -ffreestanding -flax-vector-conversions=none %s -triple=i686-unknown-unknown -target-feature +avx512fp16 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK

// RUN: %clang_cc1 -x c -ffreestanding -flax-vector-conversions=none %s -triple=x86_64-unknown-unknown -target-feature +avx512fp16 -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s --check-prefixes=CHECK,X64
// RUN: %clang_cc1 -x c -ffreestanding -flax-vector-conversions=none %s -triple=i686-unknown-unknown -target-feature +avx512fp16 -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -x c++ -ffreestanding -flax-vector-conversions=none %s -triple=x86_64-unknown-unknown -target-feature +avx512fp16 -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s --check-prefixes=CHECK,X64
// RUN: %clang_cc1 -x c++ -ffreestanding -flax-vector-conversions=none %s -triple=i686-unknown-unknown -target-feature +avx512fp16 -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s --check-prefixes=CHECK


#include <immintrin.h>
#include "builtin_test_helpers.h"

_Float16 test_mm512_cvtsh_h(__m512h __A) {
  // CHECK-LABEL: test_mm512_cvtsh_h
  // CHECK: extractelement <32 x half> %{{.*}}, i32 0
  return _mm512_cvtsh_h(__A);
}
TEST_CONSTEXPR(_mm512_cvtsh_h((__m512h){-32.0, 31.0, -30.0, 29.0, -28.0, 27.0, -26.0, 25.0, -24.0, 23.0, -22.0, 21.0, -20.0, 19.0, -18.0, 17.0, -16.0, 15.0, -14.0, 13.0, -12.0, 11.0, -10.0, 9.0, -8.0, 7.0, -6.0, 5.0, -4.0, 3.0, -2.0, 1.0}) == -32.0);

__m128h test_mm_setzero_ph(void) {
  // CHECK-LABEL: test_mm_setzero_ph
  // CHECK: zeroinitializer
  return _mm_setzero_ph();
}
TEST_CONSTEXPR(match_m128h(_mm_setzero_ph(), +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f));

__m256h test_mm256_setzero_ph(void) {
  // CHECK-LABEL: test_mm256_setzero_ph
  // CHECK: zeroinitializer
  return _mm256_setzero_ph();
}
TEST_CONSTEXPR(match_m256h(_mm256_setzero_ph(), +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f));

__m256h test_mm256_undefined_ph(void) {
  // CHECK-LABEL: test_mm256_undefined_ph
  // CHECK: ret <16 x half> zeroinitializer
  return _mm256_undefined_ph();
}

__m512h test_mm512_setzero_ph(void) {
  // CHECK-LABEL: test_mm512_setzero_ph
  // CHECK: zeroinitializer
  return _mm512_setzero_ph();
}
TEST_CONSTEXPR(match_m512h(_mm512_setzero_ph(), +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f, +0.0f));

__m128h test_mm_undefined_ph(void) {
  // CHECK-LABEL: test_mm_undefined_ph
  // CHECK: ret <8 x half> zeroinitializer
  return _mm_undefined_ph();
}

__m512h test_mm512_undefined_ph(void) {
  // CHECK-LABEL: test_mm512_undefined_ph
  // CHECK: ret <32 x half> zeroinitializer
  return _mm512_undefined_ph();
}

__m512h test_mm512_set1_ph(_Float16 h) {
  // CHECK-LABEL: test_mm512_set1_ph
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
TEST_CONSTEXPR(match_m512h(_mm512_set1_ph(-101.0f), -101.0f, -101.0f, -101.0f, -101.0f, -101.0f, -101.0f, -101.0f, -101.0f, -101.0f, -101.0f, -101.0f, -101.0f, -101.0f, -101.0f, -101.0f, -101.0f, -101.0f, -101.0f, -101.0f, -101.0f, -101.0f, -101.0f, -101.0f, -101.0f, -101.0f, -101.0f, -101.0f, -101.0f, -101.0f, -101.0f, -101.0f, -101.0f));

__m512h test_mm512_set1_pch(_Float16 _Complex h) {
  // CHECK-LABEL: test_mm512_set1_pch
  // CHECK: insertelement <16 x float> {{.*}}, i32 0
  // CHECK: insertelement <16 x float> {{.*}}, i32 1
  // CHECK: insertelement <16 x float> {{.*}}, i32 2
  // CHECK: insertelement <16 x float> {{.*}}, i32 3
  // CHECK: insertelement <16 x float> {{.*}}, i32 4
  // CHECK: insertelement <16 x float> {{.*}}, i32 5
  // CHECK: insertelement <16 x float> {{.*}}, i32 6
  // CHECK: insertelement <16 x float> {{.*}}, i32 7
  // CHECK: insertelement <16 x float> {{.*}}, i32 8
  // CHECK: insertelement <16 x float> {{.*}}, i32 9
  // CHECK: insertelement <16 x float> {{.*}}, i32 10
  // CHECK: insertelement <16 x float> {{.*}}, i32 11
  // CHECK: insertelement <16 x float> {{.*}}, i32 12
  // CHECK: insertelement <16 x float> {{.*}}, i32 13
  // CHECK: insertelement <16 x float> {{.*}}, i32 14
  // CHECK: insertelement <16 x float> {{.*}}, i32 15
  // CHECK: bitcast <16 x float>{{.*}} to <32 x half>
  return _mm512_set1_pch(h);
}
TEST_CONSTEXPR(match_m512h(_mm512_set1_pch(1.0), 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0));

__m512h test_mm512_set_ph(_Float16 __h1, _Float16 __h2, _Float16 __h3, _Float16 __h4,
                          _Float16 __h5, _Float16 __h6, _Float16 __h7, _Float16 __h8,
                          _Float16 __h9, _Float16 __h10, _Float16 __h11, _Float16 __h12,
                          _Float16 __h13, _Float16 __h14, _Float16 __h15, _Float16 __h16,
                          _Float16 __h17, _Float16 __h18, _Float16 __h19, _Float16 __h20,
                          _Float16 __h21, _Float16 __h22, _Float16 __h23, _Float16 __h24,
                          _Float16 __h25, _Float16 __h26, _Float16 __h27, _Float16 __h28,
                          _Float16 __h29, _Float16 __h30, _Float16 __h31, _Float16 __h32) {
  // CHECK-LABEL: test_mm512_set_ph
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
TEST_CONSTEXPR(match_m512h(_mm512_set_ph(32.0f, -31.0f, 30.0f, -29.0f, 28.0f, -27.0f, 26.0f, -25.0f,
                                        24.0f, -23.0f, 22.0f, -21.0f, 20.0f, -19.0f, 18.0f, -17.0f,
                                        16.0f, -15.0f, 14.0f, -13.0f, 12.0f, -11.0f, 10.0f, -9.0f,
                                        8.0f, -7.0f, 6.0f, -5.0f, 4.0f, -3.0f, 2.0f, -1.0f), -1.0f, 2.0f, -3.0f, 4.0f, -5.0f, 6.0f, -7.0f, 8.0f,
                                        -9.0f, 10.0f, -11.0f, 12.0f, -13.0f, 14.0f, -15.0f, 16.0f,
                                        -17.0f, 18.0f, -19.0f, 20.0f, -21.0f, 22.0f, -23.0f, 24.0f,
                                        -25.0f, 26.0f, -27.0f, 28.0f, -29.0f, 30.0f, -31.0f, 32.0f));

__m512h test_mm512_setr_ph(_Float16 __h1, _Float16 __h2, _Float16 __h3, _Float16 __h4,
                           _Float16 __h5, _Float16 __h6, _Float16 __h7, _Float16 __h8,
                           _Float16 __h9, _Float16 __h10, _Float16 __h11, _Float16 __h12,
                           _Float16 __h13, _Float16 __h14, _Float16 __h15, _Float16 __h16,
                           _Float16 __h17, _Float16 __h18, _Float16 __h19, _Float16 __h20,
                           _Float16 __h21, _Float16 __h22, _Float16 __h23, _Float16 __h24,
                           _Float16 __h25, _Float16 __h26, _Float16 __h27, _Float16 __h28,
                           _Float16 __h29, _Float16 __h30, _Float16 __h31, _Float16 __h32) {
  // CHECK-LABEL: test_mm512_setr_ph
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
TEST_CONSTEXPR(match_m512h(_mm512_setr_ph(1.0f, -3.0f, 5.0f, -7.0f, 9.0f, -11.0f, 13.0f, -15.0f,
                                         17.0f, -19.0f, 21.0f, -23.0f, 25.0f, -27.0f, 29.0f, -31.0f,
                                         33.0f, -35.0f, 37.0f, -39.0f, 41.0f, -43.0f, 45.0f, -47.0f,
                                         49.0f, -51.0f, 53.0f, -55.0f, 57.0f, -59.0f, 61.0f, -63.0f),
                          1.0f, -3.0f, 5.0f, -7.0f, 9.0f, -11.0f, 13.0f, -15.0f,
                          17.0f, -19.0f, 21.0f, -23.0f, 25.0f, -27.0f, 29.0f, -31.0f,
                          33.0f, -35.0f, 37.0f, -39.0f, 41.0f, -43.0f, 45.0f, -47.0f,
                          49.0f, -51.0f, 53.0f, -55.0f, 57.0f, -59.0f, 61.0f, -63.0f));

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
TEST_CONSTEXPR(match_m128h(_mm256_castph256_ph128((__m256h){-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0, 10.0, -11.0, 12.0, -13.0, 14.0, -15.0, -16.0}), -1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0));

__m128h test_mm512_castph512_ph128(__m512h __a) {
  // CHECK-LABEL: test_mm512_castph512_ph128
  // CHECK: shufflevector <32 x half> %{{.*}}, <32 x half> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  return _mm512_castph512_ph128(__a);
}
TEST_CONSTEXPR(match_m128h(_mm512_castph512_ph128((__m512h){0.0, -1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0, 10.0, -11.0, 12.0, -13.0, 14.0, -15.0, -16.0, -17.0, 18.0, -19.0, 20.0, -21.0, 22.0, -23.0, 24.0, -25.0, 26.0, -27.0, 28.0, -29.0, 30.0, -31.0}), 0.0, -1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0));

__m256h test_mm512_castph512_ph256(__m512h __a) {
  // CHECK-LABEL: test_mm512_castph512_ph256
  // CHECK: shufflevector <32 x half> %{{.*}}, <32 x half> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  return _mm512_castph512_ph256(__a);
}
TEST_CONSTEXPR(match_m256h(_mm512_castph512_ph256((__m512h){-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0, 10.0, -11.0, 12.0, -13.0, 14.0, -15.0, -16.0, -17.0, 18.0, -19.0, 20.0, -21.0, 22.0, -23.0, 24.0, -25.0, 26.0, -27.0, 28.0, -29.0, 30.0, -31.0, 32.0}), -1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0, 10.0, -11.0, 12.0, -13.0, 14.0, -15.0, -16.0));

__m256h test_mm256_castph128_ph256(__m128h __a) {
  // CHECK-LABEL: test_mm256_castph128_ph256
  // CHECK: [[A:%.*]] = freeze <8 x half> poison 
  // CHECK: shufflevector <8 x half> %{{.*}}, <8 x half> [[A]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  return _mm256_castph128_ph256(__a);
}

__m512h test_mm512_castph128_ph512(__m128h __a) {
  // CHECK-LABEL: test_mm512_castph128_ph512
  // CHECK: [[B:%.*]] = freeze <16 x half> poison
  // CHECK: store <16 x half> [[B]], ptr [[BA:%.*]]
  // CHECK: [[A:%.*]] = freeze <8 x half> poison
  // CHECK: [[SV:%.*]] = shufflevector <8 x half> %{{.*}}, <8 x half> [[A]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  // CHECK: [[C:%.*]] = load <16 x half>, ptr [[BA]]
  // CHECK: shufflevector <16 x half> [[SV]], <16 x half> [[C]], <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  return _mm512_castph128_ph512(__a);
}

__m512h test_mm512_castph256_ph512(__m256h __a) {
  // CHECK-LABEL: test_mm512_castph256_ph512
  // CHECK: [[A:%.*]] = freeze <16 x half> poison 
  // CHECK: shufflevector <16 x half> %{{.*}}, <16 x half> [[A]], <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  return _mm512_castph256_ph512(__a);
}

__m256h test_mm256_zextph128_ph256(__m128h __a) {
  // CHECK-LABEL: test_mm256_zextph128_ph256
  // CHECK: shufflevector <8 x half> %{{.*}}, <8 x half> {{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  return _mm256_zextph128_ph256(__a);
}
TEST_CONSTEXPR(match_m256h(_mm256_zextph128_ph256((__m128h){1.0f16, 2.0f16, 3.0f16, 4.0f16, 5.0f16, 6.0f16, 7.0f16, 8.0f16}), 1.0f16, 2.0f16, 3.0f16, 4.0f16, 5.0f16, 6.0f16, 7.0f16, 8.0f16, 0.0f16, 0.0f16, 0.0f16, 0.0f16, 0.0f16, 0.0f16, 0.0f16, 0.0f16));

__m512h test_mm512_zextph128_ph512(__m128h __a) {
  // CHECK-LABEL: test_mm512_zextph128_ph512
  // CHECK: shufflevector <8 x half> %{{.*}}, <8 x half> {{.*}}, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  return _mm512_zextph128_ph512(__a);
}
TEST_CONSTEXPR(match_m512h(_mm512_zextph128_ph512((__m128h){1.0f16, 2.0f16, 3.0f16, 4.0f16, 5.0f16, 6.0f16, 7.0f16, 8.0f16}), 1.0f16, 2.0f16, 3.0f16, 4.0f16, 5.0f16, 6.0f16, 7.0f16, 8.0f16, 0.0f16, 0.0f16, 0.0f16, 0.0f16, 0.0f16, 0.0f16, 0.0f16, 0.0f16, 0.0f16, 0.0f16, 0.0f16, 0.0f16, 0.0f16, 0.0f16, 0.0f16, 0.0f16, 0.0f16, 0.0f16, 0.0f16, 0.0f16, 0.0f16, 0.0f16, 0.0f16, 0.0f16));

__m512h test_mm512_zextph256_ph512(__m256h __a) {
  // CHECK-LABEL: test_mm512_zextph256_ph512
  // CHECK: shufflevector <16 x half> %{{.*}}, <16 x half> {{.*}}, <32 x i32>
  return _mm512_zextph256_ph512(__a);
}
TEST_CONSTEXPR(match_m512h(_mm512_zextph256_ph512((__m256h){1.0f16, 2.0f16, 3.0f16, 4.0f16, 5.0f16, 6.0f16, 7.0f16, 8.0f16, 9.0f16, 10.0f16, 11.0f16, 12.0f16, 13.0f16, 14.0f16, 15.0f16, 16.0f16}), 1.0f16, 2.0f16, 3.0f16, 4.0f16, 5.0f16, 6.0f16, 7.0f16, 8.0f16, 9.0f16, 10.0f16, 11.0f16, 12.0f16, 13.0f16, 14.0f16, 15.0f16, 16.0f16, 0.0f16, 0.0f16, 0.0f16, 0.0f16, 0.0f16, 0.0f16, 0.0f16, 0.0f16, 0.0f16, 0.0f16, 0.0f16, 0.0f16, 0.0f16, 0.0f16, 0.0f16, 0.0f16));

int test_mm_comi_round_sh(__m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_comi_round_sh
  // CHECK: @llvm.x86.avx512fp16.vcomi.sh(<8 x half> %{{.}}, <8 x half> %{{.}}, i32 0, i32 8)
  return _mm_comi_round_sh(__A, __B, 0, _MM_FROUND_NO_EXC);
}

int test_mm_comi_sh(__m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_comi_sh
  // CHECK: @llvm.x86.avx512fp16.vcomi.sh(<8 x half> %{{.}}, <8 x half> %{{.}}, i32 0, i32 4)
  return _mm_comi_sh(__A, __B, 0);
}

int test_mm_comieq_sh(__m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_comieq_sh
  // CHECK: @llvm.x86.avx512fp16.vcomi.sh(<8 x half> %{{.}}, <8 x half> %{{.}}, i32 16, i32 4)
  return _mm_comieq_sh(__A, __B);
}

int test_mm_comilt_sh(__m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_comilt_sh
  // CHECK: @llvm.x86.avx512fp16.vcomi.sh(<8 x half> %{{.}}, <8 x half> %{{.}}, i32 1, i32 4)
  return _mm_comilt_sh(__A, __B);
}

int test_mm_comile_sh(__m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_comile_sh
  // CHECK: @llvm.x86.avx512fp16.vcomi.sh(<8 x half> %{{.}}, <8 x half> %{{.}}, i32 2, i32 4)
  return _mm_comile_sh(__A, __B);
}

int test_mm_comigt_sh(__m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_comigt_sh
  // CHECK: @llvm.x86.avx512fp16.vcomi.sh(<8 x half> %{{.}}, <8 x half> %{{.}}, i32 14, i32 4)
  return _mm_comigt_sh(__A, __B);
}

int test_mm_comige_sh(__m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_comige_sh
  // CHECK: @llvm.x86.avx512fp16.vcomi.sh(<8 x half> %{{.}}, <8 x half> %{{.}}, i32 13, i32 4)
  return _mm_comige_sh(__A, __B);
}

int test_mm_comineq_sh(__m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_comineq_sh
  // CHECK: @llvm.x86.avx512fp16.vcomi.sh(<8 x half> %{{.}}, <8 x half> %{{.}}, i32 20, i32 4)
  return _mm_comineq_sh(__A, __B);
}

int test_mm_ucomieq_sh(__m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_ucomieq_sh
  // CHECK: @llvm.x86.avx512fp16.vcomi.sh(<8 x half> %{{.}}, <8 x half> %{{.}}, i32 0, i32 4)
  return _mm_ucomieq_sh(__A, __B);
}

int test_mm_ucomilt_sh(__m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_ucomilt_sh
  // CHECK: @llvm.x86.avx512fp16.vcomi.sh(<8 x half> %{{.}}, <8 x half> %{{.}}, i32 17, i32 4)
  return _mm_ucomilt_sh(__A, __B);
}

int test_mm_ucomile_sh(__m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_ucomile_sh
  // CHECK: @llvm.x86.avx512fp16.vcomi.sh(<8 x half> %{{.}}, <8 x half> %{{.}}, i32 18, i32 4)
  return _mm_ucomile_sh(__A, __B);
}

int test_mm_ucomigt_sh(__m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_ucomigt_sh
  // CHECK: @llvm.x86.avx512fp16.vcomi.sh(<8 x half> %{{.}}, <8 x half> %{{.}}, i32 30, i32 4)
  return _mm_ucomigt_sh(__A, __B);
}

int test_mm_ucomige_sh(__m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_ucomige_sh
  // CHECK: @llvm.x86.avx512fp16.vcomi.sh(<8 x half> %{{.}}, <8 x half> %{{.}}, i32 29, i32 4)
  return _mm_ucomige_sh(__A, __B);
}

int test_mm_ucomineq_sh(__m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_ucomineq_sh
  // CHECK: @llvm.x86.avx512fp16.vcomi.sh(<8 x half> %{{.}}, <8 x half> %{{.}}, i32 4, i32 4)
  return _mm_ucomineq_sh(__A, __B);
}

__m512h test_mm512_add_ph(__m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_add_ph
  // CHECK: %{{.*}} = fadd <32 x half> %{{.*}}, %{{.*}}
  return _mm512_add_ph(__A, __B);
}

__m512h test_mm512_mask_add_ph(__m512h __W, __mmask32 __U, __m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_mask_add_ph
  // CHECK: %{{.*}} = fadd <32 x half> %{{.*}}, %{{.*}}
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}
  return (__m512h)_mm512_mask_add_ph(__W, __U, __A, __B);
}

__m512h test_mm512_maskz_add_ph(__mmask32 __U, __m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_maskz_add_ph
  // CHECK: %{{.*}} = fadd <32 x half> %{{.*}}, %{{.*}}
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}
  return _mm512_maskz_add_ph(__U, __A, __B);
}

__m512h test_mm512_add_round_ph(__m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_add_round_ph
  // CHECK: @llvm.x86.avx512fp16.add.ph.512
  return _mm512_add_round_ph(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512h test_mm512_mask_add_round_ph(__m512h __W, __mmask32 __U, __m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_mask_add_round_ph
  // CHECK: @llvm.x86.avx512fp16.add.ph.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}
  return _mm512_mask_add_round_ph(__W, __U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512h test_mm512_maskz_add_round_ph(__mmask32 __U, __m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_maskz_add_round_ph
  // CHECK: @llvm.x86.avx512fp16.add.ph.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}
  return _mm512_maskz_add_round_ph(__U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_sub_ph(__m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_sub_ph
  // CHECK: %{{.*}} = fsub <32 x half> %{{.*}}, %{{.*}}
  return _mm512_sub_ph(__A, __B);
}

__m512h test_mm512_mask_sub_ph(__m512h __W, __mmask32 __U, __m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_mask_sub_ph
  // CHECK: %{{.*}} = fsub <32 x half> %{{.*}}, %{{.*}}
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}
  return (__m512h)_mm512_mask_sub_ph(__W, __U, __A, __B);
}

__m512h test_mm512_maskz_sub_ph(__mmask32 __U, __m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_maskz_sub_ph
  // CHECK: %{{.*}} = fsub <32 x half> %{{.*}}, %{{.*}}
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}
  return _mm512_maskz_sub_ph(__U, __A, __B);
}

__m512h test_mm512_sub_round_ph(__m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_sub_round_ph
  // CHECK: @llvm.x86.avx512fp16.sub.ph.512
  return _mm512_sub_round_ph(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512h test_mm512_mask_sub_round_ph(__m512h __W, __mmask32 __U, __m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_mask_sub_round_ph
  // CHECK: @llvm.x86.avx512fp16.sub.ph.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}
  return _mm512_mask_sub_round_ph(__W, __U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512h test_mm512_maskz_sub_round_ph(__mmask32 __U, __m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_maskz_sub_round_ph
  // CHECK: @llvm.x86.avx512fp16.sub.ph.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}
  return _mm512_maskz_sub_round_ph(__U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_mul_ph(__m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_mul_ph
  // CHECK: %{{.*}} = fmul <32 x half> %{{.*}}, %{{.*}}
  return _mm512_mul_ph(__A, __B);
}

__m512h test_mm512_mask_mul_ph(__m512h __W, __mmask32 __U, __m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_mask_mul_ph
  // CHECK: %{{.*}} = fmul <32 x half> %{{.*}}, %{{.*}}
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}
  return (__m512h)_mm512_mask_mul_ph(__W, __U, __A, __B);
}

__m512h test_mm512_maskz_mul_ph(__mmask32 __U, __m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_maskz_mul_ph
  // CHECK: %{{.*}} = fmul <32 x half> %{{.*}}, %{{.*}}
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}
  return _mm512_maskz_mul_ph(__U, __A, __B);
}

__m512h test_mm512_mul_round_ph(__m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_mul_round_ph
  // CHECK: @llvm.x86.avx512fp16.mul.ph.512
  return _mm512_mul_round_ph(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512h test_mm512_mask_mul_round_ph(__m512h __W, __mmask32 __U, __m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_mask_mul_round_ph
  // CHECK: @llvm.x86.avx512fp16.mul.ph.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}
  return _mm512_mask_mul_round_ph(__W, __U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512h test_mm512_maskz_mul_round_ph(__mmask32 __U, __m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_maskz_mul_round_ph
  // CHECK: @llvm.x86.avx512fp16.mul.ph.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}
  return _mm512_maskz_mul_round_ph(__U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_div_ph(__m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_div_ph
  // CHECK: %{{.*}} = fdiv <32 x half> %{{.*}}, %{{.*}}
  return _mm512_div_ph(__A, __B);
}

__m512h test_mm512_mask_div_ph(__m512h __W, __mmask32 __U, __m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_mask_div_ph
  // CHECK: %{{.*}} = fdiv <32 x half> %{{.*}}, %{{.*}}
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}
  return (__m512h)_mm512_mask_div_ph(__W, __U, __A, __B);
}

__m512h test_mm512_maskz_div_ph(__mmask32 __U, __m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_maskz_div_ph
  // CHECK: %{{.*}} = fdiv <32 x half> %{{.*}}, %{{.*}}
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}
  return _mm512_maskz_div_ph(__U, __A, __B);
}

__m512h test_mm512_div_round_ph(__m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_div_round_ph
  // CHECK: @llvm.x86.avx512fp16.div.ph.512
  return _mm512_div_round_ph(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512h test_mm512_mask_div_round_ph(__m512h __W, __mmask32 __U, __m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_mask_div_round_ph
  // CHECK: @llvm.x86.avx512fp16.div.ph.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}
  return _mm512_mask_div_round_ph(__W, __U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}
__m512h test_mm512_maskz_div_round_ph(__mmask32 __U, __m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_maskz_div_round_ph
  // CHECK: @llvm.x86.avx512fp16.div.ph.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}
  return _mm512_maskz_div_round_ph(__U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_min_ph(__m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_min_ph
  // CHECK: @llvm.x86.avx512fp16.min.ph.512
  return _mm512_min_ph(__A, __B);
}

__m512h test_mm512_mask_min_ph(__m512h __W, __mmask32 __U, __m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_mask_min_ph
  // CHECK: @llvm.x86.avx512fp16.min.ph.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}
  return (__m512h)_mm512_mask_min_ph(__W, __U, __A, __B);
}

__m512h test_mm512_maskz_min_ph(__mmask32 __U, __m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_maskz_min_ph
  // CHECK: @llvm.x86.avx512fp16.min.ph.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}
  return _mm512_maskz_min_ph(__U, __A, __B);
}

__m512h test_mm512_min_round_ph(__m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_min_round_ph
  // CHECK: @llvm.x86.avx512fp16.min.ph.512
  return _mm512_min_round_ph(__A, __B, _MM_FROUND_NO_EXC);
}
__m512h test_mm512_mask_min_round_ph(__m512h __W, __mmask32 __U, __m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_mask_min_round_ph
  // CHECK: @llvm.x86.avx512fp16.min.ph.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}
  return _mm512_mask_min_round_ph(__W, __U, __A, __B, _MM_FROUND_NO_EXC);
}
__m512h test_mm512_maskz_min_round_ph(__mmask32 __U, __m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_maskz_min_round_ph
  // CHECK: @llvm.x86.avx512fp16.min.ph.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}
  return _mm512_maskz_min_round_ph(__U, __A, __B, _MM_FROUND_NO_EXC);
}

__m512h test_mm512_max_ph(__m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_max_ph
  // CHECK: @llvm.x86.avx512fp16.max.ph.512

  return _mm512_max_ph(__A, __B);
}

__m512h test_mm512_mask_max_ph(__m512h __W, __mmask32 __U, __m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_mask_max_ph
  // CHECK: @llvm.x86.avx512fp16.max.ph.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}
  return (__m512h)_mm512_mask_max_ph(__W, __U, __A, __B);
}

__m512h test_mm512_maskz_max_ph(__mmask32 __U, __m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_maskz_max_ph
  // CHECK: @llvm.x86.avx512fp16.max.ph.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}
  return _mm512_maskz_max_ph(__U, __A, __B);
}

__m512h test_mm512_max_round_ph(__m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_max_round_ph
  // CHECK: @llvm.x86.avx512fp16.max.ph.512
  return _mm512_max_round_ph(__A, __B, _MM_FROUND_NO_EXC);
}

__m512h test_mm512_mask_max_round_ph(__m512h __W, __mmask32 __U, __m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_mask_max_round_ph
  // CHECK: @llvm.x86.avx512fp16.max.ph.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}
  return _mm512_mask_max_round_ph(__W, __U, __A, __B, _MM_FROUND_NO_EXC);
}

__m512h test_mm512_maskz_max_round_ph(__mmask32 __U, __m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_maskz_max_round_ph
  // CHECK: @llvm.x86.avx512fp16.max.ph.512
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}
  return _mm512_maskz_max_round_ph(__U, __A, __B, _MM_FROUND_NO_EXC);
}

__m512h test_mm512_abs_ph(__m512h a) {
  // CHECK-LABEL: test_mm512_abs_ph
  // CHECK: and <16 x i32>
  return _mm512_abs_ph(a);
}
TEST_CONSTEXPR(match_m512h(_mm512_abs_ph((__m512h){-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0, 10.0, -11.0, 12.0, -13.0, 14.0, -15.0, -16.0, -17.0, 18.0, -19.0, 20.0, -21.0, 22.0, -23.0, 24.0, -25.0, 26.0, -27.0, 28.0, -29.0, 30.0, -31.0, 32.0}), 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0));

__m512h test_mm512_conj_pch(__m512h __A) {
  // CHECK-LABEL: test_mm512_conj_pch
  // CHECK:  %{{.*}} = bitcast <32 x half> %{{.*}} to <8 x i64>
  // CHECK:  %{{.*}} = bitcast <8 x i64> %{{.*}} to <16 x i32>
  // CHECK:  %{{.*}} = bitcast <8 x i64> %{{.*}} to <16 x i32>
  // CHECK:  %{{.*}} = xor <16 x i32> %{{.*}}, %{{.*}}
  // CHECK:  %{{.*}} = bitcast <16 x i32> %{{.*}} to <8 x i64>
  // CHECK:  %{{.*}} = bitcast <8 x i64> %{{.*}} to <32 x half>
  return _mm512_conj_pch(__A);
}

__m512h test_mm512_mask_conj_pch(__m512h __W, __mmask32 __U, __m512h __A) {
  // CHECK-LABEL: test_mm512_mask_conj_pch
  // CHECK:  %{{.*}} = trunc i32 %{{.*}} to i16
  // CHECK:  %{{.*}} = bitcast <32 x half> %{{.*}} to <8 x i64>
  // CHECK:  %{{.*}} = bitcast <8 x i64> %{{.*}} to <16 x i32>
  // CHECK:  %{{.*}} = bitcast <8 x i64> %{{.*}} to <16 x i32>
  // CHECK:  %{{.*}} = xor <16 x i32> %{{.*}}, %{{.*}}
  // CHECK:  %{{.*}} = bitcast <16 x i32> %{{.*}} to <8 x i64>
  // CHECK:  %{{.*}} = bitcast <8 x i64> %{{.*}} to <32 x half>
  // CHECK:  %{{.*}} = bitcast <32 x half> %{{.*}} to <16 x float>
  // CHECK:  %{{.*}} = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK:  %{{.*}} = select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  // CHECK:  %{{.*}} = bitcast <16 x float> %{{.*}} to <32 x half>
  return _mm512_mask_conj_pch(__W, __U, __A);
}

__m512h test_mm512_maskz_conj_pch(__mmask32 __U, __m512h __A) {
  // CHECK-LABEL: test_mm512_maskz_conj_pch
  // CHECK:  %{{.*}} = trunc i32 %{{.*}} to i16
  // CHECK:  %{{.*}} = bitcast <32 x half> %{{.*}} to <8 x i64>
  // CHECK:  %{{.*}} = bitcast <8 x i64> %{{.*}} to <16 x i32>
  // CHECK:  %{{.*}} = bitcast <8 x i64> %{{.*}} to <16 x i32>
  // CHECK:  %{{.*}} = xor <16 x i32> %{{.*}}, %{{.*}}
  // CHECK:  %{{.*}} = bitcast <16 x i32> %{{.*}} to <8 x i64>
  // CHECK:  %{{.*}} = bitcast <8 x i64> %{{.*}} to <32 x half>
  // CHECK:  %{{.*}} = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK:  %{{.*}} = select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  // CHECK:  %{{.*}} = bitcast <16 x float> %{{.*}} to <32 x half>
  return _mm512_maskz_conj_pch(__U, __A);
}

__m128h test_mm_add_round_sh(__m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_add_round_sh
  // CHECK: @llvm.x86.avx512fp16.mask.add.sh.round
  return _mm_add_round_sh(__A, __B, _MM_FROUND_NO_EXC | _MM_FROUND_TO_ZERO);
}
__m128h test_mm_mask_add_round_sh(__m128h __W, __mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_mask_add_round_sh
  // CHECK: @llvm.x86.avx512fp16.mask.add.sh.round
  return _mm_mask_add_round_sh(__W, __U, __A, __B, _MM_FROUND_NO_EXC | _MM_FROUND_TO_ZERO);
}
__m128h test_mm_maskz_add_round_sh(__mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_maskz_add_round_sh
  // CHECK: @llvm.x86.avx512fp16.mask.add.sh.round
  return _mm_maskz_add_round_sh(__U, __A, __B, _MM_FROUND_NO_EXC | _MM_FROUND_TO_ZERO);
}
__m128h test_mm_mask_add_sh(__m128h __W, __mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_mask_add_sh
  // CHECK: %{{.*}} = extractelement <8 x half> %{{.*}}, i32 0
  // CHECK: %{{.*}} = extractelement <8 x half> %{{.*}}, i32 0
  // CHECK: %{{.*}} = fadd half %{{.*}}, %{{.*}}
  // CHECK: %{{.*}} = insertelement <8 x half> %{{.*}}, half %{{.*}}, i32 0
  // CHECK: %{{.*}} = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: %{{.*}} = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: %{{.*}} = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: %{{.*}} = extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: %{{.*}} = select i1 %{{.*}}, half %{{.*}}, half %{{.*}}
  // CHECK-NEXT: %{{.*}} = insertelement <8 x half> %{{.*}}, half %{{.*}}, i64 0
  return _mm_mask_add_sh(__W, __U, __A, __B);
}
__m128h test_mm_maskz_add_sh(__mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_maskz_add_sh
  // CHECK: %{{.*}} = extractelement <8 x half> %{{.*}}, i32 0
  // CHECK: %{{.*}} = extractelement <8 x half> %{{.*}}, i32 0
  // CHECK: %{{.*}} = fadd half %{{.*}}, %{{.*}}
  // CHECK: %{{.*}} = insertelement <8 x half> %{{.*}}, half %{{.*}}, i32 0
  // CHECK: %{{.*}} = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: %{{.*}} = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: %{{.*}} = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: %{{.*}} = extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: %{{.*}} = select i1 %{{.*}}, half %{{.*}}, half %{{.*}}
  // CHECK-NEXT: %{{.*}} = insertelement <8 x half> %{{.*}}, half %{{.*}}, i64 0
  return _mm_maskz_add_sh(__U, __A, __B);
}

__m128h test_mm_add_sh(__m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_add_sh
  // CHECK: %{{.*}} = extractelement <8 x half> %{{.*}}, i32 0
  // CHECK: %{{.*}} = extractelement <8 x half> %{{.*}}, i32 0
  // CHECK: %{{.*}} = fadd half %{{.*}}, %{{.*}}
  // CHECK: %{{.*}} = insertelement <8 x half> %{{.*}}, half %{{.*}}, i32 0
  return _mm_add_sh(__A, __B);
}

__m128h test_mm_sub_round_sh(__m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_sub_round_sh
  // CHECK: @llvm.x86.avx512fp16.mask.sub.sh.round
  return _mm_sub_round_sh(__A, __B, _MM_FROUND_NO_EXC | _MM_FROUND_TO_ZERO);
}
__m128h test_mm_mask_sub_round_sh(__m128h __W, __mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_mask_sub_round_sh
  // CHECK: @llvm.x86.avx512fp16.mask.sub.sh.round
  return _mm_mask_sub_round_sh(__W, __U, __A, __B, _MM_FROUND_NO_EXC | _MM_FROUND_TO_ZERO);
}
__m128h test_mm_maskz_sub_round_sh(__mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_maskz_sub_round_sh
  // CHECK: @llvm.x86.avx512fp16.mask.sub.sh.round
  return _mm_maskz_sub_round_sh(__U, __A, __B, _MM_FROUND_NO_EXC | _MM_FROUND_TO_ZERO);
}
__m128h test_mm_mask_sub_sh(__m128h __W, __mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_mask_sub_sh
  // CHECK: %{{.*}} = extractelement <8 x half> %{{.*}}, i32 0
  // CHECK: %{{.*}} = extractelement <8 x half> %{{.*}}, i32 0
  // CHECK: %{{.*}} = fsub half %{{.*}}, %{{.*}}
  // CHECK: %{{.*}} = insertelement <8 x half> %{{.*}}, half %{{.*}}, i32 0
  // CHECK: %{{.*}} = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: %{{.*}} = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: %{{.*}} = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: %{{.*}} = extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: %{{.*}} = select i1 %{{.*}}, half %{{.*}}, half %{{.*}}
  // CHECK-NEXT: %{{.*}} = insertelement <8 x half> %{{.*}}, half %{{.*}}, i64 0
  return _mm_mask_sub_sh(__W, __U, __A, __B);
}
__m128h test_mm_maskz_sub_sh(__mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_maskz_sub_sh
  // CHECK: %{{.*}} = extractelement <8 x half> %{{.*}}, i32 0
  // CHECK: %{{.*}} = extractelement <8 x half> %{{.*}}, i32 0
  // CHECK: %{{.*}} = fsub half %{{.*}}, %{{.*}}
  // CHECK: %{{.*}} = insertelement <8 x half> %{{.*}}, half %{{.*}}, i32 0
  // CHECK: %{{.*}} = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: %{{.*}} = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: %{{.*}} = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: %{{.*}} = extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: %{{.*}} = select i1 %{{.*}}, half %{{.*}}, half %{{.*}}
  // CHECK-NEXT: %{{.*}} = insertelement <8 x half> %{{.*}}, half %{{.*}}, i64 0
  return _mm_maskz_sub_sh(__U, __A, __B);
}

__m128h test_mm_sub_sh(__m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_sub_sh
  // CHECK: %{{.*}} = extractelement <8 x half> %{{.*}}, i32 0
  // CHECK: %{{.*}} = extractelement <8 x half> %{{.*}}, i32 0
  // CHECK: %{{.*}} = fsub half %{{.*}}, %{{.*}}
  // CHECK: %{{.*}} = insertelement <8 x half> %{{.*}}, half %{{.*}}, i32 0
  return _mm_sub_sh(__A, __B);
}

__m128h test_mm_mul_round_sh(__m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_mul_round_sh
  // CHECK: @llvm.x86.avx512fp16.mask.mul.sh.round
  return _mm_mul_round_sh(__A, __B, _MM_FROUND_NO_EXC | _MM_FROUND_TO_ZERO);
}
__m128h test_mm_mask_mul_round_sh(__m128h __W, __mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_mask_mul_round_sh
  // CHECK: @llvm.x86.avx512fp16.mask.mul.sh.round
  return _mm_mask_mul_round_sh(__W, __U, __A, __B, _MM_FROUND_NO_EXC | _MM_FROUND_TO_ZERO);
}
__m128h test_mm_maskz_mul_round_sh(__mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_maskz_mul_round_sh
  // CHECK: @llvm.x86.avx512fp16.mask.mul.sh.round
  return _mm_maskz_mul_round_sh(__U, __A, __B, _MM_FROUND_NO_EXC | _MM_FROUND_TO_ZERO);
}
__m128h test_mm_mask_mul_sh(__m128h __W, __mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_mask_mul_sh
  // CHECK: %{{.*}} = extractelement <8 x half> %{{.*}}, i32 0
  // CHECK: %{{.*}} = extractelement <8 x half> %{{.*}}, i32 0
  // CHECK: %{{.*}} = fmul half %{{.*}}, %{{.*}}
  // CHECK: %{{.*}} = insertelement <8 x half> %{{.*}}, half %{{.*}}, i32 0
  // CHECK: %{{.*}} = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: %{{.*}} = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: %{{.*}} = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: %{{.*}} = extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: %{{.*}} = select i1 %{{.*}}, half %{{.*}}, half %{{.*}}
  // CHECK-NEXT: %{{.*}} = insertelement <8 x half> %{{.*}}, half %{{.*}}, i64 0
  return _mm_mask_mul_sh(__W, __U, __A, __B);
}
__m128h test_mm_maskz_mul_sh(__mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_maskz_mul_sh
  // CHECK: %{{.*}} = extractelement <8 x half> %{{.*}}, i32 0
  // CHECK: %{{.*}} = extractelement <8 x half> %{{.*}}, i32 0
  // CHECK: %{{.*}} = fmul half %{{.*}}, %{{.*}}
  // CHECK: %{{.*}} = insertelement <8 x half> %{{.*}}, half %{{.*}}, i32 0
  // CHECK: %{{.*}} = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: %{{.*}} = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: %{{.*}} = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: %{{.*}} = extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: %{{.*}} = select i1 %{{.*}}, half %{{.*}}, half %{{.*}}
  // CHECK-NEXT: %{{.*}} = insertelement <8 x half> %{{.*}}, half %{{.*}}, i64 0
  return _mm_maskz_mul_sh(__U, __A, __B);
}

__m128h test_mm_mul_sh(__m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_mul_sh
  // CHECK: %{{.*}} = extractelement <8 x half> %{{.*}}, i32 0
  // CHECK: %{{.*}} = extractelement <8 x half> %{{.*}}, i32 0
  // CHECK: %{{.*}} = fmul half %{{.*}}, %{{.*}}
  // CHECK: %{{.*}} = insertelement <8 x half> %{{.*}}, half %{{.*}}, i32 0
  return _mm_mul_sh(__A, __B);
}

__m128h test_mm_div_round_sh(__m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_div_round_sh
  // CHECK: @llvm.x86.avx512fp16.mask.div.sh.round
  return _mm_div_round_sh(__A, __B, _MM_FROUND_NO_EXC | _MM_FROUND_TO_ZERO);
}
__m128h test_mm_mask_div_round_sh(__m128h __W, __mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_mask_div_round_sh
  // CHECK: @llvm.x86.avx512fp16.mask.div.sh.round
  return _mm_mask_div_round_sh(__W, __U, __A, __B, _MM_FROUND_NO_EXC | _MM_FROUND_TO_ZERO);
}
__m128h test_mm_maskz_div_round_sh(__mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_maskz_div_round_sh
  // CHECK: @llvm.x86.avx512fp16.mask.div.sh.round
  return _mm_maskz_div_round_sh(__U, __A, __B, _MM_FROUND_NO_EXC | _MM_FROUND_TO_ZERO);
}
__m128h test_mm_mask_div_sh(__m128h __W, __mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_mask_div_sh
  // CHECK: %{{.*}} = extractelement <8 x half> %{{.*}}, i32 0
  // CHECK: %{{.*}} = extractelement <8 x half> %{{.*}}, i32 0
  // CHECK: %{{.*}} = fdiv half %{{.*}}, %{{.*}}
  // CHECK: %{{.*}} = insertelement <8 x half> %{{.*}}, half %{{.*}}, i32 0
  // CHECK: %{{.*}} = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: %{{.*}} = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: %{{.*}} = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: %{{.*}} = extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: %{{.*}} = select i1 %{{.*}}, half %{{.*}}, half %{{.*}}
  // CHECK-NEXT: %{{.*}} = insertelement <8 x half> %{{.*}}, half %{{.*}}, i64 0
  return _mm_mask_div_sh(__W, __U, __A, __B);
}
__m128h test_mm_maskz_div_sh(__mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_maskz_div_sh
  // CHECK: %{{.*}} = extractelement <8 x half> %{{.*}}, i32 0
  // CHECK: %{{.*}} = extractelement <8 x half> %{{.*}}, i32 0
  // CHECK: %{{.*}} = fdiv half %{{.*}}, %{{.*}}
  // CHECK: %{{.*}} = insertelement <8 x half> %{{.*}}, half %{{.*}}, i32 0
  // CHECK: %{{.*}} = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: %{{.*}} = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: %{{.*}} = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: %{{.*}} = extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: %{{.*}} = select i1 %{{.*}}, half %{{.*}}, half %{{.*}}
  // CHECK-NEXT: %{{.*}} = insertelement <8 x half> %{{.*}}, half %{{.*}}, i64 0
  return _mm_maskz_div_sh(__U, __A, __B);
}

__m128h test_mm_div_sh(__m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_div_sh
  // CHECK: %{{.*}} = extractelement <8 x half> %{{.*}}, i32 0
  // CHECK: %{{.*}} = extractelement <8 x half> %{{.*}}, i32 0
  // CHECK: %{{.*}} = fdiv half %{{.*}}, %{{.*}}
  // CHECK: %{{.*}} = insertelement <8 x half> %{{.*}}, half %{{.*}}, i32 0
  return _mm_div_sh(__A, __B);
}

__m128h test_mm_min_round_sh(__m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_min_round_sh
  // CHECK: @llvm.x86.avx512fp16.mask.min.sh.round
  return _mm_min_round_sh(__A, __B, 0x08);
}
__m128h test_mm_mask_min_round_sh(__m128h __W, __mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_mask_min_round_sh
  // CHECK: @llvm.x86.avx512fp16.mask.min.sh.round
  return _mm_mask_min_round_sh(__W, __U, __A, __B, 0x08);
}
__m128h test_mm_maskz_min_round_sh(__mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_maskz_min_round_sh
  // CHECK: @llvm.x86.avx512fp16.mask.min.sh.round
  return _mm_maskz_min_round_sh(__U, __A, __B, 0x08);
}
__m128h test_mm_mask_min_sh(__m128h __W, __mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_mask_min_sh
  // CHECK: @llvm.x86.avx512fp16.mask.min.sh.round
  return _mm_mask_min_sh(__W, __U, __A, __B);
}
__m128h test_mm_maskz_min_sh(__mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_maskz_min_sh
  // CHECK: @llvm.x86.avx512fp16.mask.min.sh.round
  return _mm_maskz_min_sh(__U, __A, __B);
}

__m128h test_mm_min_sh(__m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_min_sh
  // CHECK: @llvm.x86.avx512fp16.mask.min.sh.round
  return _mm_min_sh(__A, __B);
}

__m128h test_mm_max_round_sh(__m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_max_round_sh
  // CHECK: @llvm.x86.avx512fp16.mask.max.sh.round
  return _mm_max_round_sh(__A, __B, 0x08);
}
__m128h test_mm_mask_max_round_sh(__m128h __W, __mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_mask_max_round_sh
  // CHECK: @llvm.x86.avx512fp16.mask.max.sh.round
  return _mm_mask_max_round_sh(__W, __U, __A, __B, 0x08);
}
__m128h test_mm_maskz_max_round_sh(__mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_maskz_max_round_sh
  // CHECK: @llvm.x86.avx512fp16.mask.max.sh.round
  return _mm_maskz_max_round_sh(__U, __A, __B, 0x08);
}
__m128h test_mm_mask_max_sh(__m128h __W, __mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_mask_max_sh
  // CHECK: @llvm.x86.avx512fp16.mask.max.sh.round
  return _mm_mask_max_sh(__W, __U, __A, __B);
}
__m128h test_mm_maskz_max_sh(__mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_maskz_max_sh
  // CHECK: @llvm.x86.avx512fp16.mask.max.sh.round
  return _mm_maskz_max_sh(__U, __A, __B);
}

__m128h test_mm_max_sh(__m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_max_sh
  // CHECK: @llvm.x86.avx512fp16.mask.max.sh.round
  return _mm_max_sh(__A, __B);
}
__mmask32 test_mm512_cmp_round_ph_mask(__m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_cmp_round_ph_mask
  // CHECK: fcmp oeq <32 x half> %{{.*}}, %{{.*}}
  return _mm512_cmp_round_ph_mask(a, b, 0, _MM_FROUND_NO_EXC);
}

__mmask32 test_mm512_mask_cmp_round_ph_mask(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_round_ph_mask
  // CHECK: [[CMP:%.*]] = fcmp oeq <32 x half> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_round_ph_mask(m, a, b, 0, _MM_FROUND_NO_EXC);
}

__mmask32 test_mm512_cmp_ph_mask_eq_oq(__m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_cmp_ph_mask_eq_oq
  // CHECK: fcmp oeq <32 x half> %{{.*}}, %{{.*}}
  return _mm512_cmp_ph_mask(a, b, _CMP_EQ_OQ);
}

__mmask32 test_mm512_cmp_ph_mask_lt_os(__m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_cmp_ph_mask_lt_os
  // CHECK: fcmp olt <32 x half> %{{.*}}, %{{.*}}
  return _mm512_cmp_ph_mask(a, b, _CMP_LT_OS);
}

__mmask32 test_mm512_cmp_ph_mask_le_os(__m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_cmp_ph_mask_le_os
  // CHECK: fcmp ole <32 x half> %{{.*}}, %{{.*}}
  return _mm512_cmp_ph_mask(a, b, _CMP_LE_OS);
}

__mmask32 test_mm512_cmp_ph_mask_unord_q(__m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_cmp_ph_mask_unord_q
  // CHECK: fcmp uno <32 x half> %{{.*}}, %{{.*}}
  return _mm512_cmp_ph_mask(a, b, _CMP_UNORD_Q);
}

__mmask32 test_mm512_cmp_ph_mask_neq_uq(__m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_cmp_ph_mask_neq_uq
  // CHECK: fcmp une <32 x half> %{{.*}}, %{{.*}}
  return _mm512_cmp_ph_mask(a, b, _CMP_NEQ_UQ);
}

__mmask32 test_mm512_cmp_ph_mask_nlt_us(__m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_cmp_ph_mask_nlt_us
  // CHECK: fcmp uge <32 x half> %{{.*}}, %{{.*}}
  return _mm512_cmp_ph_mask(a, b, _CMP_NLT_US);
}

__mmask32 test_mm512_cmp_ph_mask_nle_us(__m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_cmp_ph_mask_nle_us
  // CHECK: fcmp ugt <32 x half> %{{.*}}, %{{.*}}
  return _mm512_cmp_ph_mask(a, b, _CMP_NLE_US);
}

__mmask32 test_mm512_cmp_ph_mask_ord_q(__m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_cmp_ph_mask_ord_q
  // CHECK: fcmp ord <32 x half> %{{.*}}, %{{.*}}
  return _mm512_cmp_ph_mask(a, b, _CMP_ORD_Q);
}

__mmask32 test_mm512_cmp_ph_mask_eq_uq(__m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_cmp_ph_mask_eq_uq
  // CHECK: fcmp ueq <32 x half> %{{.*}}, %{{.*}}
  return _mm512_cmp_ph_mask(a, b, _CMP_EQ_UQ);
}

__mmask32 test_mm512_cmp_ph_mask_nge_us(__m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_cmp_ph_mask_nge_us
  // CHECK: fcmp ult <32 x half> %{{.*}}, %{{.*}}
  return _mm512_cmp_ph_mask(a, b, _CMP_NGE_US);
}

__mmask32 test_mm512_cmp_ph_mask_ngt_us(__m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_cmp_ph_mask_ngt_us
  // CHECK: fcmp ule <32 x half> %{{.*}}, %{{.*}}
  return _mm512_cmp_ph_mask(a, b, _CMP_NGT_US);
}

__mmask32 test_mm512_cmp_ph_mask_false_oq(__m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_cmp_ph_mask_false_oq
  // CHECK: fcmp false <32 x half> %{{.*}}, %{{.*}}
  return _mm512_cmp_ph_mask(a, b, _CMP_FALSE_OQ);
}

__mmask32 test_mm512_cmp_ph_mask_neq_oq(__m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_cmp_ph_mask_neq_oq
  // CHECK: fcmp one <32 x half> %{{.*}}, %{{.*}}
  return _mm512_cmp_ph_mask(a, b, _CMP_NEQ_OQ);
}

__mmask32 test_mm512_cmp_ph_mask_ge_os(__m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_cmp_ph_mask_ge_os
  // CHECK: fcmp oge <32 x half> %{{.*}}, %{{.*}}
  return _mm512_cmp_ph_mask(a, b, _CMP_GE_OS);
}

__mmask32 test_mm512_cmp_ph_mask_gt_os(__m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_cmp_ph_mask_gt_os
  // CHECK: fcmp ogt <32 x half> %{{.*}}, %{{.*}}
  return _mm512_cmp_ph_mask(a, b, _CMP_GT_OS);
}

__mmask32 test_mm512_cmp_ph_mask_true_uq(__m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_cmp_ph_mask_true_uq
  // CHECK: fcmp true <32 x half> %{{.*}}, %{{.*}}
  return _mm512_cmp_ph_mask(a, b, _CMP_TRUE_UQ);
}

__mmask32 test_mm512_cmp_ph_mask_eq_os(__m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_cmp_ph_mask_eq_os
  // CHECK: fcmp oeq <32 x half> %{{.*}}, %{{.*}}
  return _mm512_cmp_ph_mask(a, b, _CMP_EQ_OS);
}

__mmask32 test_mm512_cmp_ph_mask_lt_oq(__m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_cmp_ph_mask_lt_oq
  // CHECK: fcmp olt <32 x half> %{{.*}}, %{{.*}}
  return _mm512_cmp_ph_mask(a, b, _CMP_LT_OQ);
}

__mmask32 test_mm512_cmp_ph_mask_le_oq(__m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_cmp_ph_mask_le_oq
  // CHECK: fcmp ole <32 x half> %{{.*}}, %{{.*}}
  return _mm512_cmp_ph_mask(a, b, _CMP_LE_OQ);
}

__mmask32 test_mm512_cmp_ph_mask_unord_s(__m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_cmp_ph_mask_unord_s
  // CHECK: fcmp uno <32 x half> %{{.*}}, %{{.*}}
  return _mm512_cmp_ph_mask(a, b, _CMP_UNORD_S);
}

__mmask32 test_mm512_cmp_ph_mask_neq_us(__m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_cmp_ph_mask_neq_us
  // CHECK: fcmp une <32 x half> %{{.*}}, %{{.*}}
  return _mm512_cmp_ph_mask(a, b, _CMP_NEQ_US);
}

__mmask32 test_mm512_cmp_ph_mask_nlt_uq(__m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_cmp_ph_mask_nlt_uq
  // CHECK: fcmp uge <32 x half> %{{.*}}, %{{.*}}
  return _mm512_cmp_ph_mask(a, b, _CMP_NLT_UQ);
}

__mmask32 test_mm512_cmp_ph_mask_nle_uq(__m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_cmp_ph_mask_nle_uq
  // CHECK: fcmp ugt <32 x half> %{{.*}}, %{{.*}}
  return _mm512_cmp_ph_mask(a, b, _CMP_NLE_UQ);
}

__mmask32 test_mm512_cmp_ph_mask_ord_s(__m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_cmp_ph_mask_ord_s
  // CHECK: fcmp ord <32 x half> %{{.*}}, %{{.*}}
  return _mm512_cmp_ph_mask(a, b, _CMP_ORD_S);
}

__mmask32 test_mm512_cmp_ph_mask_eq_us(__m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_cmp_ph_mask_eq_us
  // CHECK: fcmp ueq <32 x half> %{{.*}}, %{{.*}}
  return _mm512_cmp_ph_mask(a, b, _CMP_EQ_US);
}

__mmask32 test_mm512_cmp_ph_mask_nge_uq(__m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_cmp_ph_mask_nge_uq
  // CHECK: fcmp ult <32 x half> %{{.*}}, %{{.*}}
  return _mm512_cmp_ph_mask(a, b, _CMP_NGE_UQ);
}

__mmask32 test_mm512_cmp_ph_mask_ngt_uq(__m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_cmp_ph_mask_ngt_uq
  // CHECK: fcmp ule <32 x half> %{{.*}}, %{{.*}}
  return _mm512_cmp_ph_mask(a, b, _CMP_NGT_UQ);
}

__mmask32 test_mm512_cmp_ph_mask_false_os(__m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_cmp_ph_mask_false_os
  // CHECK: fcmp false <32 x half> %{{.*}}, %{{.*}}
  return _mm512_cmp_ph_mask(a, b, _CMP_FALSE_OS);
}

__mmask32 test_mm512_cmp_ph_mask_neq_os(__m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_cmp_ph_mask_neq_os
  // CHECK: fcmp one <32 x half> %{{.*}}, %{{.*}}
  return _mm512_cmp_ph_mask(a, b, _CMP_NEQ_OS);
}

__mmask32 test_mm512_cmp_ph_mask_ge_oq(__m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_cmp_ph_mask_ge_oq
  // CHECK: fcmp oge <32 x half> %{{.*}}, %{{.*}}
  return _mm512_cmp_ph_mask(a, b, _CMP_GE_OQ);
}

__mmask32 test_mm512_cmp_ph_mask_gt_oq(__m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_cmp_ph_mask_gt_oq
  // CHECK: fcmp ogt <32 x half> %{{.*}}, %{{.*}}
  return _mm512_cmp_ph_mask(a, b, _CMP_GT_OQ);
}

__mmask32 test_mm512_cmp_ph_mask_true_us(__m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_cmp_ph_mask_true_us
  // CHECK: fcmp true <32 x half> %{{.*}}, %{{.*}}
  return _mm512_cmp_ph_mask(a, b, _CMP_TRUE_US);
}

__mmask32 test_mm512_mask_cmp_ph_mask_eq_oq(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_eq_oq
  // CHECK: [[CMP:%.*]] = fcmp oeq <32 x half> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_EQ_OQ);
}

__mmask32 test_mm512_mask_cmp_ph_mask_lt_os(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_lt_os
  // CHECK: [[CMP:%.*]] = fcmp olt <32 x half> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_LT_OS);
}

__mmask32 test_mm512_mask_cmp_ph_mask_le_os(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_le_os
  // CHECK: [[CMP:%.*]] = fcmp ole <32 x half> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_LE_OS);
}

__mmask32 test_mm512_mask_cmp_ph_mask_unord_q(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_unord_q
  // CHECK: [[CMP:%.*]] = fcmp uno <32 x half> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_UNORD_Q);
}

__mmask32 test_mm512_mask_cmp_ph_mask_neq_uq(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_neq_uq
  // CHECK: [[CMP:%.*]] = fcmp une <32 x half> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_NEQ_UQ);
}

__mmask32 test_mm512_mask_cmp_ph_mask_nlt_us(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_nlt_us
  // CHECK: [[CMP:%.*]] = fcmp uge <32 x half> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_NLT_US);
}

__mmask32 test_mm512_mask_cmp_ph_mask_nle_us(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_nle_us
  // CHECK: [[CMP:%.*]] = fcmp ugt <32 x half> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_NLE_US);
}

__mmask32 test_mm512_mask_cmp_ph_mask_ord_q(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_ord_q
  // CHECK: [[CMP:%.*]] = fcmp ord <32 x half> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_ORD_Q);
}

__mmask32 test_mm512_mask_cmp_ph_mask_eq_uq(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_eq_uq
  // CHECK: [[CMP:%.*]] = fcmp ueq <32 x half> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_EQ_UQ);
}

__mmask32 test_mm512_mask_cmp_ph_mask_nge_us(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_nge_us
  // CHECK: [[CMP:%.*]] = fcmp ult <32 x half> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_NGE_US);
}

__mmask32 test_mm512_mask_cmp_ph_mask_ngt_us(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_ngt_us
  // CHECK: [[CMP:%.*]] = fcmp ule <32 x half> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_NGT_US);
}

__mmask32 test_mm512_mask_cmp_ph_mask_false_oq(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_false_oq
  // CHECK: [[CMP:%.*]] = fcmp false <32 x half> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_FALSE_OQ);
}

__mmask32 test_mm512_mask_cmp_ph_mask_neq_oq(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_neq_oq
  // CHECK: [[CMP:%.*]] = fcmp one <32 x half> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_NEQ_OQ);
}

__mmask32 test_mm512_mask_cmp_ph_mask_ge_os(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_ge_os
  // CHECK: [[CMP:%.*]] = fcmp oge <32 x half> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_GE_OS);
}

__mmask32 test_mm512_mask_cmp_ph_mask_gt_os(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_gt_os
  // CHECK: [[CMP:%.*]] = fcmp ogt <32 x half> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_GT_OS);
}

__mmask32 test_mm512_mask_cmp_ph_mask_true_uq(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_true_uq
  // CHECK: [[CMP:%.*]] = fcmp true <32 x half> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_TRUE_UQ);
}

__mmask32 test_mm512_mask_cmp_ph_mask_eq_os(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_eq_os
  // CHECK: [[CMP:%.*]] = fcmp oeq <32 x half> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_EQ_OS);
}

__mmask32 test_mm512_mask_cmp_ph_mask_lt_oq(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_lt_oq
  // CHECK: [[CMP:%.*]] = fcmp olt <32 x half> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_LT_OQ);
}

__mmask32 test_mm512_mask_cmp_ph_mask_le_oq(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_le_oq
  // CHECK: [[CMP:%.*]] = fcmp ole <32 x half> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_LE_OQ);
}

__mmask32 test_mm512_mask_cmp_ph_mask_unord_s(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_unord_s
  // CHECK: [[CMP:%.*]] = fcmp uno <32 x half> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_UNORD_S);
}

__mmask32 test_mm512_mask_cmp_ph_mask_neq_us(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_neq_us
  // CHECK: [[CMP:%.*]] = fcmp une <32 x half> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_NEQ_US);
}

__mmask32 test_mm512_mask_cmp_ph_mask_nlt_uq(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_nlt_uq
  // CHECK: [[CMP:%.*]] = fcmp uge <32 x half> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_NLT_UQ);
}

__mmask32 test_mm512_mask_cmp_ph_mask_nle_uq(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_nle_uq
  // CHECK: [[CMP:%.*]] = fcmp ugt <32 x half> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_NLE_UQ);
}

__mmask32 test_mm512_mask_cmp_ph_mask_ord_s(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_ord_s
  // CHECK: [[CMP:%.*]] = fcmp ord <32 x half> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_ORD_S);
}

__mmask32 test_mm512_mask_cmp_ph_mask_eq_us(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_eq_us
  // CHECK: [[CMP:%.*]] = fcmp ueq <32 x half> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_EQ_US);
}

__mmask32 test_mm512_mask_cmp_ph_mask_nge_uq(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_nge_uq
  // CHECK: [[CMP:%.*]] = fcmp ult <32 x half> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_NGE_UQ);
}

__mmask32 test_mm512_mask_cmp_ph_mask_ngt_uq(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_ngt_uq
  // CHECK: [[CMP:%.*]] = fcmp ule <32 x half> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_NGT_UQ);
}

__mmask32 test_mm512_mask_cmp_ph_mask_false_os(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_false_os
  // CHECK: [[CMP:%.*]] = fcmp false <32 x half> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_FALSE_OS);
}

__mmask32 test_mm512_mask_cmp_ph_mask_neq_os(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_neq_os
  // CHECK: [[CMP:%.*]] = fcmp one <32 x half> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_NEQ_OS);
}

__mmask32 test_mm512_mask_cmp_ph_mask_ge_oq(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_ge_oq
  // CHECK: [[CMP:%.*]] = fcmp oge <32 x half> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_GE_OQ);
}

__mmask32 test_mm512_mask_cmp_ph_mask_gt_oq(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_gt_oq
  // CHECK: [[CMP:%.*]] = fcmp ogt <32 x half> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_GT_OQ);
}

__mmask32 test_mm512_mask_cmp_ph_mask_true_us(__mmask32 m, __m512h a, __m512h b) {
  // CHECK-LABEL: test_mm512_mask_cmp_ph_mask_true_us
  // CHECK: [[CMP:%.*]] = fcmp true <32 x half> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> [[CMP]], {{.*}}
  return _mm512_mask_cmp_ph_mask(m, a, b, _CMP_TRUE_US);
}

__mmask8 test_mm_cmp_round_sh_mask(__m128h __X, __m128h __Y) {
  // CHECK-LABEL: test_mm_cmp_round_sh_mask
  // CHECK: @llvm.x86.avx512fp16.mask.cmp.sh
  return _mm_cmp_round_sh_mask(__X, __Y, _CMP_NLT_US, _MM_FROUND_NO_EXC);
}

__mmask8 test_mm_mask_cmp_round_sh_mask(__mmask8 __M, __m128h __X, __m128h __Y) {
  // CHECK-LABEL: test_mm_mask_cmp_round_sh_mask
  // CHECK: @llvm.x86.avx512fp16.mask.cmp.sh
  return _mm_mask_cmp_round_sh_mask(__M, __X, __Y, _CMP_NLT_US, _MM_FROUND_NO_EXC);
}

__mmask8 test_mm_cmp_sh_mask(__m128h __X, __m128h __Y) {
  // CHECK-LABEL: test_mm_cmp_sh_mask
  // CHECK: @llvm.x86.avx512fp16.mask.cmp.sh
  return _mm_cmp_sh_mask(__X, __Y, _CMP_NLT_US);
}

__mmask8 test_mm_mask_cmp_sh_mask(__mmask8 __M, __m128h __X, __m128h __Y) {
  // CHECK-LABEL: test_mm_mask_cmp_sh_mask
  // CHECK: @llvm.x86.avx512fp16.mask.cmp.sh
  return _mm_mask_cmp_sh_mask(__M, __X, __Y, _CMP_NLT_US);
}

// VMOVSH

__m128h test_mm_load_sh(void const *A) {
  // CHECK-LABEL: test_mm_load_sh
  // CHECK: load half, ptr %{{.*}}, align 1{{$}}
  return _mm_load_sh(A);
}

__m128h test_mm_mask_load_sh(__m128h __A, __mmask8 __U, const void *__W) {
  // CHECK-LABEL: test_mm_mask_load_sh
  // CHECK: @llvm.masked.load.v8f16.p0(ptr %{{.*}}, i32 1, <8 x i1> %{{.*}}, <8 x half> %{{.*}})
  return _mm_mask_load_sh(__A, __U, __W);
}

__m128h test_mm_maskz_load_sh(__mmask8 __U, const void *__W) {
  // CHECK-LABEL: test_mm_maskz_load_sh
  // CHECK: @llvm.masked.load.v8f16.p0(ptr %{{.*}}, i32 1, <8 x i1> %{{.*}}, <8 x half> %{{.*}})
  return _mm_maskz_load_sh(__U, __W);
}

__m512h test_mm512_load_ph(void *p) {
  // CHECK-LABEL: test_mm512_load_ph
  // CHECK: load <32 x half>, ptr %{{.*}}, align 64{{$}}
  return _mm512_load_ph(p);
}

__m256h test_mm256_load_ph(void *p) {
  // CHECK-LABEL: test_mm256_load_ph
  // CHECK: load <16 x half>, ptr %{{.*}}, align 32{{$}}
  return _mm256_load_ph(p);
}

__m128h test_mm_load_ph(void *p) {
  // CHECK-LABEL: test_mm_load_ph
  // CHECK: load <8 x half>, ptr %{{.*}}, align 16{{$}}
  return _mm_load_ph(p);
}

__m512h test_mm512_loadu_ph(void *p) {
  // CHECK-LABEL: test_mm512_loadu_ph
  // CHECK: load <32 x half>, ptr {{.*}}, align 1{{$}}
  return _mm512_loadu_ph(p);
}

__m256h test_mm256_loadu_ph(void *p) {
  // CHECK-LABEL: test_mm256_loadu_ph
  // CHECK: load <16 x half>, ptr {{.*}}, align 1{{$}}
  return _mm256_loadu_ph(p);
}

__m128h test_mm_loadu_ph(void *p) {
  // CHECK-LABEL: test_mm_loadu_ph
  // CHECK: load <8 x half>, ptr {{.*}}, align 1{{$}}
  return _mm_loadu_ph(p);
}

void test_mm_store_sh(void *A, __m128h B) {
  // CHECK-LABEL: test_mm_store_sh
  // CHECK: extractelement <8 x half> %{{.*}}, i32 0
  // CHECK: store half %{{.*}}, ptr %{{.*}}, align 1{{$}}
  _mm_store_sh(A, B);
}

void test_mm_mask_store_sh(void *__P, __mmask8 __U, __m128h __A) {
  // CHECK-LABEL: test_mm_mask_store_sh
  // CHECK: call void @llvm.masked.store.v8f16.p0(<8 x half> %{{.*}}, ptr %{{.*}}, i32 1, <8 x i1> %{{.*}})
  _mm_mask_store_sh(__P, __U, __A);
}

void test_mm512_store_ph(void *p, __m512h a) {
  // CHECK-LABEL: test_mm512_store_ph
  // CHECK: store <32 x half> %{{.*}}, ptr %{{.*}}, align 64
  _mm512_store_ph(p, a);
}

void test_mm256_store_ph(void *p, __m256h a) {
  // CHECK-LABEL: test_mm256_store_ph
  // CHECK: store <16 x half> %{{.*}}, ptr %{{.*}}, align 32
  _mm256_store_ph(p, a);
}

void test_mm_store_ph(void *p, __m128h a) {
  // CHECK-LABEL: test_mm_store_ph
  // CHECK: store <8 x half> %{{.*}}, ptr %{{.*}}, align 16
  _mm_store_ph(p, a);
}

void test_mm512_storeu_ph(void *p, __m512h a) {
  // CHECK-LABEL: test_mm512_storeu_ph
  // CHECK: store <32 x half> %{{.*}}, ptr %{{.*}}, align 1{{$}}
  // CHECK-NEXT: ret void
  _mm512_storeu_ph(p, a);
}

void test_mm256_storeu_ph(void *p, __m256h a) {
  // CHECK-LABEL: test_mm256_storeu_ph
  // CHECK: store <16 x half> %{{.*}}, ptr %{{.*}}, align 1{{$}}
  // CHECK-NEXT: ret void
  _mm256_storeu_ph(p, a);
}

void test_mm_storeu_ph(void *p, __m128h a) {
  // CHECK-LABEL: test_mm_storeu_ph
  // CHECK: store <8 x half> %{{.*}}, ptr %{{.*}}, align 1{{$}}
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
  // CHECK-LABEL: test_mm_mask_move_sh
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
  // CHECK-LABEL: test_mm_maskz_move_sh
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
  // CHECK: insertelement <8 x i16> poison, i16 %{{.*}}, i32 0
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 0, i32 1
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 0, i32 2
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 0, i32 3
  return _mm_cvtsi16_si128(A);
}

__m512h test_mm512_rcp_ph(__m512h __A) {
  // CHECK-LABEL: test_mm512_rcp_ph
  // CHECK: @llvm.x86.avx512fp16.mask.rcp.ph.512
  return _mm512_rcp_ph(__A);
}

__m512h test_mm512_mask_rcp_ph(__m512h __W, __mmask32 __U, __m512h __A) {
  // CHECK-LABEL: test_mm512_mask_rcp_ph
  // CHECK: @llvm.x86.avx512fp16.mask.rcp.ph.512
  return (__m512h)_mm512_mask_rcp_ph(__W, __U, __A);
}

__m512h test_mm512_maskz_rcp_ph(__mmask32 __U, __m512h __A) {
  // CHECK-LABEL: test_mm512_maskz_rcp_ph
  // CHECK: @llvm.x86.avx512fp16.mask.rcp.ph.512
  return _mm512_maskz_rcp_ph(__U, __A);
}

__m512h test_mm512_rsqrt_ph(__m512h __A) {
  // CHECK-LABEL: test_mm512_rsqrt_ph
  // CHECK: @llvm.x86.avx512fp16.mask.rsqrt.ph.512
  return _mm512_rsqrt_ph(__A);
}

__m512h test_mm512_mask_rsqrt_ph(__m512h __W, __mmask32 __U, __m512h __A) {
  // CHECK-LABEL: test_mm512_mask_rsqrt_ph
  // CHECK: @llvm.x86.avx512fp16.mask.rsqrt.ph.512
  return (__m512h)_mm512_mask_rsqrt_ph(__W, __U, __A);
}

__m512h test_mm512_maskz_rsqrt_ph(__mmask32 __U, __m512h __A) {
  // CHECK-LABEL: test_mm512_maskz_rsqrt_ph
  // CHECK: @llvm.x86.avx512fp16.mask.rsqrt.ph.512
  return _mm512_maskz_rsqrt_ph(__U, __A);
}

__m512h test_mm512_getmant_round_ph(__m512h __A) {
  // CHECK-LABEL: test_mm512_getmant_round_ph
  // CHECK: @llvm.x86.avx512fp16.mask.getmant.ph.512
  return _mm512_getmant_round_ph(__A, _MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan, _MM_FROUND_NO_EXC);
}

__m512h test_mm512_mask_getmant_round_ph(__m512h __W, __mmask32 __U, __m512h __A) {
  // CHECK-LABEL: test_mm512_mask_getmant_round_ph
  // CHECK: @llvm.x86.avx512fp16.mask.getmant.ph.512
  return _mm512_mask_getmant_round_ph(__W, __U, __A, _MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan, _MM_FROUND_NO_EXC);
}

__m512h test_mm512_maskz_getmant_round_ph(__mmask32 __U, __m512h __A) {
  // CHECK-LABEL: test_mm512_maskz_getmant_round_ph
  // CHECK: @llvm.x86.avx512fp16.mask.getmant.ph.512
  return _mm512_maskz_getmant_round_ph(__U, __A, _MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan, _MM_FROUND_NO_EXC);
}

__m512h test_mm512_getmant_ph(__m512h __A) {
  // CHECK-LABEL: test_mm512_getmant_ph
  // CHECK: @llvm.x86.avx512fp16.mask.getmant.ph.512
  return _mm512_getmant_ph(__A, _MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan);
}

__m512h test_mm512_mask_getmant_ph(__m512h __W, __mmask32 __U, __m512h __A) {
  // CHECK-LABEL: test_mm512_mask_getmant_ph
  // CHECK: @llvm.x86.avx512fp16.mask.getmant.ph.512
  return _mm512_mask_getmant_ph(__W, __U, __A, _MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan);
}

__m512h test_mm512_maskz_getmant_ph(__mmask32 __U, __m512h __A) {
  // CHECK-LABEL: test_mm512_maskz_getmant_ph
  // CHECK: @llvm.x86.avx512fp16.mask.getmant.ph.512
  return _mm512_maskz_getmant_ph(__U, __A, _MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan);
}

__m512h test_mm512_scalef_round_ph(__m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_scalef_round_ph
  // CHECK: @llvm.x86.avx512fp16.mask.scalef.ph.512
  return _mm512_scalef_round_ph(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_mask_scalef_round_ph(__m512h __W, __mmask32 __U, __m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_mask_scalef_round_ph
  // CHECK: @llvm.x86.avx512fp16.mask.scalef.ph.512
  return _mm512_mask_scalef_round_ph(__W, __U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_maskz_scalef_round_ph(__mmask32 __U, __m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_maskz_scalef_round_ph
  // CHECK: @llvm.x86.avx512fp16.mask.scalef.ph.512
  return _mm512_maskz_scalef_round_ph(__U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_scalef_ph(__m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_scalef_ph
  // CHECK: @llvm.x86.avx512fp16.mask.scalef.ph.512
  return _mm512_scalef_ph(__A, __B);
}

__m512h test_mm512_mask_scalef_ph(__m512h __W, __mmask32 __U, __m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_mask_scalef_ph
  // CHECK: @llvm.x86.avx512fp16.mask.scalef.ph.512
  return _mm512_mask_scalef_ph(__W, __U, __A, __B);
}

__m512h test_mm512_maskz_scalef_ph(__mmask32 __U, __m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_maskz_scalef_ph
  // CHECK: @llvm.x86.avx512fp16.mask.scalef.ph.512
  return _mm512_maskz_scalef_ph(__U, __A, __B);
}

__m512h test_mm512_mask_roundscale_ph(__m512h __W, __mmask16 __U, __m512h __A) {
  // CHECK-LABEL: test_mm512_mask_roundscale_ph
  // CHECK: @llvm.x86.avx512fp16.mask.rndscale.ph.512
  return _mm512_mask_roundscale_ph(__W, __U, __A, 1);
}

__m512h test_mm512_maskz_roundscale_ph(__mmask16 __U, __m512h __A) {
  // CHECK-LABEL: test_mm512_maskz_roundscale_ph
  // CHECK: @llvm.x86.avx512fp16.mask.rndscale.ph.512
  return _mm512_maskz_roundscale_ph(__U, __A, 1);
}

__m512h test_mm512_mask_roundscale_round_ph(__m512h __A, __mmask16 __U, __m512h __C) {
  // CHECK-LABEL: test_mm512_mask_roundscale_round_ph
  // CHECK: @llvm.x86.avx512fp16.mask.rndscale.ph.512
  return _mm512_mask_roundscale_round_ph(__A, __U, __C, 3, _MM_FROUND_NO_EXC);
}

__m512h test_mm512_maskz_roundscale_round_ph(__m512h __A, __mmask16 __U) {
  // CHECK-LABEL: test_mm512_maskz_roundscale_round_ph
  // CHECK: @llvm.x86.avx512fp16.mask.rndscale.ph.512
  return _mm512_maskz_roundscale_round_ph(__U, __A, 3, _MM_FROUND_NO_EXC);
}

__m512h test_mm512_roundscale_round_ph(__m512h __A) {
  // CHECK-LABEL: test_mm512_roundscale_round_ph
  // CHECK: @llvm.x86.avx512fp16.mask.rndscale.ph.512
  return _mm512_roundscale_round_ph(__A, 3, _MM_FROUND_NO_EXC);
}

__m512h test_mm512_roundscale_ph(__m512h __A) {
  // CHECK-LABEL: test_mm512_roundscale_ph
  // CHECK: @llvm.x86.avx512fp16.mask.rndscale.ph.512
  return _mm512_roundscale_ph(__A, 3);
}

__m512h test_mm512_getexp_round_ph(__m512h __A) {
  // CHECK-LABEL: test_mm512_getexp_round_ph
  // CHECK: @llvm.x86.avx512fp16.mask.getexp.ph.512
  return _mm512_getexp_round_ph(__A, _MM_FROUND_NO_EXC);
}

__m512h test_mm512_mask_getexp_round_ph(__m512h __W, __mmask32 __U, __m512h __A) {
  // CHECK-LABEL: test_mm512_mask_getexp_round_ph
  // CHECK: @llvm.x86.avx512fp16.mask.getexp.ph.512
  return _mm512_mask_getexp_round_ph(__W, __U, __A, _MM_FROUND_NO_EXC);
}

__m512h test_mm512_maskz_getexp_round_ph(__mmask32 __U, __m512h __A) {
  // CHECK-LABEL: test_mm512_maskz_getexp_round_ph
  // CHECK: @llvm.x86.avx512fp16.mask.getexp.ph.512
  return _mm512_maskz_getexp_round_ph(__U, __A, _MM_FROUND_NO_EXC);
}

__m512h test_mm512_getexp_ph(__m512h __A) {
  // CHECK-LABEL: test_mm512_getexp_ph
  // CHECK: @llvm.x86.avx512fp16.mask.getexp.ph.512
  return _mm512_getexp_ph(__A);
}

__m512h test_mm512_mask_getexp_ph(__m512h __W, __mmask32 __U, __m512h __A) {
  // CHECK-LABEL: test_mm512_mask_getexp_ph
  // CHECK: @llvm.x86.avx512fp16.mask.getexp.ph.512
  return _mm512_mask_getexp_ph(__W, __U, __A);
}

__m512h test_mm512_maskz_getexp_ph(__mmask32 __U, __m512h __A) {
  // CHECK-LABEL: test_mm512_maskz_getexp_ph
  // CHECK: @llvm.x86.avx512fp16.mask.getexp.ph.512
  return _mm512_maskz_getexp_ph(__U, __A);
}

__m512h test_mm512_mask_reduce_ph(__m512h __W, __mmask16 __U, __m512h __A) {
  // CHECK-LABEL: test_mm512_mask_reduce_ph
  // CHECK: @llvm.x86.avx512fp16.mask.reduce.ph.512
  return _mm512_mask_reduce_ph(__W, __U, __A, 1);
}

__m512h test_mm512_maskz_reduce_ph(__mmask16 __U, __m512h __A) {
  // CHECK-LABEL: test_mm512_maskz_reduce_ph
  // CHECK: @llvm.x86.avx512fp16.mask.reduce.ph.512
  return _mm512_maskz_reduce_ph(__U, __A, 1);
}

__m512h test_mm512_mask_reduce_round_ph(__m512h __A, __mmask16 __U, __m512h __C) {
  // CHECK-LABEL: test_mm512_mask_reduce_round_ph
  // CHECK: @llvm.x86.avx512fp16.mask.reduce.ph.512
  return _mm512_mask_reduce_round_ph(__A, __U, __C, 3, _MM_FROUND_NO_EXC);
}

__m512h test_mm512_maskz_reduce_round_ph(__m512h __A, __mmask16 __U) {
  // CHECK-LABEL: test_mm512_maskz_reduce_round_ph
  // CHECK: @llvm.x86.avx512fp16.mask.reduce.ph.512
  return _mm512_maskz_reduce_round_ph(__U, __A, 3, _MM_FROUND_NO_EXC);
}

__m512h test_mm512_reduce_round_ph(__m512h __A) {
  // CHECK-LABEL: test_mm512_reduce_round_ph
  // CHECK: @llvm.x86.avx512fp16.mask.reduce.ph.512
  return _mm512_reduce_round_ph(__A, 3, _MM_FROUND_NO_EXC);
}

__m512h test_mm512_reduce_ph(__m512h __A) {
  // CHECK-LABEL: test_mm512_reduce_ph
  // CHECK: @llvm.x86.avx512fp16.mask.reduce.ph.512
  return _mm512_reduce_ph(__A, 3);
}
__m128h test_mm_rcp_sh(__m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_rcp_sh
  // CHECK: @llvm.x86.avx512fp16.mask.rcp.sh
  return _mm_rcp_sh(__A, __B);
}

__m128h test_mm_mask_rcp_sh(__m128h __W, __mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_mask_rcp_sh
  // CHECK: @llvm.x86.avx512fp16.mask.rcp.sh
  return _mm_mask_rcp_sh(__W, __U, __A, __B);
}

__m128h test_mm_maskz_rcp_sh(__mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_maskz_rcp_sh
  // CHECK: @llvm.x86.avx512fp16.mask.rcp.sh
  return _mm_maskz_rcp_sh(__U, __A, __B);
}

__m128h test_mm_rsqrt_sh(__m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_rsqrt_sh
  // CHECK: @llvm.x86.avx512fp16.mask.rsqrt.sh
  return _mm_rsqrt_sh(__A, __B);
}

__m128h test_mm_mask_rsqrt_sh(__m128h __W, __mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_mask_rsqrt_sh
  // CHECK: @llvm.x86.avx512fp16.mask.rsqrt.sh
  return _mm_mask_rsqrt_sh(__W, __U, __A, __B);
}

__m128h test_mm_maskz_rsqrt_sh(__mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_maskz_rsqrt_sh
  // CHECK: @llvm.x86.avx512fp16.mask.rsqrt.sh
  return _mm_maskz_rsqrt_sh(__U, __A, __B);
}

__m128h test_mm_getmant_round_sh(__m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_getmant_round_sh
  // CHECK: @llvm.x86.avx512fp16.mask.getmant.sh
  return _mm_getmant_round_sh(__A, __B, _MM_MANT_NORM_1_2, _MM_MANT_SIGN_src, 8);
}

__m128h test_mm_getmant_sh(__m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_getmant_sh
  // CHECK: @llvm.x86.avx512fp16.mask.getmant.sh
  return _mm_getmant_sh(__A, __B, _MM_MANT_NORM_1_2, _MM_MANT_SIGN_src);
}

__m128h test_mm_mask_getmant_sh(__m128h __W, __mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_mask_getmant_sh
  // CHECK: @llvm.x86.avx512fp16.mask.getmant.sh
  return _mm_mask_getmant_sh(__W, __U, __A, __B, 1, 2);
}

__m128h test_mm_mask_getmant_round_sh(__m128h __W, __mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_mask_getmant_round_sh
  // CHECK: @llvm.x86.avx512fp16.mask.getmant.sh
  return _mm_mask_getmant_round_sh(__W, __U, __A, __B, 1, 2, _MM_FROUND_NO_EXC);
}

__m128h test_mm_maskz_getmant_sh(__mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_maskz_getmant_sh
  // CHECK: @llvm.x86.avx512fp16.mask.getmant.sh
  return _mm_maskz_getmant_sh(__U, __A, __B, 1, 2);
}

__m128h test_mm_maskz_getmant_round_sh(__mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_maskz_getmant_round_sh
  // CHECK: @llvm.x86.avx512fp16.mask.getmant.sh
  return _mm_maskz_getmant_round_sh(__U, __A, __B, 1, 2, _MM_FROUND_NO_EXC);
}

__m128h test_mm_getexp_round_sh(__m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_getexp_round_sh
  // CHECK: @llvm.x86.avx512fp16.mask.getexp.sh
  return _mm_getexp_round_sh(__A, __B, 8);
}

__m128h test_mm_getexp_sh(__m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_getexp_sh
  // CHECK: @llvm.x86.avx512fp16.mask.getexp.sh
  return _mm_getexp_sh(__A, __B);
}

__m128h test_mm_mask_getexp_sh(__m128h __W, __mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_mask_getexp_sh
  // CHECK: @llvm.x86.avx512fp16.mask.getexp.sh
  return _mm_mask_getexp_sh(__W, __U, __A, __B);
}

__m128h test_mm_mask_getexp_round_sh(__m128h __W, __mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_mask_getexp_round_sh
  // CHECK: @llvm.x86.avx512fp16.mask.getexp.sh
  return _mm_mask_getexp_round_sh(__W, __U, __A, __B, _MM_FROUND_NO_EXC);
}

__m128h test_mm_maskz_getexp_sh(__mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_maskz_getexp_sh
  // CHECK: @llvm.x86.avx512fp16.mask.getexp.sh
  return _mm_maskz_getexp_sh(__U, __A, __B);
}

__m128h test_mm_maskz_getexp_round_sh(__mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_maskz_getexp_round_sh
  // CHECK: @llvm.x86.avx512fp16.mask.getexp.sh
  return _mm_maskz_getexp_round_sh(__U, __A, __B, _MM_FROUND_NO_EXC);
}

__m128h test_mm_scalef_round_sh(__m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_scalef_round_sh
  // CHECK: @llvm.x86.avx512fp16.mask.scalef.sh(<8 x half> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}}, i8 -1, i32 11)
  return _mm_scalef_round_sh(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm_scalef_sh(__m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_scalef_sh
  // CHECK: @llvm.x86.avx512fp16.mask.scalef.sh
  return _mm_scalef_sh(__A, __B);
}

__m128h test_mm_mask_scalef_sh(__m128h __W, __mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_mask_scalef_sh
  // CHECK: @llvm.x86.avx512fp16.mask.scalef.sh
  return _mm_mask_scalef_sh(__W, __U, __A, __B);
}

__m128h test_mm_mask_scalef_round_sh(__m128h __W, __mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_mask_scalef_round_sh
  // CHECK: @llvm.x86.avx512fp16.mask.scalef.sh(<8 x half> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}}, i8 %{{.*}}, i32 11)
  return _mm_mask_scalef_round_sh(__W, __U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm_maskz_scalef_sh(__mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_maskz_scalef_sh
  // CHECK: @llvm.x86.avx512fp16.mask.scalef.sh
  return _mm_maskz_scalef_sh(__U, __A, __B);
}

__m128h test_mm_maskz_scalef_round_sh(__mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_maskz_scalef_round_sh
  // CHECK: @llvm.x86.avx512fp16.mask.scalef.sh(<8 x half> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}}, i8 %{{.*}}, i32 11)
  return _mm_maskz_scalef_round_sh(__U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm_roundscale_round_sh(__m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_roundscale_round_sh
  // CHECK: @llvm.x86.avx512fp16.mask.rndscale.sh
  return _mm_roundscale_round_sh(__A, __B, 3, _MM_FROUND_NO_EXC);
}

__m128h test_mm_roundscale_sh(__m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_roundscale_sh
  // CHECK: @llvm.x86.avx512fp16.mask.rndscale.sh
  return _mm_roundscale_sh(__A, __B, 3);
}

__m128h test_mm_mask_roundscale_sh(__m128h __W, __mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_mask_roundscale_sh
  // CHECK: @llvm.x86.avx512fp16.mask.rndscale.sh
  return _mm_mask_roundscale_sh(__W, __U, __A, __B, 3);
}

__m128h test_mm_mask_roundscale_round_sh(__m128h __W, __mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_mask_roundscale_round_sh
  // CHECK: @llvm.x86.avx512fp16.mask.rndscale.sh
  return _mm_mask_roundscale_round_sh(__W, __U, __A, __B, 3, _MM_FROUND_NO_EXC);
}

__m128h test_mm_maskz_roundscale_round_sh(__mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_maskz_roundscale_round_sh
  // CHECK: @llvm.x86.avx512fp16.mask.rndscale.sh
  return _mm_maskz_roundscale_round_sh(__U, __A, __B, 3, _MM_FROUND_NO_EXC);
}

__m128h test_mm_maskz_roundscale_sh(__mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_maskz_roundscale_sh
  // CHECK: @llvm.x86.avx512fp16.mask.rndscale.sh
  return _mm_maskz_roundscale_sh(__U, __A, __B, 3);
}

__m128h test_mm_reduce_sh(__m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_reduce_sh
  // CHECK: @llvm.x86.avx512fp16.mask.reduce.sh
  return _mm_reduce_sh(__A, __B, 4);
}

__m128h test_mm_mask_reduce_sh(__m128h __W, __mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_mask_reduce_sh
  // CHECK: @llvm.x86.avx512fp16.mask.reduce.sh
  return _mm_mask_reduce_sh(__W, __U, __A, __B, 4);
}

__m128h test_mm_maskz_reduce_sh(__mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_maskz_reduce_sh
  // CHECK: @llvm.x86.avx512fp16.mask.reduce.sh
  return _mm_maskz_reduce_sh(__U, __A, __B, 4);
}

__m128h test_mm_reduce_round_sh(__m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_reduce_round_sh
  // CHECK: @llvm.x86.avx512fp16.mask.reduce.sh
  return _mm_reduce_round_sh(__A, __B, 4, 8);
}

__m128h test_mm_mask_reduce_round_sh(__m128h __W, __mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_mask_reduce_round_sh
  // CHECK: @llvm.x86.avx512fp16.mask.reduce.sh
  return _mm_mask_reduce_round_sh(__W, __U, __A, __B, 4, 8);
}

__m128h test_mm_maskz_reduce_round_sh(__mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_maskz_reduce_round_sh
  // CHECK: @llvm.x86.avx512fp16.mask.reduce.sh
  return _mm_maskz_reduce_round_sh(__U, __A, __B, 4, 8);
}

__m512h test_mm512_sqrt_round_ph(__m512h __A) {
  // CHECK-LABEL: test_mm512_sqrt_round_ph
  // CHECK: @llvm.x86.avx512fp16.sqrt.ph.512(<32 x half> %{{.*}}, i32 11)
  return _mm512_sqrt_round_ph(__A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_mask_sqrt_round_ph(__m512h __W, __mmask32 __U, __m512h __A) {
  // CHECK-LABEL: test_mm512_mask_sqrt_round_ph
  // CHECK: @llvm.x86.avx512fp16.sqrt.ph.512(<32 x half> %{{.*}}, i32 11)
  // CHECK: bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}
  return _mm512_mask_sqrt_round_ph(__W, __U, __A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_maskz_sqrt_round_ph(__mmask32 __U, __m512h __A) {
  // CHECK-LABEL: test_mm512_maskz_sqrt_round_ph
  // CHECK: @llvm.x86.avx512fp16.sqrt.ph.512(<32 x half> %{{.*}}, i32 11)
  // CHECK: bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> {{.*}}
  return _mm512_maskz_sqrt_round_ph(__U, __A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_sqrt_ph(__m512h __A) {
  // CHECK-LABEL: test_mm512_sqrt_ph
  // CHECK: @llvm.sqrt.v32f16(<32 x half> %{{.*}})
  return _mm512_sqrt_ph(__A);
}
__m512h test_mm512_mask_sqrt_ph(__m512h __W, __mmask32 __U, __m512h __A) {
  // CHECK-LABEL: test_mm512_mask_sqrt_ph
  // CHECK: @llvm.sqrt.v32f16(<32 x half> %{{.*}})
  // CHECK: bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}
  return _mm512_mask_sqrt_ph(__W, __U, __A);
}
__m512h test_mm512_maskz_sqrt_ph(__mmask32 __U, __m512h __A) {
  // CHECK-LABEL: test_mm512_maskz_sqrt_ph
  // CHECK: @llvm.sqrt.v32f16(<32 x half> %{{.*}})
  // CHECK: bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> {{.*}}
  return _mm512_maskz_sqrt_ph(__U, __A);
}

__m128h test_mm_sqrt_round_sh(__m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_sqrt_round_sh
  // CHECK: call <8 x half> @llvm.x86.avx512fp16.mask.sqrt.sh(<8 x half> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}}, i8 -1, i32 11)
  return _mm_sqrt_round_sh(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm_mask_sqrt_round_sh(__m128h __W, __mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_mask_sqrt_round_sh
  // CHECK: call <8 x half> @llvm.x86.avx512fp16.mask.sqrt.sh(<8 x half> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}}, i8 {{.*}}, i32 11)
  return _mm_mask_sqrt_round_sh(__W, __U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm_maskz_sqrt_round_sh(__mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_maskz_sqrt_round_sh
  // CHECK: call <8 x half> @llvm.x86.avx512fp16.mask.sqrt.sh(<8 x half> %{{.*}}, <8 x half> %{{.*}}, <8 x half> %{{.*}}, i8 {{.*}}, i32 11)
  return _mm_maskz_sqrt_round_sh(__U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm_sqrt_sh(__m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_sqrt_sh
  // CHECK: %{{.*}} = call half @llvm.sqrt.f16(half %{{.*}})
  return _mm_sqrt_sh(__A, __B);
}
__m128h test_mm_mask_sqrt_sh(__m128h __W, __mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_mask_sqrt_sh
  // CHECK: %{{.*}} = call half @llvm.sqrt.f16(half %{{.*}})
  return _mm_mask_sqrt_sh(__W, __U, __A, __B);
}
__m128h test_mm_maskz_sqrt_sh(__mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_maskz_sqrt_sh
  // CHECK: %{{.*}} = call half @llvm.sqrt.f16(half %{{.*}})
  return _mm_maskz_sqrt_sh(__U, __A, __B);
}

__mmask32 test_mm512_mask_fpclass_ph_mask(__mmask32 __U, __m512h __A) {
  // CHECK-LABEL: test_mm512_mask_fpclass_ph_mask
  // CHECK: @llvm.x86.avx512fp16.fpclass.ph.512
  return _mm512_mask_fpclass_ph_mask(__U, __A, 4);
}

__mmask32 test_mm512_fpclass_ph_mask(__m512h __A) {
  // CHECK-LABEL: test_mm512_fpclass_ph_mask
  // CHECK: @llvm.x86.avx512fp16.fpclass.ph.512
  return _mm512_fpclass_ph_mask(__A, 4);
}

__mmask8 test_mm_fpclash_sh_mask(__m128 __A) {
  // CHECK-LABEL: test_mm_fpclash_sh_mask
  // CHECK: @llvm.x86.avx512fp16.mask.fpclass.sh
  return _mm_fpclass_sh_mask(__A, 2);
}

__mmask8 test_mm_mask_fpclash_sh_mask(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: test_mm_mask_fpclash_sh_mask
  // CHECK: @llvm.x86.avx512fp16.mask.fpclass.sh
  return _mm_mask_fpclass_sh_mask(__U, __A, 2);
}

__m128h test_mm512_cvt_roundpd_ph(__m512d A) {
  // CHECK-LABEL: test_mm512_cvt_roundpd_ph
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtpd2ph.512
  return _mm512_cvt_roundpd_ph(A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm512_mask_cvt_roundpd_ph(__m128h A, __mmask8 B, __m512d C) {
  // CHECK-LABEL: test_mm512_mask_cvt_roundpd_ph
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtpd2ph.512
  return _mm512_mask_cvt_roundpd_ph(A, B, C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm512_maskz_cvt_roundpd_ph(__mmask8 A, __m512d B) {
  // CHECK-LABEL: test_mm512_maskz_cvt_roundpd_ph
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtpd2ph.512
  return _mm512_maskz_cvt_roundpd_ph(A, B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm512_cvtpd_ph(__m512d A) {
  // CHECK-LABEL: test_mm512_cvtpd_ph
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtpd2ph.512
  return _mm512_cvtpd_ph(A);
}

__m128h test_mm512_mask_cvtpd_ph(__m128h A, __mmask8 B, __m512d C) {
  // CHECK-LABEL: test_mm512_mask_cvtpd_ph
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtpd2ph.512
  return _mm512_mask_cvtpd_ph(A, B, C);
}

__m128h test_mm512_maskz_cvtpd_ph(__mmask8 A, __m512d B) {
  // CHECK-LABEL: test_mm512_maskz_cvtpd_ph
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtpd2ph.512
  return _mm512_maskz_cvtpd_ph(A, B);
}

__m512d test_mm512_cvt_roundph_pd(__m128h A) {
  // CHECK-LABEL: test_mm512_cvt_roundph_pd
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2pd.512
  return _mm512_cvt_roundph_pd(A, _MM_FROUND_NO_EXC);
}

__m512d test_mm512_mask_cvt_roundph_pd(__m512d A, __mmask8 B, __m128h C) {
  // CHECK-LABEL: test_mm512_mask_cvt_roundph_pd
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2pd.512
  return _mm512_mask_cvt_roundph_pd(A, B, C, _MM_FROUND_NO_EXC);
}

__m512d test_mm512_maskz_cvt_roundph_pd(__mmask8 A, __m128h B) {
  // CHECK-LABEL: test_mm512_maskz_cvt_roundph_pd
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2pd.512
  return _mm512_maskz_cvt_roundph_pd(A, B, _MM_FROUND_NO_EXC);
}

__m512d test_mm512_cvtph_pd(__m128h A) {
  // CHECK-LABEL: test_mm512_cvtph_pd
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2pd.512
  return _mm512_cvtph_pd(A);
}

__m512d test_mm512_mask_cvtph_pd(__m512d A, __mmask8 B, __m128h C) {
  // CHECK-LABEL: test_mm512_mask_cvtph_pd
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2pd.512
  return _mm512_mask_cvtph_pd(A, B, C);
}

__m512d test_mm512_maskz_cvtph_pd(__mmask8 A, __m128h B) {
  // CHECK-LABEL: test_mm512_maskz_cvtph_pd
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2pd.512
  return _mm512_maskz_cvtph_pd(A, B);
}

__m128 test_mm_cvt_roundsh_ss(__m128 A, __m128h B) {
  // CHECK-LABEL: test_mm_cvt_roundsh_ss
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtsh2ss.round
  return _mm_cvt_roundsh_ss(A, B, _MM_FROUND_NO_EXC);
}

__m128 test_mm_mask_cvt_roundsh_ss(__m128 A, __mmask8 B, __m128 C, __m128h D) {
  // CHECK-LABEL: test_mm_mask_cvt_roundsh_ss
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtsh2ss.round
  return _mm_mask_cvt_roundsh_ss(A, B, C, D, _MM_FROUND_NO_EXC);
}

__m128 test_mm_maskz_cvt_roundsh_ss(__mmask8 A, __m128 B, __m128h C) {
  // CHECK-LABEL: test_mm_maskz_cvt_roundsh_ss
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtsh2ss.round
  return _mm_maskz_cvt_roundsh_ss(A, B, C, _MM_FROUND_NO_EXC);
}

__m128 test_mm_cvtsh_ss(__m128 A, __m128h B) {
  // CHECK-LABEL: test_mm_cvtsh_ss
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtsh2ss.round
  return _mm_cvtsh_ss(A, B);
}

__m128 test_mm_mask_cvtsh_ss(__m128 A, __mmask8 B, __m128 C, __m128h D) {
  // CHECK-LABEL: test_mm_mask_cvtsh_ss
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtsh2ss.round
  return _mm_mask_cvtsh_ss(A, B, C, D);
}

__m128 test_mm_maskz_cvtsh_ss(__mmask8 A, __m128 B, __m128h C) {
  // CHECK-LABEL: test_mm_maskz_cvtsh_ss
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtsh2ss.round
  return _mm_maskz_cvtsh_ss(A, B, C);
}

__m128h test_mm_cvt_roundss_sh(__m128h A, __m128 B) {
  // CHECK-LABEL: test_mm_cvt_roundss_sh
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtss2sh.round
  return _mm_cvt_roundss_sh(A, B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm_mask_cvt_roundss_sh(__m128h A, __mmask8 B, __m128h C, __m128 D) {
  // CHECK-LABEL: test_mm_mask_cvt_roundss_sh
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtss2sh.round
  return _mm_mask_cvt_roundss_sh(A, B, C, D, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm_maskz_cvt_roundss_sh(__mmask8 A, __m128h B, __m128 C) {
  // CHECK-LABEL: test_mm_maskz_cvt_roundss_sh
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtss2sh.round
  return _mm_maskz_cvt_roundss_sh(A, B, C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm_cvtss_sh(__m128h A, __m128 B) {
  // CHECK-LABEL: test_mm_cvtss_sh
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtss2sh.round
  return _mm_cvtss_sh(A, B);
}

__m128h test_mm_mask_cvtss_sh(__m128h A, __mmask8 B, __m128h C, __m128 D) {
  // CHECK-LABEL: test_mm_mask_cvtss_sh
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtss2sh.round
  return _mm_mask_cvtss_sh(A, B, C, D);
}

__m128h test_mm_maskz_cvtss_sh(__mmask8 A, __m128h B, __m128 C) {
  // CHECK-LABEL: test_mm_maskz_cvtss_sh
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtss2sh.round
  return _mm_maskz_cvtss_sh(A, B, C);
}

__m128h test_mm_cvt_roundsd_sh(__m128h A, __m128d B) {
  // CHECK-LABEL: test_mm_cvt_roundsd_sh
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtsd2sh.round
  return _mm_cvt_roundsd_sh(A, B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm_mask_cvt_roundsd_sh(__m128h A, __mmask8 B, __m128h C, __m128d D) {
  // CHECK-LABEL: test_mm_mask_cvt_roundsd_sh
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtsd2sh.round
  return _mm_mask_cvt_roundsd_sh(A, B, C, D, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm_maskz_cvt_roundsd_sh(__mmask8 A, __m128h B, __m128d C) {
  // CHECK-LABEL: test_mm_maskz_cvt_roundsd_sh
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtsd2sh.round
  return _mm_maskz_cvt_roundsd_sh(A, B, C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm_cvtsd_sh(__m128h A, __m128d B) {
  // CHECK-LABEL: test_mm_cvtsd_sh
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtsd2sh.round
  return _mm_cvtsd_sh(A, B);
}

__m128h test_mm_mask_cvtsd_sh(__m128h A, __mmask8 B, __m128h C, __m128d D) {
  // CHECK-LABEL: test_mm_mask_cvtsd_sh
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtsd2sh.round
  return _mm_mask_cvtsd_sh(A, B, C, D);
}

__m128h test_mm_maskz_cvtsd_sh(__mmask8 A, __m128h B, __m128d C) {
  // CHECK-LABEL: test_mm_maskz_cvtsd_sh
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtsd2sh.round
  return _mm_maskz_cvtsd_sh(A, B, C);
}

__m128d test_mm_cvt_roundsh_sd(__m128d A, __m128h B) {
  // CHECK-LABEL: test_mm_cvt_roundsh_sd
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtsh2sd.round
  return _mm_cvt_roundsh_sd(A, B, _MM_FROUND_NO_EXC);
}

__m128d test_mm_mask_cvt_roundsh_sd(__m128d A, __mmask8 B, __m128d C, __m128h D) {
  // CHECK-LABEL: test_mm_mask_cvt_roundsh_sd
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtsh2sd.round
  return _mm_mask_cvt_roundsh_sd(A, B, C, D, _MM_FROUND_NO_EXC);
}

__m128d test_mm_maskz_cvt_roundsh_sd(__mmask8 A, __m128d B, __m128h C) {
  // CHECK-LABEL: test_mm_maskz_cvt_roundsh_sd
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtsh2sd.round
  return _mm_maskz_cvt_roundsh_sd(A, B, C, _MM_FROUND_NO_EXC);
}

__m128d test_mm_cvtsh_sd(__m128d A, __m128h B) {
  // CHECK-LABEL: test_mm_cvtsh_sd
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtsh2sd.round
  return _mm_cvtsh_sd(A, B);
}

__m128d test_mm_mask_cvtsh_sd(__m128d A, __mmask8 B, __m128d C, __m128h D) {
  // CHECK-LABEL: test_mm_mask_cvtsh_sd
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtsh2sd.round
  return _mm_mask_cvtsh_sd(A, B, C, D);
}

__m128d test_mm_maskz_cvtsh_sd(__mmask8 A, __m128d B, __m128h C) {
  // CHECK-LABEL: test_mm_maskz_cvtsh_sd
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtsh2sd.round
  return _mm_maskz_cvtsh_sd(A, B, C);
}

__m512i test_mm512_cvt_roundph_epi16(__m512h A) {
  // CHECK-LABEL: test_mm512_cvt_roundph_epi16
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2w.512
  return _mm512_cvt_roundph_epi16(A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512i test_mm512_mask_cvt_roundph_epi16(__m512i A, __mmask32 B, __m512h C) {
  // CHECK-LABEL: test_mm512_mask_cvt_roundph_epi16
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2w.512
  return _mm512_mask_cvt_roundph_epi16(A, B, C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512i test_mm512_maskz_cvt_roundph_epi16(__mmask32 A, __m512h B) {
  // CHECK-LABEL: test_mm512_maskz_cvt_roundph_epi16
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2w.512
  return _mm512_maskz_cvt_roundph_epi16(A, B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512i test_mm512_cvtph_epi16(__m512h A) {
  // CHECK-LABEL: test_mm512_cvtph_epi16
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2w.512
  return _mm512_cvtph_epi16(A);
}

__m512i test_mm512_mask_cvtph_epi16(__m512i A, __mmask32 B, __m512h C) {
  // CHECK-LABEL: test_mm512_mask_cvtph_epi16
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2w.512
  return _mm512_mask_cvtph_epi16(A, B, C);
}

__m512i test_mm512_maskz_cvtph_epi16(__mmask32 A, __m512h B) {
  // CHECK-LABEL: test_mm512_maskz_cvtph_epi16
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2w.512
  return _mm512_maskz_cvtph_epi16(A, B);
}

__m512i test_mm512_cvtt_roundph_epi16(__m512h A) {
  // CHECK-LABEL: test_mm512_cvtt_roundph_epi16
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2w.512
  return _mm512_cvtt_roundph_epi16(A, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_mask_cvtt_roundph_epi16(__m512i A, __mmask32 B, __m512h C) {
  // CHECK-LABEL: test_mm512_mask_cvtt_roundph_epi16
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2w.512
  return _mm512_mask_cvtt_roundph_epi16(A, B, C, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_maskz_cvtt_roundph_epi16(__mmask32 A, __m512h B) {
  // CHECK-LABEL: test_mm512_maskz_cvtt_roundph_epi16
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2w.512
  return _mm512_maskz_cvtt_roundph_epi16(A, B, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_cvttph_epi16(__m512h A) {
  // CHECK-LABEL: test_mm512_cvttph_epi16
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2w.512
  return _mm512_cvttph_epi16(A);
}

__m512i test_mm512_mask_cvttph_epi16(__m512i A, __mmask32 B, __m512h C) {
  // CHECK-LABEL: test_mm512_mask_cvttph_epi16
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2w.512
  return _mm512_mask_cvttph_epi16(A, B, C);
}

__m512i test_mm512_maskz_cvttph_epi16(__mmask32 A, __m512h B) {
  // CHECK-LABEL: test_mm512_maskz_cvttph_epi16
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2w.512
  return _mm512_maskz_cvttph_epi16(A, B);
}

__m512h test_mm512_cvt_roundepi16_ph(__m512i A) {
  // CHECK-LABEL: test_mm512_cvt_roundepi16_ph
  // CHECK:   @llvm.x86.avx512.sitofp.round.v32f16.v32i16
  return _mm512_cvt_roundepi16_ph(A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_mask_cvt_roundepi16_ph(__m512h A, __mmask32 B, __m512i C) {
  // CHECK-LABEL: test_mm512_mask_cvt_roundepi16_ph
  // CHECK: @llvm.x86.avx512.sitofp.round.v32f16.v32i16
  return _mm512_mask_cvt_roundepi16_ph(A, B, C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_maskz_cvt_roundepi16_ph(__mmask32 A, __m512i B) {
  // CHECK-LABEL: test_mm512_maskz_cvt_roundepi16_ph
  // CHECK: @llvm.x86.avx512.sitofp.round.v32f16.v32i16
  return _mm512_maskz_cvt_roundepi16_ph(A, B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_cvtepi16_ph(__m512i A) {
  // CHECK-LABEL: test_mm512_cvtepi16_ph
  // CHECK: %{{.*}} = sitofp <32 x i16> %{{.*}} to <32 x half>
  return _mm512_cvtepi16_ph(A);
}

__m512h test_mm512_mask_cvtepi16_ph(__m512h A, __mmask32 B, __m512i C) {
  // CHECK-LABEL: test_mm512_mask_cvtepi16_ph
  // CHECK: %{{.*}} = sitofp <32 x i16> %{{.*}} to <32 x half>
  return _mm512_mask_cvtepi16_ph(A, B, C);
}

__m512h test_mm512_maskz_cvtepi16_ph(__mmask32 A, __m512i B) {
  // CHECK-LABEL: test_mm512_maskz_cvtepi16_ph
  // CHECK: %{{.*}} = sitofp <32 x i16> %{{.*}} to <32 x half>
  return _mm512_maskz_cvtepi16_ph(A, B);
}

__m512i test_mm512_cvt_roundph_epu16(__m512h A) {
  // CHECK-LABEL: test_mm512_cvt_roundph_epu16
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2uw.512
  return _mm512_cvt_roundph_epu16(A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512i test_mm512_mask_cvt_roundph_epu16(__m512i A, __mmask32 B, __m512h C) {
  // CHECK-LABEL: test_mm512_mask_cvt_roundph_epu16
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2uw.512
  return _mm512_mask_cvt_roundph_epu16(A, B, C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512i test_mm512_maskz_cvt_roundph_epu16(__mmask32 A, __m512h B) {
  // CHECK-LABEL: test_mm512_maskz_cvt_roundph_epu16
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2uw.512
  return _mm512_maskz_cvt_roundph_epu16(A, B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512i test_mm512_cvtph_epu16(__m512h A) {
  // CHECK-LABEL: test_mm512_cvtph_epu16
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2uw.512
  return _mm512_cvtph_epu16(A);
}

__m512i test_mm512_mask_cvtph_epu16(__m512i A, __mmask32 B, __m512h C) {
  // CHECK-LABEL: test_mm512_mask_cvtph_epu16
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2uw.512
  return _mm512_mask_cvtph_epu16(A, B, C);
}

__m512i test_mm512_maskz_cvtph_epu16(__mmask32 A, __m512h B) {
  // CHECK-LABEL: test_mm512_maskz_cvtph_epu16
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2uw.512
  return _mm512_maskz_cvtph_epu16(A, B);
}

__m512i test_mm512_cvtt_roundph_epu16(__m512h A) {
  // CHECK-LABEL: test_mm512_cvtt_roundph_epu16
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2uw.512
  return _mm512_cvtt_roundph_epu16(A, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_mask_cvtt_roundph_epu16(__m512i A, __mmask32 B, __m512h C) {
  // CHECK-LABEL: test_mm512_mask_cvtt_roundph_epu16
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2uw.512
  return _mm512_mask_cvtt_roundph_epu16(A, B, C, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_maskz_cvtt_roundph_epu16(__mmask32 A, __m512h B) {
  // CHECK-LABEL: test_mm512_maskz_cvtt_roundph_epu16
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2uw.512
  return _mm512_maskz_cvtt_roundph_epu16(A, B, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_cvttph_epu16(__m512h A) {
  // CHECK-LABEL: test_mm512_cvttph_epu16
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2uw.512
  return _mm512_cvttph_epu16(A);
}

__m512i test_mm512_mask_cvttph_epu16(__m512i A, __mmask32 B, __m512h C) {
  // CHECK-LABEL: test_mm512_mask_cvttph_epu16
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2uw.512
  return _mm512_mask_cvttph_epu16(A, B, C);
}

__m512i test_mm512_maskz_cvttph_epu16(__mmask32 A, __m512h B) {
  // CHECK-LABEL: test_mm512_maskz_cvttph_epu16
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2uw.512
  return _mm512_maskz_cvttph_epu16(A, B);
}

__m512h test_mm512_cvt_roundepu16_ph(__m512i A) {
  // CHECK-LABEL: test_mm512_cvt_roundepu16_ph
  // CHECK: @llvm.x86.avx512.uitofp.round.v32f16.v32i16
  return _mm512_cvt_roundepu16_ph(A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_mask_cvt_roundepu16_ph(__m512h A, __mmask32 B, __m512i C) {
  // CHECK-LABEL: test_mm512_mask_cvt_roundepu16_ph
  // CHECK: @llvm.x86.avx512.uitofp.round.v32f16.v32i16
  return _mm512_mask_cvt_roundepu16_ph(A, B, C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_maskz_cvt_roundepu16_ph(__mmask32 A, __m512i B) {
  // CHECK-LABEL: test_mm512_maskz_cvt_roundepu16_ph
  // CHECK: @llvm.x86.avx512.uitofp.round.v32f16.v32i16
  return _mm512_maskz_cvt_roundepu16_ph(A, B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_cvtepu16_ph(__m512i A) {
  // CHECK-LABEL: test_mm512_cvtepu16_ph
  // CHECK: %{{.*}} = uitofp <32 x i16> %{{.*}} to <32 x half>
  return _mm512_cvtepu16_ph(A);
}

__m512h test_mm512_mask_cvtepu16_ph(__m512h A, __mmask32 B, __m512i C) {
  // CHECK-LABEL: test_mm512_mask_cvtepu16_ph
  // CHECK: %{{.*}} = uitofp <32 x i16> %{{.*}} to <32 x half>
  return _mm512_mask_cvtepu16_ph(A, B, C);
}

__m512h test_mm512_maskz_cvtepu16_ph(__mmask32 A, __m512i B) {
  // CHECK-LABEL: test_mm512_maskz_cvtepu16_ph
  // CHECK: %{{.*}} = uitofp <32 x i16> %{{.*}} to <32 x half>
  return _mm512_maskz_cvtepu16_ph(A, B);
}

__m512i test_mm512_cvt_roundph_epi32(__m256h A) {
  // CHECK-LABEL: test_mm512_cvt_roundph_epi32
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2dq.512
  return _mm512_cvt_roundph_epi32(A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512i test_mm512_mask_cvt_roundph_epi32(__m512i A, __mmask16 B, __m256h C) {
  // CHECK-LABEL: test_mm512_mask_cvt_roundph_epi32
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2dq.512
  return _mm512_mask_cvt_roundph_epi32(A, B, C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512i test_mm512_maskz_cvt_roundph_epi32(__mmask16 A, __m256h B) {
  // CHECK-LABEL: test_mm512_maskz_cvt_roundph_epi32
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2dq.512
  return _mm512_maskz_cvt_roundph_epi32(A, B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512i test_mm512_cvtph_epi32(__m256h A) {
  // CHECK-LABEL: test_mm512_cvtph_epi32
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2dq.512
  return _mm512_cvtph_epi32(A);
}

__m512i test_mm512_mask_cvtph_epi32(__m512i A, __mmask16 B, __m256h C) {
  // CHECK-LABEL: test_mm512_mask_cvtph_epi32
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2dq.512
  return _mm512_mask_cvtph_epi32(A, B, C);
}

__m512i test_mm512_maskz_cvtph_epi32(__mmask16 A, __m256h B) {
  // CHECK-LABEL: test_mm512_maskz_cvtph_epi32
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2dq.512
  return _mm512_maskz_cvtph_epi32(A, B);
}

__m512i test_mm512_cvt_roundph_epu32(__m256h A) {
  // CHECK-LABEL: test_mm512_cvt_roundph_epu32
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2udq.512
  return _mm512_cvt_roundph_epu32(A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512i test_mm512_mask_cvt_roundph_epu32(__m512i A, __mmask16 B, __m256h C) {
  // CHECK-LABEL: test_mm512_mask_cvt_roundph_epu32
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2udq.512
  return _mm512_mask_cvt_roundph_epu32(A, B, C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512i test_mm512_maskz_cvt_roundph_epu32(__mmask16 A, __m256h B) {
  // CHECK-LABEL: test_mm512_maskz_cvt_roundph_epu32
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2udq.512
  return _mm512_maskz_cvt_roundph_epu32(A, B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512i test_mm512_cvtph_epu32(__m256h A) {
  // CHECK-LABEL: test_mm512_cvtph_epu32
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2udq.512
  return _mm512_cvtph_epu32(A);
}

__m512i test_mm512_mask_cvtph_epu32(__m512i A, __mmask16 B, __m256h C) {
  // CHECK-LABEL: test_mm512_mask_cvtph_epu32
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2udq.512
  return _mm512_mask_cvtph_epu32(A, B, C);
}

__m512i test_mm512_maskz_cvtph_epu32(__mmask16 A, __m256h B) {
  // CHECK-LABEL: test_mm512_maskz_cvtph_epu32
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2udq.512
  return _mm512_maskz_cvtph_epu32(A, B);
}

__m256h test_mm512_cvt_roundepi32_ph(__m512i A) {
  // CHECK-LABEL: test_mm512_cvt_roundepi32_ph
  // CHECK: @llvm.x86.avx512.sitofp.round.v16f16.v16i32
  return _mm512_cvt_roundepi32_ph(A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm512_mask_cvt_roundepi32_ph(__m256h A, __mmask16 B, __m512i C) {
  // CHECK-LABEL: test_mm512_mask_cvt_roundepi32_ph
  // CHECK: @llvm.x86.avx512.sitofp.round.v16f16.v16i32
  return _mm512_mask_cvt_roundepi32_ph(A, B, C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm512_maskz_cvt_roundepi32_ph(__mmask16 A, __m512i B) {
  // CHECK-LABEL: test_mm512_maskz_cvt_roundepi32_ph
  // CHECK: @llvm.x86.avx512.sitofp.round.v16f16.v16i32
  return _mm512_maskz_cvt_roundepi32_ph(A, B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm512_cvtepi32_ph(__m512i A) {
  // CHECK-LABEL: test_mm512_cvtepi32_ph
  // CHECK: %{{.*}} = sitofp <16 x i32> %{{.*}} to <16 x half>
  return _mm512_cvtepi32_ph(A);
}

__m256h test_mm512_mask_cvtepi32_ph(__m256h A, __mmask16 B, __m512i C) {
  // CHECK-LABEL: test_mm512_mask_cvtepi32_ph
  // CHECK: %{{.*}} = sitofp <16 x i32> %{{.*}} to <16 x half>
  return _mm512_mask_cvtepi32_ph(A, B, C);
}

__m256h test_mm512_maskz_cvtepi32_ph(__mmask16 A, __m512i B) {
  // CHECK-LABEL: test_mm512_maskz_cvtepi32_ph
  // CHECK: %{{.*}} = sitofp <16 x i32> %{{.*}} to <16 x half>
  return _mm512_maskz_cvtepi32_ph(A, B);
}

__m256h test_mm512_cvt_roundepu32_ph(__m512i A) {
  // CHECK-LABEL: test_mm512_cvt_roundepu32_ph
  // CHECK: @llvm.x86.avx512.uitofp.round.v16f16.v16i32
  return _mm512_cvt_roundepu32_ph(A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm512_mask_cvt_roundepu32_ph(__m256h A, __mmask16 B, __m512i C) {
  // CHECK-LABEL: test_mm512_mask_cvt_roundepu32_ph
  // CHECK: @llvm.x86.avx512.uitofp.round.v16f16.v16i32
  return _mm512_mask_cvt_roundepu32_ph(A, B, C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm512_maskz_cvt_roundepu32_ph(__mmask16 A, __m512i B) {
  // CHECK-LABEL: test_mm512_maskz_cvt_roundepu32_ph
  // CHECK: @llvm.x86.avx512.uitofp.round.v16f16.v16i32
  return _mm512_maskz_cvt_roundepu32_ph(A, B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm512_cvtepu32_ph(__m512i A) {
  // CHECK-LABEL: test_mm512_cvtepu32_ph
  // CHECK: %{{.*}} = uitofp <16 x i32> %{{.*}} to <16 x half>
  return _mm512_cvtepu32_ph(A);
}

__m256h test_mm512_mask_cvtepu32_ph(__m256h A, __mmask16 B, __m512i C) {
  // CHECK-LABEL: test_mm512_mask_cvtepu32_ph
  // CHECK: %{{.*}} = uitofp <16 x i32> %{{.*}} to <16 x half>
  return _mm512_mask_cvtepu32_ph(A, B, C);
}

__m256h test_mm512_maskz_cvtepu32_ph(__mmask16 A, __m512i B) {
  // CHECK-LABEL: test_mm512_maskz_cvtepu32_ph
  // CHECK: %{{.*}} = uitofp <16 x i32> %{{.*}} to <16 x half>
  return _mm512_maskz_cvtepu32_ph(A, B);
}

__m512i test_mm512_cvtt_roundph_epi32(__m256h A) {
  // CHECK-LABEL: test_mm512_cvtt_roundph_epi32
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2dq.512
  return _mm512_cvtt_roundph_epi32(A, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_mask_cvtt_roundph_epi32(__m512i A, __mmask16 B, __m256h C) {
  // CHECK-LABEL: test_mm512_mask_cvtt_roundph_epi32
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2dq.512
  return _mm512_mask_cvtt_roundph_epi32(A, B, C, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_maskz_cvtt_roundph_epi32(__mmask16 A, __m256h B) {
  // CHECK-LABEL: test_mm512_maskz_cvtt_roundph_epi32
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2dq.512
  return _mm512_maskz_cvtt_roundph_epi32(A, B, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_cvttph_epi32(__m256h A) {
  // CHECK-LABEL: test_mm512_cvttph_epi32
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2dq.512
  return _mm512_cvttph_epi32(A);
}

__m512i test_mm512_mask_cvttph_epi32(__m512i A, __mmask16 B, __m256h C) {
  // CHECK-LABEL: test_mm512_mask_cvttph_epi32
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2dq.512
  return _mm512_mask_cvttph_epi32(A, B, C);
}

__m512i test_mm512_maskz_cvttph_epi32(__mmask16 A, __m256h B) {
  // CHECK-LABEL: test_mm512_maskz_cvttph_epi32
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2dq.512
  return _mm512_maskz_cvttph_epi32(A, B);
}

__m512i test_mm512_cvtt_roundph_epu32(__m256h A) {
  // CHECK-LABEL: test_mm512_cvtt_roundph_epu32
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2udq.512
  return _mm512_cvtt_roundph_epu32(A, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_mask_cvtt_roundph_epu32(__m512i A, __mmask16 B, __m256h C) {
  // CHECK-LABEL: test_mm512_mask_cvtt_roundph_epu32
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2udq.512
  return _mm512_mask_cvtt_roundph_epu32(A, B, C, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_maskz_cvtt_roundph_epu32(__mmask16 A, __m256h B) {
  // CHECK-LABEL: test_mm512_maskz_cvtt_roundph_epu32
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2udq.512
  return _mm512_maskz_cvtt_roundph_epu32(A, B, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_cvttph_epu32(__m256h A) {
  // CHECK-LABEL: test_mm512_cvttph_epu32
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2udq.512
  return _mm512_cvttph_epu32(A);
}

__m512i test_mm512_mask_cvttph_epu32(__m512i A, __mmask16 B, __m256h C) {
  // CHECK-LABEL: test_mm512_mask_cvttph_epu32
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2udq.512
  return _mm512_mask_cvttph_epu32(A, B, C);
}

__m512i test_mm512_maskz_cvttph_epu32(__mmask16 A, __m256h B) {
  // CHECK-LABEL: test_mm512_maskz_cvttph_epu32
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2udq.512
  return _mm512_maskz_cvttph_epu32(A, B);
}

__m128h test_mm512_cvt_roundepi64_ph(__m512i A) {
  // CHECK-LABEL: test_mm512_cvt_roundepi64_ph
  // CHECK: @llvm.x86.avx512.sitofp.round.v8f16.v8i64
  return _mm512_cvt_roundepi64_ph(A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm512_mask_cvt_roundepi64_ph(__m128h A, __mmask8 B, __m512i C) {
  // CHECK-LABEL: test_mm512_mask_cvt_roundepi64_ph
  // CHECK: @llvm.x86.avx512.sitofp.round.v8f16.v8i64
  return _mm512_mask_cvt_roundepi64_ph(A, B, C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm512_maskz_cvt_roundepi64_ph(__mmask8 A, __m512i B) {
  // CHECK-LABEL: test_mm512_maskz_cvt_roundepi64_ph
  // CHECK: @llvm.x86.avx512.sitofp.round.v8f16.v8i64
  return _mm512_maskz_cvt_roundepi64_ph(A, B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm512_cvtepi64_ph(__m512i A) {
  // CHECK-LABEL: test_mm512_cvtepi64_ph
  // CHECK: %{{.*}} = sitofp <8 x i64> %{{.*}} to <8 x half>
  return _mm512_cvtepi64_ph(A);
}

__m128h test_mm512_mask_cvtepi64_ph(__m128h A, __mmask8 B, __m512i C) {
  // CHECK-LABEL: test_mm512_mask_cvtepi64_ph
  // CHECK: %{{.*}} = sitofp <8 x i64> %{{.*}} to <8 x half>
  return _mm512_mask_cvtepi64_ph(A, B, C);
}

__m128h test_mm512_maskz_cvtepi64_ph(__mmask8 A, __m512i B) {
  // CHECK-LABEL: test_mm512_maskz_cvtepi64_ph
  // CHECK: %{{.*}} = sitofp <8 x i64> %{{.*}} to <8 x half>
  return _mm512_maskz_cvtepi64_ph(A, B);
}

__m512i test_mm512_cvt_roundph_epi64(__m128h A) {
  // CHECK-LABEL: test_mm512_cvt_roundph_epi64
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2qq.512
  return _mm512_cvt_roundph_epi64(A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512i test_mm512_mask_cvt_roundph_epi64(__m512i A, __mmask8 B, __m128h C) {
  // CHECK-LABEL: test_mm512_mask_cvt_roundph_epi64
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2qq.512
  return _mm512_mask_cvt_roundph_epi64(A, B, C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512i test_mm512_maskz_cvt_roundph_epi64(__mmask8 A, __m128h B) {
  // CHECK-LABEL: test_mm512_maskz_cvt_roundph_epi64
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2qq.512
  return _mm512_maskz_cvt_roundph_epi64(A, B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512i test_mm512_cvtph_epi64(__m128h A) {
  // CHECK-LABEL: test_mm512_cvtph_epi64
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2qq.512
  return _mm512_cvtph_epi64(A);
}

__m512i test_mm512_mask_cvtph_epi64(__m512i A, __mmask8 B, __m128h C) {
  // CHECK-LABEL: test_mm512_mask_cvtph_epi64
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2qq.512
  return _mm512_mask_cvtph_epi64(A, B, C);
}

__m512i test_mm512_maskz_cvtph_epi64(__mmask8 A, __m128h B) {
  // CHECK-LABEL: test_mm512_maskz_cvtph_epi64
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2qq.512
  return _mm512_maskz_cvtph_epi64(A, B);
}

__m128h test_mm512_cvt_roundepu64_ph(__m512i A) {
  // CHECK-LABEL: test_mm512_cvt_roundepu64_ph
  // CHECK: @llvm.x86.avx512.uitofp.round.v8f16.v8i64
  return _mm512_cvt_roundepu64_ph(A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm512_mask_cvt_roundepu64_ph(__m128h A, __mmask8 B, __m512i C) {
  // CHECK-LABEL: test_mm512_mask_cvt_roundepu64_ph
  // CHECK: @llvm.x86.avx512.uitofp.round.v8f16.v8i64
  return _mm512_mask_cvt_roundepu64_ph(A, B, C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm512_maskz_cvt_roundepu64_ph(__mmask8 A, __m512i B) {
  // CHECK-LABEL: test_mm512_maskz_cvt_roundepu64_ph
  // CHECK: @llvm.x86.avx512.uitofp.round.v8f16.v8i64
  return _mm512_maskz_cvt_roundepu64_ph(A, B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm512_cvtepu64_ph(__m512i A) {
  // CHECK-LABEL: test_mm512_cvtepu64_ph
  // CHECK: %{{.*}} = uitofp <8 x i64> %{{.*}} to <8 x half>
  return _mm512_cvtepu64_ph(A);
}

__m128h test_mm512_mask_cvtepu64_ph(__m128h A, __mmask8 B, __m512i C) {
  // CHECK-LABEL: test_mm512_mask_cvtepu64_ph
  // CHECK: %{{.*}} = uitofp <8 x i64> %{{.*}} to <8 x half>
  return _mm512_mask_cvtepu64_ph(A, B, C);
}

__m128h test_mm512_maskz_cvtepu64_ph(__mmask8 A, __m512i B) {
  // CHECK-LABEL: test_mm512_maskz_cvtepu64_ph
  // CHECK: %{{.*}} = uitofp <8 x i64> %{{.*}} to <8 x half>
  return _mm512_maskz_cvtepu64_ph(A, B);
}

__m512i test_mm512_cvt_roundph_epu64(__m128h A) {
  // CHECK-LABEL: test_mm512_cvt_roundph_epu64
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2uqq.512
  return _mm512_cvt_roundph_epu64(A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512i test_mm512_mask_cvt_roundph_epu64(__m512i A, __mmask8 B, __m128h C) {
  // CHECK-LABEL: test_mm512_mask_cvt_roundph_epu64
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2uqq.512
  return _mm512_mask_cvt_roundph_epu64(A, B, C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512i test_mm512_maskz_cvt_roundph_epu64(__mmask8 A, __m128h B) {
  // CHECK-LABEL: test_mm512_maskz_cvt_roundph_epu64
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2uqq.512
  return _mm512_maskz_cvt_roundph_epu64(A, B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512i test_mm512_cvtph_epu64(__m128h A) {
  // CHECK-LABEL: test_mm512_cvtph_epu64
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2uqq.512
  return _mm512_cvtph_epu64(A);
}

__m512i test_mm512_mask_cvtph_epu64(__m512i A, __mmask8 B, __m128h C) {
  // CHECK-LABEL: test_mm512_mask_cvtph_epu64
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2uqq.512
  return _mm512_mask_cvtph_epu64(A, B, C);
}

__m512i test_mm512_maskz_cvtph_epu64(__mmask8 A, __m128h B) {
  // CHECK-LABEL: test_mm512_maskz_cvtph_epu64
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2uqq.512
  return _mm512_maskz_cvtph_epu64(A, B);
}

__m512i test_mm512_cvtt_roundph_epi64(__m128h A) {
  // CHECK-LABEL: test_mm512_cvtt_roundph_epi64
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2qq.512
  return _mm512_cvtt_roundph_epi64(A, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_mask_cvtt_roundph_epi64(__m512i A, __mmask8 B, __m128h C) {
  // CHECK-LABEL: test_mm512_mask_cvtt_roundph_epi64
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2qq.512
  return _mm512_mask_cvtt_roundph_epi64(A, B, C, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_maskz_cvtt_roundph_epi64(__mmask8 A, __m128h B) {
  // CHECK-LABEL: test_mm512_maskz_cvtt_roundph_epi64
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2qq.512
  return _mm512_maskz_cvtt_roundph_epi64(A, B, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_cvttph_epi64(__m128h A) {
  // CHECK-LABEL: test_mm512_cvttph_epi64
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2qq.512
  return _mm512_cvttph_epi64(A);
}

__m512i test_mm512_mask_cvttph_epi64(__m512i A, __mmask8 B, __m128h C) {
  // CHECK-LABEL: test_mm512_mask_cvttph_epi64
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2qq.512
  return _mm512_mask_cvttph_epi64(A, B, C);
}

__m512i test_mm512_maskz_cvttph_epi64(__mmask8 A, __m128h B) {
  // CHECK-LABEL: test_mm512_maskz_cvttph_epi64
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2qq.512
  return _mm512_maskz_cvttph_epi64(A, B);
}

__m512i test_mm512_cvtt_roundph_epu64(__m128h A) {
  // CHECK-LABEL: test_mm512_cvtt_roundph_epu64
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2uqq.512
  return _mm512_cvtt_roundph_epu64(A, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_mask_cvtt_roundph_epu64(__m512i A, __mmask8 B, __m128h C) {
  // CHECK-LABEL: test_mm512_mask_cvtt_roundph_epu64
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2uqq.512
  return _mm512_mask_cvtt_roundph_epu64(A, B, C, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_maskz_cvtt_roundph_epu64(__mmask8 A, __m128h B) {
  // CHECK-LABEL: test_mm512_maskz_cvtt_roundph_epu64
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2uqq.512
  return _mm512_maskz_cvtt_roundph_epu64(A, B, _MM_FROUND_NO_EXC);
}

__m512i test_mm512_cvttph_epu64(__m128h A) {
  // CHECK-LABEL: test_mm512_cvttph_epu64
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2uqq.512
  return _mm512_cvttph_epu64(A);
}

__m512i test_mm512_mask_cvttph_epu64(__m512i A, __mmask8 B, __m128h C) {
  // CHECK-LABEL: test_mm512_mask_cvttph_epu64
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2uqq.512
  return _mm512_mask_cvttph_epu64(A, B, C);
}

__m512i test_mm512_maskz_cvttph_epu64(__mmask8 A, __m128h B) {
  // CHECK-LABEL: test_mm512_maskz_cvttph_epu64
  // CHECK: @llvm.x86.avx512fp16.mask.vcvttph2uqq.512
  return _mm512_maskz_cvttph_epu64(A, B);
}

int test_mm_cvt_roundsh_i32(__m128h A) {
  // CHECK-LABEL: test_mm_cvt_roundsh_i32
  // CHECK: @llvm.x86.avx512fp16.vcvtsh2si32
  return _mm_cvt_roundsh_i32(A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

int test_mm_cvtsh_i32(__m128h A) {
  // CHECK-LABEL: test_mm_cvtsh_i32
  // CHECK: @llvm.x86.avx512fp16.vcvtsh2si32
  return _mm_cvtsh_i32(A);
}

unsigned int test_mm_cvt_roundsh_u32(__m128h A) {
  // CHECK-LABEL: test_mm_cvt_roundsh_u32
  // CHECK: @llvm.x86.avx512fp16.vcvtsh2usi32
  return _mm_cvt_roundsh_u32(A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

unsigned int test_mm_cvtsh_u32(__m128h A) {
  // CHECK-LABEL: test_mm_cvtsh_u32
  // CHECK: @llvm.x86.avx512fp16.vcvtsh2usi32
  return _mm_cvtsh_u32(A);
}

#ifdef __x86_64__
long long test_mm_cvt_roundsh_i64(__m128h A) {
  // X64-LABEL: test_mm_cvt_roundsh_i64
  // X64: @llvm.x86.avx512fp16.vcvtsh2si64
  return _mm_cvt_roundsh_i64(A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

long long test_mm_cvtsh_i64(__m128h A) {
  // X64-LABEL: test_mm_cvtsh_i64
  // X64: @llvm.x86.avx512fp16.vcvtsh2si64
  return _mm_cvtsh_i64(A);
}

unsigned long long test_mm_cvt_roundsh_u64(__m128h A) {
  // X64-LABEL: test_mm_cvt_roundsh_u64
  // X64: @llvm.x86.avx512fp16.vcvtsh2usi64
  return _mm_cvt_roundsh_u64(A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

unsigned long long test_mm_cvtsh_u64(__m128h A) {
  // X64-LABEL: test_mm_cvtsh_u64
  // X64: @llvm.x86.avx512fp16.vcvtsh2usi64
  return _mm_cvtsh_u64(A);
}
#endif

__m128h test_mm_cvt_roundu32_sh(__m128h A, unsigned int B) {
  // CHECK-LABEL: test_mm_cvt_roundu32_sh
  // CHECK: @llvm.x86.avx512fp16.vcvtusi2sh
  return _mm_cvt_roundu32_sh(A, B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm_cvtu32_sh(__m128h A, unsigned int B) {
  // CHECK-LABEL: test_mm_cvtu32_sh
  // CHECK: %{{.*}} = uitofp i32 %{{.*}} to half
  return _mm_cvtu32_sh(A, B);
}

#ifdef __x86_64__
__m128h test_mm_cvt_roundu64_sh(__m128h A, unsigned long long B) {
  // X64-LABEL: test_mm_cvt_roundu64_sh
  // X64: @llvm.x86.avx512fp16.vcvtusi642sh
  return _mm_cvt_roundu64_sh(A, B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm_cvtu64_sh(__m128h A, unsigned long long B) {
  // X64-LABEL: test_mm_cvtu64_sh
  // X64: %{{.*}} = uitofp i64 %{{.*}} to half
  return _mm_cvtu64_sh(A, B);
}
#endif

__m128h test_mm_cvt_roundi32_sh(__m128h A, int B) {
  // CHECK-LABEL: test_mm_cvt_roundi32_sh
  // CHECK: @llvm.x86.avx512fp16.vcvtsi2sh
  return _mm_cvt_roundi32_sh(A, B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm_cvti32_sh(__m128h A, int B) {
  // CHECK-LABEL: test_mm_cvti32_sh
  // CHECK: %{{.*}} = sitofp i32 %{{.*}} to half
  return _mm_cvti32_sh(A, B);
}

#ifdef __x86_64__
__m128h test_mm_cvt_roundi64_sh(__m128h A, long long B) {
  // X64-LABEL: test_mm_cvt_roundi64_sh
  // X64: @llvm.x86.avx512fp16.vcvtsi642sh
  return _mm_cvt_roundi64_sh(A, B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm_cvti64_sh(__m128h A, long long B) {
  // X64-LABEL: test_mm_cvti64_sh
  // X64: %{{.*}} = sitofp i64 %{{.*}} to half
  return _mm_cvti64_sh(A, B);
}
#endif

int test_mm_cvtt_roundsh_i32(__m128h A) {
  // CHECK-LABEL: test_mm_cvtt_roundsh_i32
  // CHECK: @llvm.x86.avx512fp16.vcvttsh2si32
  return _mm_cvtt_roundsh_i32(A, _MM_FROUND_NO_EXC);
}

int test_mm_cvttsh_i32(__m128h A) {
  // CHECK-LABEL: test_mm_cvttsh_i32
  // CHECK: @llvm.x86.avx512fp16.vcvttsh2si32
  return _mm_cvttsh_i32(A);
}

#ifdef __x86_64__
long long test_mm_cvtt_roundsh_i64(__m128h A) {
  // X64-LABEL: test_mm_cvtt_roundsh_i64
  // X64: @llvm.x86.avx512fp16.vcvttsh2si64
  return _mm_cvtt_roundsh_i64(A, _MM_FROUND_NO_EXC);
}

long long test_mm_cvttsh_i64(__m128h A) {
  // X64-LABEL: test_mm_cvttsh_i64
  // X64: @llvm.x86.avx512fp16.vcvttsh2si64
  return _mm_cvttsh_i64(A);
}
#endif

unsigned int test_mm_cvtt_roundsh_u32(__m128h A) {
  // CHECK-LABEL: test_mm_cvtt_roundsh_u32
  // CHECK: @llvm.x86.avx512fp16.vcvttsh2usi32
  return _mm_cvtt_roundsh_u32(A, _MM_FROUND_NO_EXC);
}

unsigned int test_mm_cvttsh_u32(__m128h A) {
  // CHECK-LABEL: test_mm_cvttsh_u32
  // CHECK: @llvm.x86.avx512fp16.vcvttsh2usi32
  return _mm_cvttsh_u32(A);
}

#ifdef __x86_64__
unsigned long long test_mm_cvtt_roundsh_u64(__m128h A) {
  // X64-LABEL: test_mm_cvtt_roundsh_u64
  // X64: @llvm.x86.avx512fp16.vcvttsh2usi64
  return _mm_cvtt_roundsh_u64(A, _MM_FROUND_NO_EXC);
}

unsigned long long test_mm_cvttsh_u64(__m128h A) {
  // X64-LABEL: test_mm_cvttsh_u64
  // X64: @llvm.x86.avx512fp16.vcvttsh2usi64
  return _mm_cvttsh_u64(A);
}
#endif

__m512 test_mm512_cvtx_roundph_ps(__m256h A) {
  // CHECK-LABEL: test_mm512_cvtx_roundph_ps
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2psx.512
  return _mm512_cvtx_roundph_ps(A, _MM_FROUND_NO_EXC);
}

__m512 test_mm512_mask_cvtx_roundph_ps(__m512 A, __mmask16 B, __m256h C) {
  // CHECK-LABEL: test_mm512_mask_cvtx_roundph_ps
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2psx.512
  return _mm512_mask_cvtx_roundph_ps(A, B, C, _MM_FROUND_NO_EXC);
}

__m512 test_mm512_maskz_cvtx_roundph_ps(__mmask16 A, __m256h B) {
  // CHECK-LABEL: test_mm512_maskz_cvtx_roundph_ps
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2psx.512
  return _mm512_maskz_cvtx_roundph_ps(A, B, _MM_FROUND_NO_EXC);
}

__m512 test_mm512_cvtxph_ps(__m256h A) {
  // CHECK-LABEL: test_mm512_cvtxph_ps
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2psx.512
  return _mm512_cvtxph_ps(A);
}

__m512 test_mm512_mask_cvtxph_ps(__m512 A, __mmask16 B, __m256h C) {
  // CHECK-LABEL: test_mm512_mask_cvtxph_ps
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2psx.512
  return _mm512_mask_cvtxph_ps(A, B, C);
}

__m512 test_mm512_maskz_cvtxph_ps(__mmask16 A, __m256h B) {
  // CHECK-LABEL: test_mm512_maskz_cvtxph_ps
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtph2psx.512
  return _mm512_maskz_cvtxph_ps(A, B);
}

__m256h test_mm512_cvtx_roundps_ph(__m512 A) {
  // CHECK-LABEL: test_mm512_cvtx_roundps_ph
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtps2phx.512
  return _mm512_cvtx_roundps_ph(A, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm512_mask_cvtx_roundps_ph(__m256h A, __mmask16 B, __m512 C) {
  // CHECK-LABEL: test_mm512_mask_cvtx_roundps_ph
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtps2phx.512
  return _mm512_mask_cvtx_roundps_ph(A, B, C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm512_maskz_cvtx_roundps_ph(__mmask16 A, __m512 B) {
  // CHECK-LABEL: test_mm512_maskz_cvtx_roundps_ph
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtps2phx.512
  return _mm512_maskz_cvtx_roundps_ph(A, B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m256h test_mm512_cvtxps_ph(__m512 A) {
  // CHECK-LABEL: test_mm512_cvtxps_ph
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtps2phx.512
  return _mm512_cvtxps_ph(A);
}

__m256h test_mm512_mask_cvtxps_ph(__m256h A, __mmask16 B, __m512 C) {
  // CHECK-LABEL: test_mm512_mask_cvtxps_ph
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtps2phx.512
  return _mm512_mask_cvtxps_ph(A, B, C);
}

__m256h test_mm512_maskz_cvtxps_ph(__mmask16 A, __m512 B) {
  // CHECK-LABEL: test_mm512_maskz_cvtxps_ph
  // CHECK: @llvm.x86.avx512fp16.mask.vcvtps2phx.512
  return _mm512_maskz_cvtxps_ph(A, B);
}

__m512h test_mm512_fmadd_round_ph(__m512h __A, __m512h __B, __m512h __C) {
  // CHECK-LABEL: test_mm512_fmadd_round_ph
  // CHECK: @llvm.x86.avx512fp16.vfmadd.ph.512
  return _mm512_fmadd_round_ph(__A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_mask_fmadd_round_ph(__m512h __A, __mmask32 __U, __m512h __B, __m512h __C) {
  // CHECK-LABEL: test_mm512_mask_fmadd_round_ph
  // CHECK: @llvm.x86.avx512fp16.vfmadd.ph.512
  // CHECK: bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}
  return _mm512_mask_fmadd_round_ph(__A, __U, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_mask3_fmadd_round_ph(__m512h __A, __m512h __B, __m512h __C, __mmask32 __U) {
  // CHECK-LABEL: test_mm512_mask3_fmadd_round_ph
  // CHECK: @llvm.x86.avx512fp16.vfmadd.ph.512
  // CHECK: bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}
  return _mm512_mask3_fmadd_round_ph(__A, __B, __C, __U, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_maskz_fmadd_round_ph(__mmask32 __U, __m512h __A, __m512h __B, __m512h __C) {
  // CHECK-LABEL: test_mm512_maskz_fmadd_round_ph
  // CHECK: @llvm.x86.avx512fp16.vfmadd.ph.512
  // CHECK: bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> zeroinitializer
  return _mm512_maskz_fmadd_round_ph(__U, __A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_fmsub_round_ph(__m512h __A, __m512h __B, __m512h __C) {
  // CHECK-LABEL: test_mm512_fmsub_round_ph
  // CHECK: fneg
  // CHECK: @llvm.x86.avx512fp16.vfmadd.ph.512
  return _mm512_fmsub_round_ph(__A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_mask_fmsub_round_ph(__m512h __A, __mmask32 __U, __m512h __B, __m512h __C) {
  // CHECK-LABEL: test_mm512_mask_fmsub_round_ph
  // CHECK: fneg
  // CHECK: @llvm.x86.avx512fp16.vfmadd.ph.512
  // CHECK: bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}
  return _mm512_mask_fmsub_round_ph(__A, __U, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_maskz_fmsub_round_ph(__mmask32 __U, __m512h __A, __m512h __B, __m512h __C) {
  // CHECK-LABEL: test_mm512_maskz_fmsub_round_ph
  // CHECK: fneg
  // CHECK: @llvm.x86.avx512fp16.vfmadd.ph.512
  // CHECK: bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> zeroinitializer
  return _mm512_maskz_fmsub_round_ph(__U, __A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_fnmadd_round_ph(__m512h __A, __m512h __B, __m512h __C) {
  // CHECK-LABEL: test_mm512_fnmadd_round_ph
  // CHECK: fneg
  // CHECK: @llvm.x86.avx512fp16.vfmadd.ph.512
  return _mm512_fnmadd_round_ph(__A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_mask3_fnmadd_round_ph(__m512h __A, __m512h __B, __m512h __C, __mmask32 __U) {
  // CHECK-LABEL: test_mm512_mask3_fnmadd_round_ph
  // CHECK: fneg
  // CHECK: @llvm.x86.avx512fp16.vfmadd.ph.512
  // CHECK: bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}
  return _mm512_mask3_fnmadd_round_ph(__A, __B, __C, __U, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_maskz_fnmadd_round_ph(__mmask32 __U, __m512h __A, __m512h __B, __m512h __C) {
  // CHECK-LABEL: test_mm512_maskz_fnmadd_round_ph
  // CHECK: fneg
  // CHECK: @llvm.x86.avx512fp16.vfmadd.ph.512
  // CHECK: bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> zeroinitializer
  return _mm512_maskz_fnmadd_round_ph(__U, __A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_fnmsub_round_ph(__m512h __A, __m512h __B, __m512h __C) {
  // CHECK-LABEL: test_mm512_fnmsub_round_ph
  // CHECK: fneg
  // CHECK: fneg
  // CHECK: @llvm.x86.avx512fp16.vfmadd.ph.512
  return _mm512_fnmsub_round_ph(__A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_maskz_fnmsub_round_ph(__mmask32 __U, __m512h __A, __m512h __B, __m512h __C) {
  // CHECK-LABEL: test_mm512_maskz_fnmsub_round_ph
  // CHECK: fneg
  // CHECK: fneg
  // CHECK: @llvm.x86.avx512fp16.vfmadd.ph.512
  // CHECK: bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> zeroinitializer
  return _mm512_maskz_fnmsub_round_ph(__U, __A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_fmadd_ph(__m512h __A, __m512h __B, __m512h __C) {
  // CHECK-LABEL: test_mm512_fmadd_ph
  // CHECK: @llvm.fma.v32f16(<32 x half> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}})
  return _mm512_fmadd_ph(__A, __B, __C);
}

__m512h test_mm512_mask_fmadd_ph(__m512h __A, __mmask32 __U, __m512h __B, __m512h __C) {
  // CHECK-LABEL: test_mm512_mask_fmadd_ph
  // CHECK: @llvm.fma.v32f16(<32 x half> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}})
  return _mm512_mask_fmadd_ph(__A, __U, __B, __C);
}

__m512h test_mm512_mask3_fmadd_ph(__m512h __A, __m512h __B, __m512h __C, __mmask32 __U) {
  // CHECK-LABEL: test_mm512_mask3_fmadd_ph
  // CHECK: @llvm.fma.v32f16(<32 x half> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}})
  // CHECK: bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}
  return _mm512_mask3_fmadd_ph(__A, __B, __C, __U);
}

__m512h test_mm512_maskz_fmadd_ph(__mmask32 __U, __m512h __A, __m512h __B, __m512h __C) {
  // CHECK-LABEL: test_mm512_maskz_fmadd_ph
  // CHECK: @llvm.fma.v32f16(<32 x half> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}})
  // CHECK: bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> zeroinitializer
  return _mm512_maskz_fmadd_ph(__U, __A, __B, __C);
}

__m512h test_mm512_fmsub_ph(__m512h __A, __m512h __B, __m512h __C) {
  // CHECK-LABEL: test_mm512_fmsub_ph
  // CHECK: fneg
  // CHECK: @llvm.fma.v32f16(<32 x half> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}})
  return _mm512_fmsub_ph(__A, __B, __C);
}

__m512h test_mm512_mask_fmsub_ph(__m512h __A, __mmask32 __U, __m512h __B, __m512h __C) {
  // CHECK-LABEL: test_mm512_mask_fmsub_ph
  // CHECK: fneg
  // CHECK: @llvm.fma.v32f16(<32 x half> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}})
  // CHECK: bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}
  return _mm512_mask_fmsub_ph(__A, __U, __B, __C);
}

__m512h test_mm512_maskz_fmsub_ph(__mmask32 __U, __m512h __A, __m512h __B, __m512h __C) {
  // CHECK-LABEL: test_mm512_maskz_fmsub_ph
  // CHECK: fneg
  // CHECK: @llvm.fma.v32f16(<32 x half> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}})
  // CHECK: bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> zeroinitializer
  return _mm512_maskz_fmsub_ph(__U, __A, __B, __C);
}

__m512h test_mm512_fnmadd_ph(__m512h __A, __m512h __B, __m512h __C) {
  // CHECK-LABEL: test_mm512_fnmadd_ph
  // CHECK: fneg
  // CHECK: @llvm.fma.v32f16(<32 x half> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}})
  return _mm512_fnmadd_ph(__A, __B, __C);
}

__m512h test_mm512_mask3_fnmadd_ph(__m512h __A, __m512h __B, __m512h __C, __mmask32 __U) {
  // CHECK-LABEL: test_mm512_mask3_fnmadd_ph
  // CHECK: fneg
  // CHECK: @llvm.fma.v32f16(<32 x half> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}})
  // CHECK: bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}
  return _mm512_mask3_fnmadd_ph(__A, __B, __C, __U);
}

__m512h test_mm512_maskz_fnmadd_ph(__mmask32 __U, __m512h __A, __m512h __B, __m512h __C) {
  // CHECK-LABEL: test_mm512_maskz_fnmadd_ph
  // CHECK: fneg
  // CHECK: @llvm.fma.v32f16(<32 x half> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}})
  // CHECK: bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> zeroinitializer
  return _mm512_maskz_fnmadd_ph(__U, __A, __B, __C);
}

__m512h test_mm512_fnmsub_ph(__m512h __A, __m512h __B, __m512h __C) {
  // CHECK-LABEL: test_mm512_fnmsub_ph
  // CHECK: fneg
  // CHECK: fneg
  // CHECK: @llvm.fma.v32f16(<32 x half> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}})
  return _mm512_fnmsub_ph(__A, __B, __C);
}

__m512h test_mm512_maskz_fnmsub_ph(__mmask32 __U, __m512h __A, __m512h __B, __m512h __C) {
  // CHECK-LABEL: test_mm512_maskz_fnmsub_ph
  // CHECK: fneg
  // CHECK: fneg
  // CHECK: @llvm.fma.v32f16(<32 x half> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}})
  // CHECK: bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> zeroinitializer
  return _mm512_maskz_fnmsub_ph(__U, __A, __B, __C);
}

__m512h test_mm512_fmaddsub_round_ph(__m512h __A, __m512h __B, __m512h __C) {
  // CHECK-LABEL: test_mm512_fmaddsub_round_ph
  // CHECK: @llvm.x86.avx512fp16.vfmaddsub.ph.512
  return _mm512_fmaddsub_round_ph(__A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_mask_fmaddsub_round_ph(__m512h __A, __mmask32 __U, __m512h __B, __m512h __C) {
  // CHECK-LABEL: test_mm512_mask_fmaddsub_round_ph
  // CHECK: @llvm.x86.avx512fp16.vfmaddsub.ph.512
  // CHECK: bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}
  return _mm512_mask_fmaddsub_round_ph(__A, __U, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_mask3_fmaddsub_round_ph(__m512h __A, __m512h __B, __m512h __C, __mmask32 __U) {
  // CHECK-LABEL: test_mm512_mask3_fmaddsub_round_ph
  // CHECK: @llvm.x86.avx512fp16.vfmaddsub.ph.512
  // CHECK: bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}
  return _mm512_mask3_fmaddsub_round_ph(__A, __B, __C, __U, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_maskz_fmaddsub_round_ph(__mmask32 __U, __m512h __A, __m512h __B, __m512h __C) {
  // CHECK-LABEL: test_mm512_maskz_fmaddsub_round_ph
  // CHECK: @llvm.x86.avx512fp16.vfmaddsub.ph.512
  // CHECK: bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> zeroinitializer
  return _mm512_maskz_fmaddsub_round_ph(__U, __A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_fmsubadd_round_ph(__m512h __A, __m512h __B, __m512h __C) {
  // CHECK-LABEL: test_mm512_fmsubadd_round_ph
  // CHECK: fneg
  // CHECK: @llvm.x86.avx512fp16.vfmaddsub.ph.512
  return _mm512_fmsubadd_round_ph(__A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_mask_fmsubadd_round_ph(__m512h __A, __mmask32 __U, __m512h __B, __m512h __C) {
  // CHECK-LABEL: test_mm512_mask_fmsubadd_round_ph
  // CHECK: fneg
  // CHECK: @llvm.x86.avx512fp16.vfmaddsub.ph.512
  // CHECK: bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}
  return _mm512_mask_fmsubadd_round_ph(__A, __U, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_maskz_fmsubadd_round_ph(__mmask32 __U, __m512h __A, __m512h __B, __m512h __C) {
  // CHECK-LABEL: test_mm512_maskz_fmsubadd_round_ph
  // CHECK: fneg
  // CHECK: @llvm.x86.avx512fp16.vfmaddsub.ph.512
  // CHECK: bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> zeroinitializer
  return _mm512_maskz_fmsubadd_round_ph(__U, __A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_fmaddsub_ph(__m512h __A, __m512h __B, __m512h __C) {
  // CHECK-LABEL: test_mm512_fmaddsub_ph
  // CHECK-NOT: fneg
  // CHECK: @llvm.x86.avx512fp16.vfmaddsub.ph.512(<32 x half> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}, i32 4)
  return _mm512_fmaddsub_ph(__A, __B, __C);
}

__m512h test_mm512_mask_fmaddsub_ph(__m512h __A, __mmask32 __U, __m512h __B, __m512h __C) {
  // CHECK-LABEL: test_mm512_mask_fmaddsub_ph
  // CHECK-NOT: fneg
  // CHECK: @llvm.x86.avx512fp16.vfmaddsub.ph.512(<32 x half> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}, i32 4)
  // CHECK: bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}
  return _mm512_mask_fmaddsub_ph(__A, __U, __B, __C);
}

__m512h test_mm512_mask3_fmaddsub_ph(__m512h __A, __m512h __B, __m512h __C, __mmask32 __U) {
  // CHECK-LABEL: test_mm512_mask3_fmaddsub_ph
  // CHECK-NOT: fneg
  // CHECK: @llvm.x86.avx512fp16.vfmaddsub.ph.512(<32 x half> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}, i32 4)
  // CHECK: bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}
  return _mm512_mask3_fmaddsub_ph(__A, __B, __C, __U);
}

__m512h test_mm512_maskz_fmaddsub_ph(__mmask32 __U, __m512h __A, __m512h __B, __m512h __C) {
  // CHECK-LABEL: test_mm512_maskz_fmaddsub_ph
  // CHECK-NOT: fneg
  // CHECK: @llvm.x86.avx512fp16.vfmaddsub.ph.512(<32 x half> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}, i32 4)
  // CHECK: bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> zeroinitializer
  return _mm512_maskz_fmaddsub_ph(__U, __A, __B, __C);
}

__m512h test_mm512_fmsubadd_ph(__m512h __A, __m512h __B, __m512h __C) {
  // CHECK-LABEL: test_mm512_fmsubadd_ph
  // CHECK: [[NEG:%.+]] = fneg
  // CHECK: @llvm.x86.avx512fp16.vfmaddsub.ph.512(<32 x half> %{{.*}}, <32 x half> %{{.*}}, <32 x half> [[NEG]], i32 4)
  return _mm512_fmsubadd_ph(__A, __B, __C);
}

__m512h test_mm512_mask_fmsubadd_ph(__m512h __A, __mmask32 __U, __m512h __B, __m512h __C) {
  // CHECK-LABEL: test_mm512_mask_fmsubadd_ph
  // CHECK: [[NEG:%.+]] = fneg
  // CHECK: @llvm.x86.avx512fp16.vfmaddsub.ph.512(<32 x half> %{{.*}}, <32 x half> %{{.*}}, <32 x half> [[NEG]], i32 4)
  // CHECK: bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}
  return _mm512_mask_fmsubadd_ph(__A, __U, __B, __C);
}

__m512h test_mm512_maskz_fmsubadd_ph(__mmask32 __U, __m512h __A, __m512h __B, __m512h __C) {
  // CHECK-LABEL: test_mm512_maskz_fmsubadd_ph
  // CHECK: [[NEG:%.+]] = fneg
  // CHECK: @llvm.x86.avx512fp16.vfmaddsub.ph.512(<32 x half> %{{.*}}, <32 x half> %{{.*}}, <32 x half> [[NEG]], i32 4)
  // CHECK: bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> zeroinitializer
  return _mm512_maskz_fmsubadd_ph(__U, __A, __B, __C);
}

__m512h test_mm512_mask3_fmsub_round_ph(__m512h __A, __m512h __B, __m512h __C, __mmask32 __U) {
  // CHECK-LABEL: test_mm512_mask3_fmsub_round_ph
  // CHECK: fneg
  // CHECK: @llvm.x86.avx512fp16.vfmadd.ph.512
  // CHECK: bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}
  return _mm512_mask3_fmsub_round_ph(__A, __B, __C, __U, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_mask3_fmsub_ph(__m512h __A, __m512h __B, __m512h __C, __mmask32 __U) {
  // CHECK-LABEL: test_mm512_mask3_fmsub_ph
  // CHECK: fneg
  // CHECK: @llvm.fma.v32f16(<32 x half> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}})
  // CHECK: bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}
  return _mm512_mask3_fmsub_ph(__A, __B, __C, __U);
}

__m512h test_mm512_mask3_fmsubadd_round_ph(__m512h __A, __m512h __B, __m512h __C, __mmask32 __U) {
  // CHECK-LABEL: test_mm512_mask3_fmsubadd_round_ph
  // CHECK: fneg
  // CHECK: @llvm.x86.avx512fp16.vfmaddsub.ph.512
  // CHECK: bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}
  return _mm512_mask3_fmsubadd_round_ph(__A, __B, __C, __U, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_mask3_fmsubadd_ph(__m512h __A, __m512h __B, __m512h __C, __mmask32 __U) {
  // CHECK-LABEL: test_mm512_mask3_fmsubadd_ph
  // CHECK: [[NEG:%.+]] = fneg
  // CHECK: @llvm.x86.avx512fp16.vfmaddsub.ph.512(<32 x half> %{{.*}}, <32 x half> %{{.*}}, <32 x half> [[NEG]], i32 4)
  // CHECK: bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}
  return _mm512_mask3_fmsubadd_ph(__A, __B, __C, __U);
}

__m512h test_mm512_mask_fnmadd_round_ph(__m512h __A, __mmask32 __U, __m512h __B, __m512h __C) {
  // CHECK-LABEL: test_mm512_mask_fnmadd_round_ph
  // CHECK: fneg
  // CHECK: @llvm.x86.avx512fp16.vfmadd.ph.512
  // CHECK: bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}
  return _mm512_mask_fnmadd_round_ph(__A, __U, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_mask_fnmadd_ph(__m512h __A, __mmask32 __U, __m512h __B, __m512h __C) {
  // CHECK-LABEL: test_mm512_mask_fnmadd_ph
  // CHECK: fneg
  // CHECK: @llvm.fma.v32f16(<32 x half> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}})
  // CHECK: bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}
  return _mm512_mask_fnmadd_ph(__A, __U, __B, __C);
}

__m512h test_mm512_mask_fnmsub_round_ph(__m512h __A, __mmask32 __U, __m512h __B, __m512h __C) {
  // CHECK-LABEL: test_mm512_mask_fnmsub_round_ph
  // CHECK: fneg
  // CHECK: fneg
  // CHECK: @llvm.x86.avx512fp16.vfmadd.ph.512
  // CHECK: bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}
  return _mm512_mask_fnmsub_round_ph(__A, __U, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_mask3_fnmsub_round_ph(__m512h __A, __m512h __B, __m512h __C, __mmask32 __U) {
  // CHECK-LABEL: test_mm512_mask3_fnmsub_round_ph
  // CHECK: fneg
  // CHECK: fneg
  // CHECK: @llvm.x86.avx512fp16.vfmadd.ph.512
  // CHECK: bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}
  return _mm512_mask3_fnmsub_round_ph(__A, __B, __C, __U, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_mask_fnmsub_ph(__m512h __A, __mmask32 __U, __m512h __B, __m512h __C) {
  // CHECK-LABEL: test_mm512_mask_fnmsub_ph
  // CHECK: fneg
  // CHECK: fneg
  // CHECK: @llvm.fma.v32f16(<32 x half> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}})
  // CHECK: bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}
  return _mm512_mask_fnmsub_ph(__A, __U, __B, __C);
}

__m512h test_mm512_mask3_fnmsub_ph(__m512h __A, __m512h __B, __m512h __C, __mmask32 __U) {
  // CHECK-LABEL: test_mm512_mask3_fnmsub_ph
  // CHECK: fneg
  // CHECK: fneg
  // CHECK: @llvm.fma.v32f16(<32 x half> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}})
  // CHECK: bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}
  return _mm512_mask3_fnmsub_ph(__A, __B, __C, __U);
}

__m128h test_mm_fmadd_sh(__m128h __W, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_fmadd_sh
  // CHECK: [[A:%.+]] = extractelement <8 x half> [[ORIGA:%.+]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call half @llvm.fma.f16(half [[A]], half [[B]], half [[C]])
  return _mm_fmadd_sh(__W, __A, __B);
}

__m128h test_mm_mask_fmadd_sh(__m128h __W, __mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_mask_fmadd_sh
  // CHECK: [[A:%.+]] = extractelement <8 x half> [[ORIGA:%.+]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call half @llvm.fma.f16(half [[A]], half [[B]], half [[C]])
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, half [[FMA]], half [[A]]
  // CHECK-NEXT: insertelement <8 x half> [[ORIGA]], half [[SEL]], i64 0
  return _mm_mask_fmadd_sh(__W, __U, __A, __B);
}

__m128h test_mm_fmadd_round_sh(__m128h __A, __m128h __B, __m128h __C) {
  // CHECK-LABEL: test_mm_fmadd_round_sh
  // CHECK: [[A:%.+]] = extractelement <8 x half> [[ORIGA:%.+]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call half @llvm.x86.avx512fp16.vfmadd.f16(half [[A]], half [[B]], half [[C]], i32 11)
  // CHECK-NEXT: insertelement <8 x half> [[ORIGA]], half [[FMA]], i64 0
  return _mm_fmadd_round_sh(__A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm_mask_fmadd_round_sh(__m128h __W, __mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_mask_fmadd_round_sh
  // CHECK: [[A:%.+]] = extractelement <8 x half> [[ORIGA:%.+]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call half @llvm.x86.avx512fp16.vfmadd.f16(half [[A]], half [[B]], half [[C]], i32 11)
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, half [[FMA]], half [[A]]
  // CHECK-NEXT: insertelement <8 x half> [[ORIGA]], half [[SEL]], i64 0
  return _mm_mask_fmadd_round_sh(__W, __U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm_maskz_fmadd_sh(__mmask8 __U, __m128h __A, __m128h __B, __m128h __C) {
  // CHECK-LABEL: test_mm_maskz_fmadd_sh
  // CHECK: [[A:%.+]] = extractelement <8 x half> [[ORIGA:%.+]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call half @llvm.fma.f16(half [[A]], half [[B]], half [[C]])
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, half [[FMA]], half 0xH0000
  // CHECK-NEXT: insertelement <8 x half> [[ORIGA]], half [[SEL]], i64 0
  return _mm_maskz_fmadd_sh(__U, __A, __B, __C);
}

__m128h test_mm_maskz_fmadd_round_sh(__mmask8 __U, __m128h __A, __m128h __B, __m128h __C) {
  // CHECK-LABEL: test_mm_maskz_fmadd_round_sh
  // CHECK: [[A:%.+]] = extractelement <8 x half> [[ORIGA:%.+]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call half @llvm.x86.avx512fp16.vfmadd.f16(half [[A]], half [[B]], half [[C]], i32 11)
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, half [[FMA]], half 0xH0000
  // CHECK-NEXT: insertelement <8 x half> [[ORIGA]], half [[SEL]], i64 0
  return _mm_maskz_fmadd_round_sh(__U, __A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm_mask3_fmadd_sh(__m128h __W, __m128h __X, __m128h __Y, __mmask8 __U) {
  // CHECK-LABEL: test_mm_mask3_fmadd_sh
  // CHECK: [[A:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <8 x half> [[ORIGC:%.+]], i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call half @llvm.fma.f16(half [[A]], half [[B]], half [[C]])
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, half [[FMA]], half [[C]]
  // CHECK-NEXT: insertelement <8 x half> [[ORIGC]], half [[SEL]], i64 0
  return _mm_mask3_fmadd_sh(__W, __X, __Y, __U);
}

__m128h test_mm_mask3_fmadd_round_sh(__m128h __W, __m128h __X, __m128h __Y, __mmask8 __U) {
  // CHECK-LABEL: test_mm_mask3_fmadd_round_sh
  // CHECK: [[A:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <8 x half> [[ORIGC:%.+]], i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call half @llvm.x86.avx512fp16.vfmadd.f16(half [[A]], half [[B]], half [[C]], i32 11)
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, half [[FMA]], half [[C]]
  // CHECK-NEXT: insertelement <8 x half> [[ORIGC]], half [[SEL]], i64 0
  return _mm_mask3_fmadd_round_sh(__W, __X, __Y, __U, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm_fmsub_sh(__m128h __W, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_fmsub_sh
  // CHECK: %{{.*}} = fneg <8 x half> %{{.*}}
  // CHECK-NEXT: %{{.*}} = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: %{{.*}} = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: %{{.*}} = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: %{{.*}} = call half @llvm.fma.f16(half %{{.*}}, half %{{.*}}, half %{{.*}})
  // CHECK-NEXT: %{{.*}} = insertelement <8 x half> %{{.*}}, half %{{.*}}, i64 0
  // CHECK-NEXT: ret <8 x half> %{{.*}}
  return _mm_fmsub_sh(__W, __A, __B);
}

__m128h test_mm_mask_fmsub_sh(__m128h __W, __mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_mask_fmsub_sh
  // CHECK: %{{.*}} = fneg <8 x half> %{{.*}}
  // CHECK: %{{.*}} = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: %{{.*}} = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: %{{.*}} = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: %{{.*}} = call half @llvm.fma.f16(half %{{.*}}, half %{{.*}}, half %{{.*}})
  // CHECK-NEXT: %{{.*}} = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: %{{.*}} = extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: %{{.*}} = select i1 %{{.*}}, half %{{.*}}, half %{{.*}}
  // CHECK-NEXT: %{{.*}} = insertelement <8 x half> %{{.*}}, half %{{.*}}, i64 0
  // CHECK-NEXT: ret <8 x half> %{{.*}}
  return _mm_mask_fmsub_sh(__W, __U, __A, __B);
}

__m128h test_mm_fmsub_round_sh(__m128h __A, __m128h __B, __m128h __C) {
  // CHECK-LABEL: test_mm_fmsub_round_sh
  // CHECK: %{{.*}} = fneg <8 x half> %{{.*}}
  // CHECK-NEXT: %{{.*}} = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: %{{.*}} = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: %{{.*}} = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: %{{.*}} = call half @llvm.x86.avx512fp16.vfmadd.f16(half %{{.*}}, half %{{.*}}, half %{{.*}}, i32 11)
  // CHECK-NEXT: %{{.*}} = insertelement <8 x half> %{{.*}}, half %{{.*}}, i64 0
  // CHECK-NEXT: ret <8 x half> %{{.*}}
  return _mm_fmsub_round_sh(__A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm_mask_fmsub_round_sh(__m128h __W, __mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_mask_fmsub_round_sh
  // CHECK: %{{.*}} = fneg <8 x half> %{{.*}}
  // CHECK: %{{.*}} = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: %{{.*}} = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: %{{.*}} = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: %{{.*}} = call half @llvm.x86.avx512fp16.vfmadd.f16(half %{{.*}}, half %{{.*}}, half %{{.*}}, i32 11)
  // CHECK-NEXT: %{{.*}} = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: %{{.*}} = extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: %{{.*}} = select i1 %{{.*}}, half %{{.*}}, half %{{.*}}
  // CHECK-NEXT: %{{.*}} = insertelement <8 x half> %{{.*}}, half %{{.*}}, i64 0
  // CHECK-NEXT: ret <8 x half> %{{.*}}
  return _mm_mask_fmsub_round_sh(__W, __U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm_maskz_fmsub_sh(__mmask8 __U, __m128h __A, __m128h __B, __m128h __C) {
  // CHECK-LABEL: test_mm_maskz_fmsub_sh
  // CHECK: %{{.*}} = fneg <8 x half> %{{.*}}
  // CHECK: %{{.*}} = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: %{{.*}} = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: %{{.*}} = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: %{{.*}} = call half @llvm.fma.f16(half %{{.*}}, half %{{.*}}, half %{{.*}})
  // CHECK-NEXT: %{{.*}} = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: %{{.*}} = extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: %{{.*}} = select i1 %{{.*}}, half %{{.*}}, half 0xH0000
  // CHECK-NEXT: %{{.*}} = insertelement <8 x half> %{{.*}}, half %{{.*}}, i64 0
  // CHECK-NEXT: ret <8 x half> %{{.*}}
  return _mm_maskz_fmsub_sh(__U, __A, __B, __C);
}

__m128h test_mm_maskz_fmsub_round_sh(__mmask8 __U, __m128h __A, __m128h __B, __m128h __C) {
  // CHECK-LABEL: test_mm_maskz_fmsub_round_sh
  // CHECK: %{{.*}} = fneg <8 x half> %{{.*}}
  // CHECK: %{{.*}} = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: %{{.*}} = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: %{{.*}} = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: %{{.*}} = call half @llvm.x86.avx512fp16.vfmadd.f16(half %{{.*}}, half %{{.*}}, half %{{.*}}, i32 11)
  // CHECK-NEXT: %{{.*}} = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: %{{.*}} = extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: %{{.*}} = select i1 %{{.*}}, half %{{.*}}, half 0xH0000
  // CHECK-NEXT: %{{.*}} = insertelement <8 x half> %{{.*}}, half %{{.*}}, i64 0
  // CHECK-NEXT: ret <8 x half> %{{.*}}
  return _mm_maskz_fmsub_round_sh(__U, __A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm_mask3_fmsub_sh(__m128h __W, __m128h __X, __m128h __Y, __mmask8 __U) {
  // CHECK-LABEL: test_mm_mask3_fmsub_sh
  // CHECK: [[NEG:%.+]] = fneg
  // CHECK: [[A:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call half @llvm.fma.f16(half [[A]], half [[B]], half [[C]])
  // CHECK-NEXT: [[C2:%.+]] = extractelement <8 x half> [[ORIGC:%.+]], i64 0
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, half [[FMA]], half [[C2]]
  // CHECK-NEXT: insertelement <8 x half> [[ORIGC]], half [[SEL]], i64 0
  return _mm_mask3_fmsub_sh(__W, __X, __Y, __U);
}

__m128h test_mm_mask3_fmsub_round_sh(__m128h __W, __m128h __X, __m128h __Y, __mmask8 __U) {
  // CHECK-LABEL: test_mm_mask3_fmsub_round_sh
  // CHECK: [[NEG:%.+]] = fneg
  // CHECK: [[A:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call half @llvm.x86.avx512fp16.vfmadd.f16(half [[A]], half [[B]], half [[C]], i32 11)
  // CHECK-NEXT: [[C2:%.+]] = extractelement <8 x half> [[ORIGC:%.+]], i64 0
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, half [[FMA]], half [[C2]]
  // CHECK-NEXT: insertelement <8 x half> [[ORIGC]], half [[SEL]], i64 0
  return _mm_mask3_fmsub_round_sh(__W, __X, __Y, __U, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm_fnmadd_sh(__m128h __W, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_fnmadd_sh
  // CHECK: [[NEG:%.+]] = fneg
  // CHECK: [[A:%.+]] = extractelement <8 x half> [[ORIGA:%.+]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call half @llvm.fma.f16(half [[A]], half [[B]], half [[C]])
  return _mm_fnmadd_sh(__W, __A, __B);
}

__m128h test_mm_mask_fnmadd_sh(__m128h __W, __mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_mask_fnmadd_sh
  // CHECK: [[NEG:%.+]] = fneg
  // CHECK: [[A:%.+]] = extractelement <8 x half> [[ORIGA:%.+]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call half @llvm.fma.f16(half [[A]], half [[B]], half [[C]])
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, half [[FMA]], half [[A]]
  // CHECK-NEXT: insertelement <8 x half> [[ORIGA]], half [[SEL]], i64 0
  return _mm_mask_fnmadd_sh(__W, __U, __A, __B);
}

__m128h test_mm_fnmadd_round_sh(__m128h __A, __m128h __B, __m128h __C) {
  // CHECK-LABEL: test_mm_fnmadd_round_sh
  // CHECK: [[NEG:%.+]] = fneg
  // CHECK: [[A:%.+]] = extractelement <8 x half> [[ORIGA:%.+]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call half @llvm.x86.avx512fp16.vfmadd.f16(half [[A]], half [[B]], half [[C]], i32 11)
  // CHECK-NEXT: insertelement <8 x half> [[ORIGA]], half [[FMA]], i64 0
  return _mm_fnmadd_round_sh(__A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm_mask_fnmadd_round_sh(__m128h __W, __mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_mask_fnmadd_round_sh
  // CHECK: [[NEG:%.+]] = fneg
  // CHECK: [[A:%.+]] = extractelement <8 x half> [[ORIGA:%.+]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call half @llvm.x86.avx512fp16.vfmadd.f16(half [[A]], half [[B]], half [[C]], i32 11)
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, half [[FMA]], half [[A]]
  // CHECK-NEXT: insertelement <8 x half> [[ORIGA]], half [[SEL]], i64 0
  return _mm_mask_fnmadd_round_sh(__W, __U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm_maskz_fnmadd_sh(__mmask8 __U, __m128h __A, __m128h __B, __m128h __C) {
  // CHECK-LABEL: test_mm_maskz_fnmadd_sh
  // CHECK: [[NEG:%.+]] = fneg
  // CHECK: [[A:%.+]] = extractelement <8 x half> [[ORIGA:%.+]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call half @llvm.fma.f16(half [[A]], half [[B]], half [[C]])
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, half [[FMA]], half 0xH0000
  // CHECK-NEXT: insertelement <8 x half> [[ORIGA]], half [[SEL]], i64 0
  return _mm_maskz_fnmadd_sh(__U, __A, __B, __C);
}

__m128h test_mm_maskz_fnmadd_round_sh(__mmask8 __U, __m128h __A, __m128h __B, __m128h __C) {
  // CHECK-LABEL: test_mm_maskz_fnmadd_round_sh
  // CHECK: [[NEG:%.+]] = fneg
  // CHECK: [[A:%.+]] = extractelement <8 x half> [[ORIGA:%.+]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call half @llvm.x86.avx512fp16.vfmadd.f16(half [[A]], half [[B]], half [[C]], i32 11)
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, half [[FMA]], half 0xH0000
  // CHECK-NEXT: insertelement <8 x half> [[ORIGA]], half [[SEL]], i64 0
  return _mm_maskz_fnmadd_round_sh(__U, __A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm_mask3_fnmadd_sh(__m128h __W, __m128h __X, __m128h __Y, __mmask8 __U) {
  // CHECK-LABEL: test_mm_mask3_fnmadd_sh
  // CHECK: [[NEG:%.+]] = fneg
  // CHECK: [[A:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <8 x half> [[ORIGC:%.+]], i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call half @llvm.fma.f16(half [[A]], half [[B]], half [[C]])
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, half [[FMA]], half [[C]]
  // CHECK-NEXT: insertelement <8 x half> [[ORIGC]], half [[SEL]], i64 0
  return _mm_mask3_fnmadd_sh(__W, __X, __Y, __U);
}

__m128h test_mm_mask3_fnmadd_round_sh(__m128h __W, __m128h __X, __m128h __Y, __mmask8 __U) {
  // CHECK-LABEL: test_mm_mask3_fnmadd_round_sh
  // CHECK: [[NEG:%.+]] = fneg
  // CHECK: [[A:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <8 x half> [[ORIGC:%.+]], i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call half @llvm.x86.avx512fp16.vfmadd.f16(half [[A]], half [[B]], half [[C]], i32 11)
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, half [[FMA]], half [[C]]
  // CHECK-NEXT: insertelement <8 x half> [[ORIGC]], half [[SEL]], i64 0
  return _mm_mask3_fnmadd_round_sh(__W, __X, __Y, __U, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm_fnmsub_sh(__m128h __W, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_fnmsub_sh
  // CHECK: [[NEG:%.+]] = fneg
  // CHECK: [[NEG2:%.+]] = fneg
  // CHECK: [[A:%.+]] = extractelement <8 x half> [[ORIGA:%.+]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call half @llvm.fma.f16(half [[A]], half [[B]], half [[C]])
  return _mm_fnmsub_sh(__W, __A, __B);
}

__m128h test_mm_mask_fnmsub_sh(__m128h __W, __mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_mask_fnmsub_sh
  // CHECK: [[NEG:%.+]] = fneg
  // CHECK: [[NEG2:%.+]] = fneg
  // CHECK: [[A:%.+]] = extractelement <8 x half> [[ORIGA:%.+]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call half @llvm.fma.f16(half [[A]], half [[B]], half [[C]])
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, half [[FMA]], half [[A]]
  // CHECK-NEXT: insertelement <8 x half> [[ORIGA]], half [[SEL]], i64 0
  return _mm_mask_fnmsub_sh(__W, __U, __A, __B);
}

__m128h test_mm_fnmsub_round_sh(__m128h __A, __m128h __B, __m128h __C) {
  // CHECK-LABEL: test_mm_fnmsub_round_sh
  // CHECK: [[NEG:%.+]] = fneg
  // CHECK: [[NEG2:%.+]] = fneg
  // CHECK: [[A:%.+]] = extractelement <8 x half> [[ORIGA:%.+]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call half @llvm.x86.avx512fp16.vfmadd.f16(half [[A]], half [[B]], half [[C]], i32 11)
  // CHECK-NEXT: insertelement <8 x half> [[ORIGA]], half [[FMA]], i64 0
  return _mm_fnmsub_round_sh(__A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm_mask_fnmsub_round_sh(__m128h __W, __mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_mask_fnmsub_round_sh
  // CHECK: [[NEG:%.+]] = fneg
  // CHECK: [[NEG2:%.+]] = fneg
  // CHECK: [[A:%.+]] = extractelement <8 x half> [[ORIGA:%.+]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call half @llvm.x86.avx512fp16.vfmadd.f16(half [[A]], half [[B]], half [[C]], i32 11)
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, half [[FMA]], half [[A]]
  // CHECK-NEXT: insertelement <8 x half> [[ORIGA]], half [[SEL]], i64 0
  return _mm_mask_fnmsub_round_sh(__W, __U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm_maskz_fnmsub_sh(__mmask8 __U, __m128h __A, __m128h __B, __m128h __C) {
  // CHECK-LABEL: test_mm_maskz_fnmsub_sh
  // CHECK: [[NEG:%.+]] = fneg
  // CHECK: [[NEG2:%.+]] = fneg
  // CHECK: [[A:%.+]] = extractelement <8 x half> [[ORIGA:%.+]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call half @llvm.fma.f16(half [[A]], half [[B]], half [[C]])
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, half [[FMA]], half 0xH0000
  // CHECK-NEXT: insertelement <8 x half> [[ORIGA]], half [[SEL]], i64 0
  return _mm_maskz_fnmsub_sh(__U, __A, __B, __C);
}

__m128h test_mm_maskz_fnmsub_round_sh(__mmask8 __U, __m128h __A, __m128h __B, __m128h __C) {
  // CHECK-LABEL: test_mm_maskz_fnmsub_round_sh
  // CHECK: [[NEG:%.+]] = fneg
  // CHECK: [[NEG2:%.+]] = fneg
  // CHECK: [[A:%.+]] = extractelement <8 x half> [[ORIGA:%.+]], i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call half @llvm.x86.avx512fp16.vfmadd.f16(half [[A]], half [[B]], half [[C]], i32 11)
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, half [[FMA]], half 0xH0000
  // CHECK-NEXT: insertelement <8 x half> [[ORIGA]], half [[SEL]], i64 0
  return _mm_maskz_fnmsub_round_sh(__U, __A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm_mask3_fnmsub_sh(__m128h __W, __m128h __X, __m128h __Y, __mmask8 __U) {
  // CHECK-LABEL: test_mm_mask3_fnmsub_sh
  // CHECK: [[NEG:%.+]] = fneg
  // CHECK: [[NEG2:%.+]] = fneg
  // CHECK: [[A:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call half @llvm.fma.f16(half [[A]], half [[B]], half [[C]])
  // CHECK-NEXT: [[C2:%.+]] = extractelement <8 x half> [[ORIGC:%.+]], i64 0
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, half [[FMA]], half [[C2]]
  // CHECK-NEXT: insertelement <8 x half> [[ORIGC]], half [[SEL]], i64 0
  return _mm_mask3_fnmsub_sh(__W, __X, __Y, __U);
}

__m128h test_mm_mask3_fnmsub_round_sh(__m128h __W, __m128h __X, __m128h __Y, __mmask8 __U) {
  // CHECK-LABEL: test_mm_mask3_fnmsub_round_sh
  // CHECK: [[NEG:%.+]] = fneg
  // CHECK: [[NEG2:%.+]] = fneg
  // CHECK: [[A:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[B:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[C:%.+]] = extractelement <8 x half> %{{.*}}, i64 0
  // CHECK-NEXT: [[FMA:%.+]] = call half @llvm.x86.avx512fp16.vfmadd.f16(half [[A]], half [[B]], half [[C]], i32 11)
  // CHECK-NEXT: [[C2:%.+]] = extractelement <8 x half> [[ORIGC:%.+]], i64 0
  // CHECK-NEXT: bitcast i8 %{{.*}} to <8 x i1>
  // CHECK-NEXT: extractelement <8 x i1> %{{.*}}, i64 0
  // CHECK-NEXT: [[SEL:%.+]] = select i1 %{{.*}}, half [[FMA]], half [[C2]]
  // CHECK-NEXT: insertelement <8 x half> [[ORIGC]], half [[SEL]], i64 0
  return _mm_mask3_fnmsub_round_sh(__W, __X, __Y, __U, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm_fcmadd_sch(__m128h __A, __m128h __B, __m128h __C) {
  // CHECK-LABEL: test_mm_fcmadd_sch
  // CHECK: @llvm.x86.avx512fp16.mask.vfcmadd.csh
  return _mm_fcmadd_sch(__A, __B, __C);
}

__m128h test_mm_mask_fcmadd_sch(__m128h __A, __mmask8 __U, __m128h __B, __m128h __C) {
  // CHECK-LABEL: test_mm_mask_fcmadd_sch
  // CHECK: @llvm.x86.avx512fp16.mask.vfcmadd.csh
  // CHECK:  %{{.*}} = shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK:  %{{.*}} = select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_mask_fcmadd_sch(__A, __U, __B, __C);
}

__m128h test_mm_maskz_fcmadd_sch(__mmask8 __U, __m128h __A, __m128h __B, __m128h __C) {
  // CHECK-LABEL: test_mm_maskz_fcmadd_sch
  // CHECK: @llvm.x86.avx512fp16.maskz.vfcmadd.csh
  return _mm_maskz_fcmadd_sch(__U, __A, __B, __C);
}

__m128h test_mm_mask3_fcmadd_sch(__m128h __A, __m128h __B, __m128h __C, __mmask8 __U) {
  // CHECK-LABEL: test_mm_mask3_fcmadd_sch
  // CHECK:  %{{.*}} = bitcast <8 x half> %{{.*}} to <4 x float>
  // CHECK:  %{{.*}} = bitcast <8 x half> %{{.*}} to <4 x float>
  // CHECK:  %{{.*}} = bitcast <8 x half> %{{.*}} to <4 x float>
  // CHECK:  %{{.*}} = call <4 x float> @llvm.x86.avx512fp16.mask.vfcmadd.csh(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 %{{.*}}, i32 4)
  // CHECK:  %{{.*}} = shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 0, i32 5, i32 6, i32 7>
  // CHECK:  %{{.*}} = bitcast <4 x float> %{{.*}} to <8 x half>
  return _mm_mask3_fcmadd_sch(__A, __B, __C, __U);
}

__m128h test_mm_fcmadd_round_sch(__m128h __A, __m128h __B, __m128h __C) {
  // CHECK-LABEL: test_mm_fcmadd_round_sch
  // CHECK: @llvm.x86.avx512fp16.mask.vfcmadd.csh
  return _mm_fcmadd_round_sch(__A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm_mask_fcmadd_round_sch(__m128h __A, __mmask8 __U, __m128h __B, __m128h __C) {
  // CHECK-LABEL: test_mm_mask_fcmadd_round_sch
  // CHECK: @llvm.x86.avx512fp16.mask.vfcmadd.csh
  // CHECK:  %{{.*}} = shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK:  %{{.*}} = select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_mask_fcmadd_round_sch(__A, __U, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm_maskz_fcmadd_round_sch(__mmask8 __U, __m128h __A, __m128h __B, __m128h __C) {
  // CHECK-LABEL: test_mm_maskz_fcmadd_round_sch
  // CHECK: @llvm.x86.avx512fp16.maskz.vfcmadd.csh
  return _mm_maskz_fcmadd_round_sch(__U, __A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm_mask3_fcmadd_round_sch(__m128h __A, __m128h __B, __m128h __C, __mmask8 __U) {
  // CHECK-LABEL: test_mm_mask3_fcmadd_round_sch
  // CHECK:  %{{.*}} = bitcast <8 x half> %{{.*}} to <4 x float>
  // CHECK:  %{{.*}} = bitcast <8 x half> %{{.*}} to <4 x float>
  // CHECK:  %{{.*}} = bitcast <8 x half> %{{.*}} to <4 x float>
  // CHECK:  %{{.*}} = call <4 x float> @llvm.x86.avx512fp16.mask.vfcmadd.csh(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 %{{.*}}, i32 11)
  // CHECK:  %{{.*}} = shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 0, i32 5, i32 6, i32 7>
  // CHECK:  %{{.*}} = bitcast <4 x float> %{{.*}} to <8 x half>
  return _mm_mask3_fcmadd_round_sch(__A, __B, __C, __U, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm_fmadd_sch(__m128h __A, __m128h __B, __m128h __C) {
  // CHECK-LABEL: test_mm_fmadd_sch
  // CHECK: @llvm.x86.avx512fp16.mask.vfmadd.csh
  return _mm_fmadd_sch(__A, __B, __C);
}

__m128h test_mm_mask_fmadd_sch(__m128h __A, __mmask8 __U, __m128h __B, __m128h __C) {
  // CHECK-LABEL: test_mm_mask_fmadd_sch
  // CHECK: @llvm.x86.avx512fp16.mask.vfmadd.csh
  // CHECK:  %{{.*}} = shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK:  %{{.*}} = select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_mask_fmadd_sch(__A, __U, __B, __C);
}

__m128h test_mm_maskz_fmadd_sch(__mmask8 __U, __m128h __A, __m128h __B, __m128h __C) {
  // CHECK-LABEL: test_mm_maskz_fmadd_sch
  // CHECK: @llvm.x86.avx512fp16.maskz.vfmadd.csh
  return _mm_maskz_fmadd_sch(__U, __A, __B, __C);
}

__m128h test_mm_mask3_fmadd_sch(__m128h __A, __m128h __B, __m128h __C, __mmask8 __U) {
  // CHECK-LABEL: test_mm_mask3_fmadd_sch
  // CHECK:  %{{.*}} = bitcast <8 x half> %{{.*}} to <4 x float>
  // CHECK:  %{{.*}} = bitcast <8 x half> %{{.*}} to <4 x float>
  // CHECK:  %{{.*}} = bitcast <8 x half> %{{.*}} to <4 x float>
  // CHECK:  %{{.*}} = call <4 x float> @llvm.x86.avx512fp16.mask.vfmadd.csh(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 %{{.*}}, i32 4)
  // CHECK:  %{{.*}} = shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 0, i32 5, i32 6, i32 7>
  // CHECK:  %{{.*}} = bitcast <4 x float> %{{.*}} to <8 x half>
  return _mm_mask3_fmadd_sch(__A, __B, __C, __U);
}

__m128h test_mm_fmadd_round_sch(__m128h __A, __m128h __B, __m128h __C) {
  // CHECK-LABEL: test_mm_fmadd_round_sch
  // CHECK: @llvm.x86.avx512fp16.mask.vfmadd.csh
  return _mm_fmadd_round_sch(__A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm_mask_fmadd_round_sch(__m128h __A, __mmask8 __U, __m128h __B, __m128h __C) {
  // CHECK-LABEL: test_mm_mask_fmadd_round_sch
  // CHECK: @llvm.x86.avx512fp16.mask.vfmadd.csh
  // CHECK:  %{{.*}} = shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK:  %{{.*}} = select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_mask_fmadd_round_sch(__A, __U, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm_maskz_fmadd_round_sch(__mmask8 __U, __m128h __A, __m128h __B, __m128h __C) {
  // CHECK-LABEL: test_mm_maskz_fmadd_round_sch
  // CHECK: @llvm.x86.avx512fp16.maskz.vfmadd.csh
  return _mm_maskz_fmadd_round_sch(__U, __A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm_mask3_fmadd_round_sch(__m128h __A, __m128h __B, __m128h __C, __mmask8 __U) {
  // CHECK-LABEL: test_mm_mask3_fmadd_round_sch
  // CHECK:  %{{.*}} = bitcast <8 x half> %{{.*}} to <4 x float>
  // CHECK:  %{{.*}} = bitcast <8 x half> %{{.*}} to <4 x float>
  // CHECK:  %{{.*}} = bitcast <8 x half> %{{.*}} to <4 x float>
  // CHECK:  %{{.*}} = call <4 x float> @llvm.x86.avx512fp16.mask.vfmadd.csh(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 %{{.*}}, i32 11)
  // CHECK:  %{{.*}} = shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 0, i32 5, i32 6, i32 7>
  // CHECK:  %{{.*}} = bitcast <4 x float> %{{.*}} to <8 x half>
  return _mm_mask3_fmadd_round_sch(__A, __B, __C, __U, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm_fcmul_sch(__m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_fcmul_sch
  // CHECK: @llvm.x86.avx512fp16.mask.vfcmul.csh
  return _mm_fcmul_sch(__A, __B);
}

__m128h test_mm_mask_fcmul_sch(__m128h __W, __mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_mask_fcmul_sch
  // CHECK: @llvm.x86.avx512fp16.mask.vfcmul.csh
  return _mm_mask_fcmul_sch(__W, __U, __A, __B);
}

__m128h test_mm_maskz_fcmul_sch(__mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_maskz_fcmul_sch
  // CHECK: @llvm.x86.avx512fp16.mask.vfcmul.csh
  return _mm_maskz_fcmul_sch(__U, __A, __B);
}

__m128h test_mm_fcmul_round_sch(__m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_fcmul_round_sch
  // CHECK: @llvm.x86.avx512fp16.mask.vfcmul.csh
  return _mm_fcmul_round_sch(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm_mask_fcmul_round_sch(__m128h __W, __mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_mask_fcmul_round_sch
  // CHECK: @llvm.x86.avx512fp16.mask.vfcmul.csh
  return _mm_mask_fcmul_round_sch(__W, __U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm_maskz_fcmul_round_sch(__mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_maskz_fcmul_round_sch
  // CHECK: @llvm.x86.avx512fp16.mask.vfcmul.csh
  return _mm_maskz_fcmul_round_sch(__U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_fcmul_pch(__m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_fcmul_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfcmul.cph.512
  return _mm512_fcmul_pch(__A, __B);
}

__m512h test_mm512_mask_fcmul_pch(__m512h __W, __mmask16 __U, __m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_mask_fcmul_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfcmul.cph.512
  return _mm512_mask_fcmul_pch(__W, __U, __A, __B);
}

__m512h test_mm512_maskz_fcmul_pch(__mmask16 __U, __m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_maskz_fcmul_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfcmul.cph.512
  return _mm512_maskz_fcmul_pch(__U, __A, __B);
}

__m512h test_mm512_fcmul_round_pch(__m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_fcmul_round_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfcmul.cph.512
  return _mm512_fcmul_round_pch(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_mask_fcmul_round_pch(__m512h __W, __mmask16 __U, __m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_mask_fcmul_round_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfcmul.cph.512
  return _mm512_mask_fcmul_round_pch(__W, __U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_maskz_fcmul_round_pch(__mmask16 __U, __m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_maskz_fcmul_round_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfcmul.cph.512
  return _mm512_maskz_fcmul_round_pch(__U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_fcmadd_pch(__m512h __A, __m512h __B, __m512h __C) {
  // CHECK-LABEL: test_mm512_fcmadd_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfcmadd.cph.512
  return _mm512_fcmadd_pch(__A, __B, __C);
}

__m512h test_mm512_mask_fcmadd_pch(__m512h __A, __mmask16 __U, __m512h __B, __m512h __C) {
  // CHECK-LABEL: test_mm512_mask_fcmadd_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfcmadd.cph.512
  // CHECK:  %{{.*}} = select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask_fcmadd_pch(__A, __U, __B, __C);
}

__m512h test_mm512_mask3_fcmadd_pch(__m512h __A, __m512h __B, __m512h __C, __mmask16 __U) {
  // CHECK-LABEL: test_mm512_mask3_fcmadd_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfcmadd.cph.512
  // CHECK-NOT:  %{{.*}} = select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask3_fcmadd_pch(__A, __B, __C, __U);
}

__m512h test_mm512_maskz_fcmadd_pch(__mmask16 __U, __m512h __A, __m512h __B, __m512h __C) {
  // CHECK-LABEL: test_mm512_maskz_fcmadd_pch
  // CHECK: @llvm.x86.avx512fp16.maskz.vfcmadd.cph.512
  return _mm512_maskz_fcmadd_pch(__U, __A, __B, __C);
}

__m512h test_mm512_fcmadd_round_pch(__m512h __A, __m512h __B, __m512h __C) {
  // CHECK-LABEL: test_mm512_fcmadd_round_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfcmadd.cph.512
  return _mm512_fcmadd_round_pch(__A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_mask_fcmadd_round_pch(__m512h __A, __mmask16 __U, __m512h __B, __m512h __C) {
  // CHECK-LABEL: test_mm512_mask_fcmadd_round_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfcmadd.cph.512
  // CHECK:  %{{.*}} = select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask_fcmadd_round_pch(__A, __U, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_mask3_fcmadd_round_pch(__m512h __A, __m512h __B, __m512h __C, __mmask16 __U) {
  // CHECK-LABEL: test_mm512_mask3_fcmadd_round_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfcmadd.cph.512
  // CHECK-NOT:  %{{.*}} = select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask3_fcmadd_round_pch(__A, __B, __C, __U, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_maskz_fcmadd_round_pch(__mmask16 __U, __m512h __A, __m512h __B, __m512h __C) {
  // CHECK-LABEL: test_mm512_maskz_fcmadd_round_pch
  // CHECK: @llvm.x86.avx512fp16.maskz.vfcmadd.cph.512
  return _mm512_maskz_fcmadd_round_pch(__U, __A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_fmul_pch(__m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_fmul_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfmul.cph.512
  return _mm512_fmul_pch(__A, __B);
}

__m512h test_mm512_mask_fmul_pch(__m512h __W, __mmask16 __U, __m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_mask_fmul_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfmul.cph.512
  return _mm512_mask_fmul_pch(__W, __U, __A, __B);
}

__m512h test_mm512_maskz_fmul_pch(__mmask16 __U, __m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_maskz_fmul_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfmul.cph.512
  return _mm512_maskz_fmul_pch(__U, __A, __B);
}

__m512h test_mm512_fmul_round_pch(__m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_fmul_round_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfmul.cph.512
  return _mm512_fmul_round_pch(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_mask_fmul_round_pch(__m512h __W, __mmask16 __U, __m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_mask_fmul_round_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfmul.cph.512
  return _mm512_mask_fmul_round_pch(__W, __U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_maskz_fmul_round_pch(__mmask16 __U, __m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_maskz_fmul_round_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfmul.cph.512
  return _mm512_maskz_fmul_round_pch(__U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_fmadd_pch(__m512h __A, __m512h __B, __m512h __C) {
  // CHECK-LABEL: test_mm512_fmadd_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfmadd.cph.512
  return _mm512_fmadd_pch(__A, __B, __C);
}

__m512h test_mm512_mask_fmadd_pch(__m512h __A, __mmask16 __U, __m512h __B, __m512h __C) {
  // CHECK-LABEL: test_mm512_mask_fmadd_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfmadd.cph.512
  // CHECK:  %{{.*}} = select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask_fmadd_pch(__A, __U, __B, __C);
}

__m512h test_mm512_mask3_fmadd_pch(__m512h __A, __m512h __B, __m512h __C, __mmask16 __U) {
  // CHECK-LABEL: test_mm512_mask3_fmadd_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfmadd.cph.512
  // CHECK-NOT:  %{{.*}} = select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask3_fmadd_pch(__A, __B, __C, __U);
}

__m512h test_mm512_maskz_fmadd_pch(__mmask16 __U, __m512h __A, __m512h __B, __m512h __C) {
  // CHECK-LABEL: test_mm512_maskz_fmadd_pch
  // CHECK: @llvm.x86.avx512fp16.maskz.vfmadd.cph.512
  return _mm512_maskz_fmadd_pch(__U, __A, __B, __C);
}

__m512h test_mm512_fmadd_round_pch(__m512h __A, __m512h __B, __m512h __C) {
  // CHECK-LABEL: test_mm512_fmadd_round_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfmadd.cph.512
  return _mm512_fmadd_round_pch(__A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_mask_fmadd_round_pch(__m512h __A, __mmask16 __U, __m512h __B, __m512h __C) {
  // CHECK-LABEL: test_mm512_mask_fmadd_round_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfmadd.cph.512
  // CHECK:  %{{.*}} = select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask_fmadd_round_pch(__A, __U, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_mask3_fmadd_round_pch(__m512h __A, __m512h __B, __m512h __C, __mmask16 __U) {
  // CHECK-LABEL: test_mm512_mask3_fmadd_round_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfmadd.cph.512
  // CHECK-NOT:  %{{.*}} = select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
  return _mm512_mask3_fmadd_round_pch(__A, __B, __C, __U, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_maskz_fmadd_round_pch(__mmask16 __U, __m512h __A, __m512h __B, __m512h __C) {
  // CHECK-LABEL: test_mm512_maskz_fmadd_round_pch
  // CHECK: @llvm.x86.avx512fp16.maskz.vfmadd.cph.512
  return _mm512_maskz_fmadd_round_pch(__U, __A, __B, __C, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm_fmul_sch(__m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_fmul_sch
  // CHECK: @llvm.x86.avx512fp16.mask.vfmul.csh
  return _mm_fmul_sch(__A, __B);
}

__m128h test_mm_mask_fmul_sch(__m128h __W, __mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_mask_fmul_sch
  // CHECK: @llvm.x86.avx512fp16.mask.vfmul.csh
  return _mm_mask_fmul_sch(__W, __U, __A, __B);
}

__m128h test_mm_maskz_fmul_sch(__mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_maskz_fmul_sch
  // CHECK: @llvm.x86.avx512fp16.mask.vfmul.csh
  return _mm_maskz_fmul_sch(__U, __A, __B);
}

__m128h test_mm_fmul_round_sch(__m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_fmul_round_sch
  // CHECK: @llvm.x86.avx512fp16.mask.vfmul.csh
  return _mm_fmul_round_sch(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm_mask_fmul_round_sch(__m128h __W, __mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_mask_fmul_round_sch
  // CHECK: @llvm.x86.avx512fp16.mask.vfmul.csh
  return _mm_mask_fmul_round_sch(__W, __U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm_maskz_fmul_round_sch(__mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_maskz_fmul_round_sch
  // CHECK: @llvm.x86.avx512fp16.mask.vfmul.csh
  return _mm_maskz_fmul_round_sch(__U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

_Float16 test_mm512_reduce_add_ph(__m512h __W) {
  // CHECK-LABEL: test_mm512_reduce_add_ph
  // CHECK: @llvm.vector.reduce.fadd.v32f16(half 0xH8000, <32 x half> %{{.*}})
  return _mm512_reduce_add_ph(__W);
}

_Float16 test_mm512_reduce_mul_ph(__m512h __W) {
  // CHECK-LABEL: test_mm512_reduce_mul_ph
  // CHECK: @llvm.vector.reduce.fmul.v32f16(half 0xH3C00, <32 x half> %{{.*}})
  return _mm512_reduce_mul_ph(__W);
}

_Float16 test_mm512_reduce_max_ph(__m512h __W) {
  // CHECK-LABEL: test_mm512_reduce_max_ph
  // CHECK: @llvm.vector.reduce.fmax.v32f16(<32 x half> %{{.*}})
  return _mm512_reduce_max_ph(__W);
}

_Float16 test_mm512_reduce_min_ph(__m512h __W) {
  // CHECK-LABEL: test_mm512_reduce_min_ph
  // CHECK: @llvm.vector.reduce.fmin.v32f16(<32 x half> %{{.*}})
  return _mm512_reduce_min_ph(__W);
}

__m512h test_mm512_mask_blend_ph(__mmask32 __U, __m512h __A, __m512h __W) {
  // CHECK-LABEL: test_mm512_mask_blend_ph
  // CHECK:  %{{.*}} = bitcast i32 %{{.*}} to <32 x i1>
  // CHECK:  %{{.*}} = select <32 x i1> %{{.*}}, <32 x half> %{{.*}}, <32 x half> %{{.*}}
  return _mm512_mask_blend_ph(__U, __A, __W);
}

__m512h test_mm512_permutex2var_ph(__m512h __A, __m512i __I, __m512h __B) {
  // CHECK-LABEL: test_mm512_permutex2var_ph
  // CHECK:  %{{.*}} = bitcast <32 x half> %{{.*}} to <32 x i16>
  // CHECK:  %{{.*}} = bitcast <8 x i64> %{{.*}} to <32 x i16>
  // CHECK:  %{{.*}} = bitcast <32 x half> %{{.*}} to <32 x i16>
  // CHECK:  %{{.*}} = call <32 x i16> @llvm.x86.avx512.vpermi2var.hi.512(<32 x i16> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}})
  // CHECK:  %{{.*}} = bitcast <32 x i16> %{{.*}} to <32 x half>
  return _mm512_permutex2var_ph(__A, __I, __B);
}

__m512h test_mm512_permutexvar_epi16(__m512i __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_permutexvar_epi16
  // CHECK:  %{{.*}} = bitcast <32 x half> %{{.*}} to <32 x i16>
  // CHECK:  %{{.*}} = bitcast <8 x i64> %{{.*}} to <32 x i16>
  // CHECK:  %{{.*}} = call <32 x i16> @llvm.x86.avx512.permvar.hi.512(<32 x i16> %{{.*}}, <32 x i16> %{{.*}})
  // CHECK:  %{{.*}} = bitcast <32 x i16> %{{.*}} to <32 x half>
  return _mm512_permutexvar_ph(__A, __B);
}

// tests below are for alias intrinsics.
__m512h test_mm512_mul_pch(__m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_mul_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfmul.cph.512
  return _mm512_mul_pch(__A, __B);
}

__m512h test_mm512_mask_mul_pch(__m512h __W, __mmask16 __U, __m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_mask_mul_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfmul.cph.512
  return _mm512_mask_mul_pch(__W, __U, __A, __B);
}

__m512h test_mm512_maskz_mul_pch(__mmask16 __U, __m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_maskz_mul_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfmul.cph.512
  return _mm512_maskz_mul_pch(__U, __A, __B);
}

__m512h test_mm512_cmul_pch(__m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_cmul_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfcmul.cph.512
  return _mm512_cmul_pch(__A, __B);
}
__m512h test_mm512_mask_cmul_pch(__m512h __W, __mmask16 __U, __m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_mask_cmul_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfcmul.cph.512
  return _mm512_mask_cmul_pch(__W, __U, __A, __B);
}

__m512h test_mm512_maskz_cmul_pch(__mmask16 __U, __m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_maskz_cmul_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfcmul.cph.512
  return _mm512_maskz_cmul_pch(__U, __A, __B);
}

__m128h test_mm_mul_sch(__m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_mul_sch
  // CHECK: @llvm.x86.avx512fp16.mask.vfmul.csh
  return _mm_mul_sch(__A, __B);
}

__m128h test_mm_mask_mul_sch(__m128h __W, __mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_mask_mul_sch
  // CHECK: @llvm.x86.avx512fp16.mask.vfmul.csh
  return _mm_mask_mul_sch(__W, __U, __A, __B);
}

__m128h test_mm_maskz_mul_sch(__mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_maskz_mul_sch
  // CHECK: @llvm.x86.avx512fp16.mask.vfmul.csh
  return _mm_maskz_mul_sch(__U, __A, __B);
}

__m128h test_mm_mul_round_sch(__m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_mul_round_sch
  // CHECK: @llvm.x86.avx512fp16.mask.vfmul.csh
  return _mm_mul_round_sch(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm_mask_mul_round_sch(__m128h __W, __mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_mask_mul_round_sch
  // CHECK: @llvm.x86.avx512fp16.mask.vfmul.csh
  return _mm_mask_mul_round_sch(__W, __U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm_maskz_mul_round_sch(__mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_maskz_mul_round_sch
  // CHECK: @llvm.x86.avx512fp16.mask.vfmul.csh
  return _mm_maskz_mul_round_sch(__U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_mul_round_pch(__m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_mul_round_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfmul.cph.512
  return _mm512_mul_round_pch(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_mask_mul_round_pch(__m512h __W, __mmask16 __U, __m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_mask_mul_round_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfmul.cph.512
  return _mm512_mask_mul_round_pch(__W, __U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_maskz_mul_round_pch(__mmask16 __U, __m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_maskz_mul_round_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfmul.cph.512
  return _mm512_maskz_mul_round_pch(__U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_cmul_round_pch(__m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_cmul_round_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfcmul.cph.512
  return _mm512_cmul_round_pch(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_mask_cmul_round_pch(__m512h __W, __mmask16 __U, __m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_mask_cmul_round_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfcmul.cph.512
  return _mm512_mask_cmul_round_pch(__W, __U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m512h test_mm512_maskz_cmul_round_pch(__mmask16 __U, __m512h __A, __m512h __B) {
  // CHECK-LABEL: test_mm512_maskz_cmul_round_pch
  // CHECK: @llvm.x86.avx512fp16.mask.vfcmul.cph.512
  return _mm512_maskz_cmul_round_pch(__U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm_cmul_sch(__m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_cmul_sch
  // CHECK: @llvm.x86.avx512fp16.mask.vfcmul.csh
  return _mm_cmul_sch(__A, __B);
}

__m128h test_mm_mask_cmul_sch(__m128h __W, __mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_mask_cmul_sch
  // CHECK: @llvm.x86.avx512fp16.mask.vfcmul.csh
  return _mm_mask_cmul_sch(__W, __U, __A, __B);
}

__m128h test_mm_maskz_cmul_sch(__mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_maskz_cmul_sch
  // CHECK: @llvm.x86.avx512fp16.mask.vfcmul.csh
  return _mm_maskz_cmul_sch(__U, __A, __B);
}

__m128h test_mm_cmul_round_sch(__m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_cmul_round_sch
  // CHECK: @llvm.x86.avx512fp16.mask.vfcmul.csh
  return _mm_cmul_round_sch(__A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm_mask_cmul_round_sch(__m128h __W, __mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_mask_cmul_round_sch
  // CHECK: @llvm.x86.avx512fp16.mask.vfcmul.csh
  return _mm_mask_cmul_round_sch(__W, __U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

__m128h test_mm_maskz_cmul_round_sch(__mmask8 __U, __m128h __A, __m128h __B) {
  // CHECK-LABEL: test_mm_maskz_cmul_round_sch
  // CHECK: @llvm.x86.avx512fp16.mask.vfcmul.csh
  return _mm_maskz_cmul_round_sch(__U, __A, __B, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

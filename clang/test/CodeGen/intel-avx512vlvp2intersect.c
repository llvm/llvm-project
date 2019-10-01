// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown -target-feature +avx512vp2intersect -target-feature +avx512vl -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 %s -ffreestanding -triple=i386-unknown-unknown -target-feature +avx512vp2intersect -target-feature +avx512vl -emit-llvm -o - -Wall -Werror | FileCheck %s

#include <immintrin.h>

void test_mm256_2intersect_epi32(__m256i a, __m256i b, __mmask8 *m0, __mmask8 *m1) {
// CHECK-LABEL: test_mm256_2intersect_epi32
// CHECK: call { <8 x i1>, <8 x i1> } @llvm.x86.avx512.vp2intersect.d.256(<8 x i32> %{{.*}}, <8 x i32> %{{.*}})
// CHECK: extractvalue { <8 x i1>, <8 x i1> } %{{.*}}, 0
// CHECK: extractvalue { <8 x i1>, <8 x i1> } %{{.*}}, 1
  _mm256_2intersect_epi32(a, b, m0, m1);
}

void test_mm256_2intersect_epi64(__m256i a, __m256i b, __mmask8 *m0, __mmask8 *m1) {
// CHECK-LABEL: test_mm256_2intersect_epi64
// CHECK: call { <4 x i1>, <4 x i1> } @llvm.x86.avx512.vp2intersect.q.256(<4 x i64> %{{.*}}, <4 x i64> %{{.*}})
// CHECK: extractvalue { <4 x i1>, <4 x i1> } %{{.*}}, 0
// CHECK: extractvalue { <4 x i1>, <4 x i1> } %{{.*}}, 1
  _mm256_2intersect_epi64(a, b, m0, m1);
}

void test_mm_2intersect_epi32(__m128i a, __m128i b, __mmask8 *m0, __mmask8 *m1) {
// CHECK-LABEL: test_mm_2intersect_epi32
// CHECK: call { <4 x i1>, <4 x i1> } @llvm.x86.avx512.vp2intersect.d.128(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
// CHECK: extractvalue { <4 x i1>, <4 x i1> } %{{.*}}, 0
// CHECK: extractvalue { <4 x i1>, <4 x i1> } %{{.*}}, 1
  _mm_2intersect_epi32(a, b, m0, m1);
}

void test_mm_2intersect_epi64(__m128i a, __m128i b, __mmask8 *m0, __mmask8 *m1) {
// CHECK-LABEL: test_mm_2intersect_epi64
// CHECK: call { <2 x i1>, <2 x i1> } @llvm.x86.avx512.vp2intersect.q.128(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
// CHECK: extractvalue { <2 x i1>, <2 x i1> } %{{.*}}, 0
// CHECK: extractvalue { <2 x i1>, <2 x i1> } %{{.*}}, 1
  _mm_2intersect_epi64(a, b, m0, m1);
}

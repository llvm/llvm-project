// RUN: %clang_cc1 -x c %s -flax-vector-conversions=none -ffreestanding -triple=x86_64-unknown-unknown -target-feature +avx512vp2intersect -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c %s -flax-vector-conversions=none -ffreestanding -triple=i386-unknown-unknown -target-feature +avx512vp2intersect -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ %s -flax-vector-conversions=none -ffreestanding -triple=x86_64-unknown-unknown -target-feature +avx512vp2intersect -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ %s -flax-vector-conversions=none -ffreestanding -triple=i386-unknown-unknown -target-feature +avx512vp2intersect -emit-llvm -o - -Wall -Werror | FileCheck %s

#include <immintrin.h>

void test_mm512_2intersect_epi32(__m512i a, __m512i b, __mmask16 *m0, __mmask16 *m1) {
// CHECK-LABEL: test_mm512_2intersect_epi32
// CHECK: call { <16 x i1>, <16 x i1> } @llvm.x86.avx512.vp2intersect.d.512(<16 x i32> %{{.*}}, <16 x i32> %{{.*}})
// CHECK: extractvalue { <16 x i1>, <16 x i1> } %{{.*}}, 0
// CHECK: extractvalue { <16 x i1>, <16 x i1> } %{{.*}}, 1
  _mm512_2intersect_epi32(a, b, m0, m1);
}

void test_mm512_2intersect_epi64(__m512i a, __m512i b, __mmask8 *m0, __mmask8 *m1) {
// CHECK-LABEL: test_mm512_2intersect_epi64
// CHECK: call { <8 x i1>, <8 x i1> } @llvm.x86.avx512.vp2intersect.q.512(<8 x i64> %{{.*}}, <8 x i64> %{{.*}})
// CHECK: extractvalue { <8 x i1>, <8 x i1> } %{{.*}}, 0
// CHECK: extractvalue { <8 x i1>, <8 x i1> } %{{.*}}, 1
  _mm512_2intersect_epi64(a, b, m0, m1);
}

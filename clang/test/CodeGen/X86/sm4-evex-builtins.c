// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-- -target-feature +sm4 \
// RUN: -target-feature +avx10.2-512 -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 %s -ffreestanding -triple=i386-- -target-feature +sm4 \
// RUN: -target-feature +avx10.2-512 -emit-llvm -o - -Wall -Werror | FileCheck %s

#include <immintrin.h>
#include <stddef.h>

__m512i test_mm512_sm4key4_epi32(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_sm4key4_epi32(
  // CHECK: call <16 x i32> @llvm.x86.vsm4key4512(<16 x i32> %{{.*}}, <16 x i32> %{{.*}})
  return _mm512_sm4key4_epi32(__A, __B);
}

__m512i test_mm512_sm4rnds4_epi32(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_sm4rnds4_epi32(
  // CHECK: call <16 x i32> @llvm.x86.vsm4rnds4512(<16 x i32> %{{.*}}, <16 x i32> %{{.*}})
  return _mm512_sm4rnds4_epi32(__A, __B);
}

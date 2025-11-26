// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-feature +avx512vbmi \
// RUN:   -fclangir -emit-cir %s -o - | FileCheck %s

#include <immintrin.h>
__m512i test_multishift(__m512i x, __m512i y) {
  return _mm512_multishift_epi64_epi8(x, y);
}

// // CHECK: cir.func @test_multishift
// // CHECK: cir.call @__builtin_ia32_vpmultishiftqb512


// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -O3 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-unknown -target-feature +avx -target-feature +avx512f  -target-feature +avx512fp16 -S -o - | FileCheck %s


#include <immintrin.h>

__m256d test_mm256_castpd128_pd256(__m128d A) {
  // CHECK-LABEL: test_mm256_castpd128_pd256
  // CHECK:         # %bb.0:
  // CHECK-NEXT:    # kill: def $xmm0 killed $xmm0 def $ymm0
  // CHECK-NEXT:    ret{{[l|q]}}
  return _mm256_castpd128_pd256(A);
}

__m256 test_mm256_castps128_ps256(__m128 A) {
  // CHECK-LABEL: test_mm256_castps128_ps256
  // CHECK:         # %bb.0:
  // CHECK-NEXT:    # kill: def $xmm0 killed $xmm0 def $ymm0
  // CHECK-NEXT:    ret{{[l|q]}}
  return _mm256_castps128_ps256(A);
}

__m256i test_mm256_castsi128_si256(__m128i A) {
  // CHECK-LABEL: test_mm256_castsi128_si256
  // CHECK:         # %bb.0:
  // CHECK-NEXT:    # kill: def $xmm0 killed $xmm0 def $ymm0
  // CHECK-NEXT:    ret{{[l|q]}}
  return _mm256_castsi128_si256(A);
}

__m256h test_mm256_castph128_ph256(__m128h A) {
  // CHECK-LABEL: test_mm256_castph128_ph256
  // CHECK:         # %bb.0:
  // CHECK-NEXT:    # kill: def $xmm0 killed $xmm0 def $ymm0
  // CHECK-NEXT:    ret{{[l|q]}}
  return _mm256_castph128_ph256(A);
}

__m512h test_mm512_castph128_ph512(__m128h A) {
  // CHECK-LABEL: test_mm512_castph128_ph512
  // CHECK:         # %bb.0:
  // CHECK-NEXT:    # kill: def $xmm0 killed $xmm0 def $zmm0
  // CHECK-NEXT:    ret{{[l|q]}}
  return _mm512_castph128_ph512(A);
}

__m512h test_mm512_castph256_ph512(__m256h A) {
  // CHECK-LABEL: test_mm512_castph256_ph512
  // CHECK:         # %bb.0:
  // CHECK-NEXT:    # kill: def $ymm0 killed $ymm0 def $zmm0
  // CHECK-NEXT:    ret{{[l|q]}}
  return _mm512_castph256_ph512(A);
}

__m512d test_mm512_castpd256_pd512(__m256d A){
  // CHECK-LABEL: test_mm512_castpd256_pd512
  // CHECK:         # %bb.0:
  // CHECK-NEXT:    # kill: def $ymm0 killed $ymm0 def $zmm0
  // CHECK-NEXT:    ret{{[l|q]}}
  return _mm512_castpd256_pd512(A);
}

__m512 test_mm512_castps256_ps512(__m256 A){
  // CHECK-LABEL: test_mm512_castps256_ps512
  // CHECK:         # %bb.0:
  // CHECK-NEXT:    # kill: def $ymm0 killed $ymm0 def $zmm0
  // CHECK-NEXT:    ret{{[l|q]}}
  return _mm512_castps256_ps512(A);
}

__m512d test_mm512_castpd128_pd512(__m128d A){
  // CHECK-LABEL: test_mm512_castpd128_pd512
  // CHECK:         # %bb.0:
  // CHECK-NEXT:    # kill: def $xmm0 killed $xmm0 def $zmm0
  // CHECK-NEXT:    ret{{[l|q]}}
  return _mm512_castpd128_pd512(A);
}

__m512 test_mm512_castps128_ps512(__m128 A){
  // CHECK-LABEL: test_mm512_castps128_ps512
  // CHECK:         # %bb.0:
  // CHECK-NEXT:    # kill: def $xmm0 killed $xmm0 def $zmm0
  // CHECK-NEXT:    ret{{[l|q]}}
  return _mm512_castps128_ps512(A);
}

__m512i test_mm512_castsi128_si512(__m128i A){
  // CHECK-LABEL: test_mm512_castsi128_si512
  // CHECK:         # %bb.0:
  // CHECK-NEXT:    # kill: def $xmm0 killed $xmm0 def $zmm0
  // CHECK-NEXT:    ret{{[l|q]}}
  return _mm512_castsi128_si512(A);
}

__m512i test_mm512_castsi256_si512(__m256i A){
  // CHECK-LABEL: test_mm512_castsi256_si512
  // CHECK:         # %bb.0:
  // CHECK-NEXT:    # kill: def $ymm0 killed $ymm0 def $zmm0
  // CHECK-NEXT:    ret{{[l|q]}}
  return _mm512_castsi256_si512(A);
}

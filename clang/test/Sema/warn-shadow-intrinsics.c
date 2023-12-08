// RUN: %clang_cc1 -ffreestanding -triple x86_64-apple-macosx10.8.0 -fsyntax-only %s

#include <emmintrin.h>

// Test that using two macros from emmintrin do not cause a
// useless -Wshadow warning.
void rdar10679282(void) {
  __m128i qf = _mm_setzero_si128();
  qf = _mm_slli_si128(_mm_add_epi64(qf, _mm_srli_si128(qf, 8)), 8); // no-warning
  (void) qf;
}

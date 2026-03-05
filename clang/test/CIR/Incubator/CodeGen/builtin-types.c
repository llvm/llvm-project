// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx10.2-256 -fclangir -emit-cir -o %t.cir
// RUN: FileCheck --check-prefix=CIR-CHECK --input-file=%t.cir %s

// CIR-CHECK: !cir.vector<!s16i x 8>
#include <emmintrin.h>
int A() { __m128i h = _mm_srli_epi16(h, 0); }

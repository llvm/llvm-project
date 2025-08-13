// REQUIRES: x86-registered-target
// RUN: %clang_cc1 %s -triple=i686-apple-darwin -target-feature +ssse3 -O1 -S -flax-vector-conversions=none -o - | FileCheck %s

#define _mm_alignr_epi8(a, b, n) (__builtin_ia32_palignr128((a), (b), (n)))
typedef long long __m128i __attribute__((__vector_size__(16), __aligned__(16)));

// CHECK: palignr $15, %xmm1, %xmm0
__m128i align1(__m128i a, __m128i b) { return _mm_alignr_epi8(a, b, 15); }
// CHECK: ret
// CHECK: ret
// CHECK-NOT: palignr
__m128i align2(__m128i a, __m128i b) { return _mm_alignr_epi8(a, b, 16); }
// CHECK: psrldq $1, %xmm0
__m128i align3(__m128i a, __m128i b) { return _mm_alignr_epi8(a, b, 17); }
// CHECK: xor
__m128i align4(__m128i a, __m128i b) { return _mm_alignr_epi8(a, b, 32); }

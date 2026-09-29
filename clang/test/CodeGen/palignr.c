// REQUIRES: x86-registered-target
// RUN: %clang_cc1 %s -triple=i686-apple-darwin -target-feature +ssse3 -O1 -S -flax-vector-conversions=none -o - | FileCheck %s

#define _mm_alignr_epi8(a, b, n) (__builtin_ia32_palignr128((a), (b), (n)))
typedef char __v16qi __attribute__((__vector_size__(16)));

// CHECK: palignr $15, %xmm1, %xmm0
__v16qi align1(__v16qi a, __v16qi b) { return _mm_alignr_epi8(a, b, 15); }
// CHECK: ret
// CHECK: ret
// CHECK-NOT: palignr
__v16qi align2(__v16qi a, __v16qi b) { return _mm_alignr_epi8(a, b, 16); }
// CHECK: psrldq $1, %xmm0
__v16qi align3(__v16qi a, __v16qi b) { return _mm_alignr_epi8(a, b, 17); }
// CHECK: xor
__v16qi align4(__v16qi a, __v16qi b) { return _mm_alignr_epi8(a, b, 32); }

// RUN: %clang_cc1 -ffreestanding -fsyntax-only -target-feature +ssse3 -target-feature +mmx -verify -triple x86_64-pc-linux-gnu %s
// RUN: %clang_cc1 -ffreestanding -fsyntax-only -target-feature +ssse3 -target-feature +mmx -verify -triple i686-apple-darwin10 %s

#include <tmmintrin.h>

__m64 test1(__m64 a, __m64 b, int c) {
   return _mm_alignr_pi8(a, b, c); // expected-error {{argument to '__builtin_ia32_psrldqi128_byteshift' must be a constant integer}}
}

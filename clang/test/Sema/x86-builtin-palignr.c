// RUN: %clang_cc1 -ffreestanding -fsyntax-only -target-feature +ssse3 -target-feature +mmx -verify -triple x86_64-pc-linux-gnu %s
// RUN: %clang_cc1 -ffreestanding -fsyntax-only -target-feature +ssse3 -target-feature +mmx -verify -triple i686-apple-darwin10 %s

#include <tmmintrin.h>

__m64 test1(__m64 a, __m64 b, int c) {
   // FIXME: The "incompatible result type" error is due to pr10112 and should
   // be removed when that is fixed.
   return _mm_alignr_pi8(a, b, c); // expected-error {{argument to '__builtin_ia32_palignr' must be a constant integer}} expected-error {{incompatible result type}}
}

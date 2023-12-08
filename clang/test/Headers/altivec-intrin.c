// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu -target-cpu power8 \
// RUN:     -target-feature +altivec -verify %s

// Test special behavior of Altivec intrinsics in this file.

#include <altivec.h>

__attribute__((__aligned__(16))) float x[20];

int main(void)
{
  vector unsigned char l = vec_lvsl (0, &x[1]); // expected-warning {{is deprecated: use assignment for unaligned little endian loads/stores}}
  vector unsigned char r = vec_lvsr (0, &x[1]); // expected-warning {{is deprecated: use assignment for unaligned little endian loads/stores}}
}
// FIXME: As noted in ms-intrin.cpp, it would be nice if we didn't have to
// hard-code the line number from altivec.h here.
// expected-note@altivec.h:* {{deprecated here}}
// expected-note@altivec.h:* {{deprecated here}}

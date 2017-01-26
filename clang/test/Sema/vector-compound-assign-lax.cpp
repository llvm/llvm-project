// RUN: %clang_cc1 -ffreestanding -triple armv7 -target-feature +neon -fsyntax-only %s -verify -Wvector-conversion -DNEON
// RUN: %clang_cc1 -ffreestanding -triple=x86_64-apple-darwin -target-feature +sse2 -fsyntax-only %s -verify -Wvector-conversion -DSSE2

// Test that there are no diagnostics for vector compound assignments where we
// support lax conversions from the RHS type to the LHS type. For context, see:
//
//   <rdar://problem/30112602> cannot convert between vector values ...
//   <rdar://problem/28639522> Cannot create binary operator with two ...
//   <rdar://problem/30110333> Invalid conversion from int64x2 to uint32x4 ...

// expected-no-diagnostics

#include <stdint.h>

typedef int v_int32x4_t __attribute__((__vector_size__(16)));
typedef unsigned v_uint32x4_t __attribute__((__vector_size__(16)));

#ifdef NEON
typedef __attribute__((neon_vector_type(8))) int16_t n_int16x8_t;
typedef __attribute__((neon_vector_type(8))) uint16_t n_uint16x8_t;

void fn1() {
  n_int16x8_t i;
  n_uint16x8_t u;
  i += u;
}
#endif

void fn2() {
  v_int32x4_t i;
  v_uint32x4_t u;
  i += u;
}

#ifdef SSE2
#include <x86intrin.h>

typedef __attribute__((__ext_vector_type__(4))) unsigned int ev_uint32x4_t;

void fn3() {
  __m128i i;
  ev_uint32x4_t u;
  i += u; // FIXME: Converting uint32x4 to int64x2 should *not* be allowed.
}
#endif

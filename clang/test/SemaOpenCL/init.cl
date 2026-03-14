// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only
// expected-no-diagnostics

typedef float float8 __attribute((ext_vector_type(8)));

typedef float float32_t;
typedef __attribute__(( __vector_size__(16) )) float32_t __neon_float32x4_t;
typedef struct __simd128_float32_t {
  __neon_float32x4_t val;
} float32x4_t;

float8 foo(float8 x) { 
  float32x4_t lo;
  float32x4_t hi;
  return (float8) (lo.val, hi.val);
}

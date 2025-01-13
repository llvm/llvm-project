// RUN: %clang_cc1 -triple aarch64 -fsyntax-only -verify %s

// REQUIRES: aarch64-registered-target

typedef __attribute__((neon_vector_type(8))) signed char int8x8_t;
typedef __attribute__((neon_vector_type(16))) signed char int8x16_t;

typedef __MFloat8x8_t mfloat8x8_t;

int8x8_t non_vector(int x) {
  return __builtin_shufflevector(x, x, 3, 2, 1, 0, 3, 2, 1, 0);
  // expected-error@-1 {{first argument to '__builtin_shufflevector' must be of vector type}}
}

mfloat8x8_t unsuported_vector(mfloat8x8_t x) {
  return __builtin_shufflevector(x, x, 3, 2, 1, 0, 3, 2, 1, 0, 0);
  // expected-error@-1 {{unsupported vector type for the result}}
}

int8x8_t non_vector_index(int8x8_t x, int p) {
  return __builtin_shufflevector(x, p);
  // expected-error@-1 {{second argument for __builtin_shufflevector must be integer vector with length equal to the length of the first argument}}
}

int8x8_t bad_vector_index_length(int8x8_t x, int8x16_t p) {
  return __builtin_shufflevector(x, p);
  // expected-error@-1 {{second argument for __builtin_shufflevector must be integer vector with length equal to the length of the first argument}}
}


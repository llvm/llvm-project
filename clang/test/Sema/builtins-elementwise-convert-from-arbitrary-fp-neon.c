// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +neon \
// RUN:   -fsyntax-only -verify %s

typedef unsigned char uint8x8_t __attribute__((neon_vector_type(8)));
typedef __mfp8 mfloat8x8_t __attribute__((neon_vector_type(8)));

void test_neon_vector(uint8x8_t src) {
  (void)__builtin_elementwise_convert_from_f8e5m2_f32(src); // expected-error {{has an unsupported vector kind}}
}

// __mfp8 is an opaque 8-bit container, so it is accepted for the 8-bit
// encodings.
void test_mfp8_scalar(__mfp8 src) {
  _Static_assert(__builtin_types_compatible_p(
      typeof(__builtin_elementwise_convert_from_f8e5m2_f32(src)), float), "");
  _Static_assert(__builtin_types_compatible_p(
      typeof(__builtin_elementwise_convert_from_f8e4m3fn_f16(src)), _Float16),
      "");
}

void test_mfp8_vector(mfloat8x8_t src) {
  (void)__builtin_elementwise_convert_from_f8e5m2_f32(src); // expected-error {{has an unsupported vector kind}}
}

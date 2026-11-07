// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -mvscale-min=1 -mvscale-max=1 -fsyntax-only -verify %s

typedef __SVUint8_t fixed_uint8_t __attribute__((arm_sve_vector_bits(128)));

void convert(__SVUint8_t src) {
  (void)__builtin_elementwise_convert_from_f8e5m2_f32(src); // expected-error {{1st argument must be a scalar or vector of 8-bit integer types (was '__SVUint8_t')}}
}

void convert_fixed(fixed_uint8_t src) {
  (void)__builtin_elementwise_convert_from_f8e5m2_f32(src); // expected-error {{1st argument must be a scalar or vector of 8-bit integer types (was 'fixed_uint8_t'}}
}

// RUN: %clang_cc1 -triple riscv64 -target-feature +v -target-feature +f -target-feature +d -mvscale-min=2 -mvscale-max=2 -fsyntax-only -verify %s

typedef __rvv_uint8m1_t fixed_rvv_uint8 __attribute__((riscv_rvv_vector_bits(128)));

void rvv_vectors(__rvv_uint8m1_t a, __rvv_bool8_t b, fixed_rvv_uint8 c) {
  (void)__builtin_elementwise_convert_from_f8e5m2_f32(a); // expected-error {{must be a scalar or a fixed-length vector declared with 'vector_size' or 'ext_vector_type'}}
  (void)__builtin_elementwise_convert_from_f8e5m2_f32(b); // expected-error {{must be a scalar or a fixed-length vector declared with 'vector_size' or 'ext_vector_type'}}
  (void)__builtin_elementwise_convert_from_f8e5m2_f32(c); // expected-error {{must be a scalar or a fixed-length vector declared with 'vector_size' or 'ext_vector_type'}}
}

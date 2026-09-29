// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +neon -target-feature +sve -target-feature -fp8 -mvscale-min=1 -mvscale-max=1 -fsyntax-only -verify %s

// Scalar __mfp8 is an opaque bit container, available without the FP8 extension.
_Float16 mfp8_to_f16(__mfp8 bits) {
  return __builtin_elementwise_convert_from_f8e5m2_f16(bits);
}

__bf16 mfp8_to_bf16(__mfp8 bits) {
  return __builtin_elementwise_convert_from_f8e4m3fn_bf16(bits);
}

float mfp8_to_f32(__mfp8 bits) {
  return __builtin_elementwise_convert_from_f8e5m3fnu_f32(bits);
}

void invalid_scalar(int bits) {
  (void)__builtin_elementwise_convert_from_f8e5m2_f32(bits); // expected-error {{must be an 8-bit integer or a vector of 8-bit integers, or '__mfp8'}}
}

typedef signed char neon_int8x8 __attribute__((neon_vector_type(8)));
typedef unsigned char neon_uint8x16 __attribute__((neon_vector_type(16)));
typedef unsigned char neon_poly8x8 __attribute__((neon_polyvector_type(8)));
typedef __mfp8 neon_mfp8x8 __attribute__((neon_vector_type(8)));
typedef __mfp8 neon_mfp8x16 __attribute__((neon_vector_type(16)));

// Widening a Neon vector while retaining its vector kind could create an
// invalid ABI type. All Neon kinds are rejected, including integer vectors.
void neon_vectors(neon_int8x8 a, neon_uint8x16 b, neon_poly8x8 c,
                  neon_mfp8x8 d, neon_mfp8x16 e) {
  (void)__builtin_elementwise_convert_from_f8e5m2_f32(a); // expected-error {{must be a scalar or a fixed-length vector declared with 'vector_size' or 'ext_vector_type'}}
  (void)__builtin_elementwise_convert_from_f8e5m2_f16(b); // expected-error {{must be a scalar or a fixed-length vector declared with 'vector_size' or 'ext_vector_type'}}
  (void)__builtin_elementwise_convert_from_f8e5m2_bf16(c); // expected-error {{must be a scalar or a fixed-length vector declared with 'vector_size' or 'ext_vector_type'}}
  (void)__builtin_elementwise_convert_from_f8e4m3fn_f32(d); // expected-error {{must be a scalar or a fixed-length vector declared with 'vector_size' or 'ext_vector_type'}}
  (void)__builtin_elementwise_convert_from_f8e5m3fnu_f32(e); // expected-error {{must be a scalar or a fixed-length vector declared with 'vector_size' or 'ext_vector_type'}}
}

typedef __SVUint8_t fixed_sve_uint8 __attribute__((arm_sve_vector_bits(128)));
typedef __SVBool_t fixed_sve_bool __attribute__((arm_sve_vector_bits(128)));

void sve_vectors(__SVUint8_t a, __SVBool_t b, fixed_sve_uint8 c,
                 fixed_sve_bool d) {
  (void)__builtin_elementwise_convert_from_f8e5m2_f32(a); // expected-error {{must be a scalar or a fixed-length vector declared with 'vector_size' or 'ext_vector_type'}}
  (void)__builtin_elementwise_convert_from_f8e5m2_f32(b); // expected-error {{must be a scalar or a fixed-length vector declared with 'vector_size' or 'ext_vector_type'}}
  (void)__builtin_elementwise_convert_from_f8e5m2_f32(c); // expected-error {{must be a scalar or a fixed-length vector declared with 'vector_size' or 'ext_vector_type'}}
  (void)__builtin_elementwise_convert_from_f8e5m2_f32(d); // expected-error {{must be a scalar or a fixed-length vector declared with 'vector_size' or 'ext_vector_type'}}
}

typedef unsigned char uchar8 __attribute__((ext_vector_type(8)));
typedef float float8 __attribute__((ext_vector_type(8)));

// An ordinary vector can produce a result larger than a Neon register.
float8 ordinary_vector(uchar8 bits) {
  return __builtin_elementwise_convert_from_f8e5m2_f32(bits);
}

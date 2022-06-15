// RUN: %clang_cc1 %s -triple armv7 -target-feature +neon -fsyntax-only -fdouble-square-bracket-attributes -verify
// RUN: %clang_cc1 %s -triple armv8 -target-feature +neon -fsyntax-only -fdouble-square-bracket-attributes -verify

typedef float float32_t;
typedef signed char poly8_t;
typedef short poly16_t;
typedef unsigned __INT64_TYPE__ uint64_t;

// Define some valid Neon types.
typedef __attribute__((neon_vector_type(2))) int int32x2_t;
typedef __attribute__((neon_vector_type(4))) int int32x4_t;
typedef __attribute__((neon_vector_type(1))) uint64_t uint64x1_t;
typedef __attribute__((neon_vector_type(2))) uint64_t uint64x2_t;
typedef __attribute__((neon_vector_type(2))) float32_t float32x2_t;
typedef __attribute__((neon_vector_type(4))) float32_t float32x4_t;
typedef __attribute__((neon_polyvector_type(16))) poly8_t  poly8x16_t;
typedef __attribute__((neon_polyvector_type(8)))  poly16_t poly16x8_t;

// Verify that we can use the [[]] spelling of the attributes.
// We intentionally use the same type alias names to check that both versions
// define the same type.
typedef int [[clang::neon_vector_type(2)]] int32x2_t;
typedef poly8_t [[clang::neon_polyvector_type(16)]] poly8x16_t;

// Verify that we can use the attributes outside of a typedef.
int [[clang::neon_vector_type(2)]] int32x2_var;
poly8_t [[clang::neon_polyvector_type(16)]] poly8x16_var;

// The attributes must have a single argument.
typedef __attribute__((neon_vector_type(2, 4))) int only_one_arg; // expected-error{{'neon_vector_type' attribute takes one argument}}

// The number of elements must be an ICE.
typedef __attribute__((neon_vector_type(2.0))) int non_int_width; // expected-error{{'neon_vector_type' attribute requires an integer constant}}

// Only certain element types are allowed.
typedef __attribute__((neon_vector_type(4))) void* ptr_elt; // expected-error{{invalid vector element type}}
typedef __attribute__((neon_polyvector_type(4))) float32_t bad_poly_elt; // expected-error{{invalid vector element type}}
struct aggr { signed char c; };
typedef __attribute__((neon_vector_type(8))) struct aggr aggregate_elt; // expected-error{{invalid vector element type}}

// The total vector size must be 64 or 128 bits.
typedef __attribute__((neon_vector_type(1))) int int32x1_t; // expected-error{{Neon vector size must be 64 or 128 bits}}
typedef __attribute__((neon_vector_type(3))) int int32x3_t; // expected-error{{Neon vector size must be 64 or 128 bits}}

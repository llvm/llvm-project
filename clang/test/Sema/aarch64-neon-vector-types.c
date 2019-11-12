// RUN: %clang_cc1 %s -triple aarch64-none-linux-gnu -target-feature +neon -fsyntax-only -verify
// RUN: %clang_cc1 %s -triple aarch64-none-linux-gnu -target-feature +neon -DUSE_LONG -fsyntax-only -verify
// RUN: %clang_cc1 %s -triple arm64-none-linux-gnu -target-feature +neon -fsyntax-only -verify
// RUN: %clang_cc1 %s -triple arm64-none-linux-gnu -target-feature +neon -DUSE_LONG -fsyntax-only -verify

// RUN: %clang_cc1 %s -triple arm64_32-apple-ios -target-feature +neon -fsyntax-only -verify

typedef float float32_t;
typedef unsigned char poly8_t;
typedef unsigned short poly16_t;

// Both "long" and "long long" should work for 64-bit arch like aarch64.
// stdint.h in gnu libc is using "long" for 64-bit arch.
#if USE_LONG
typedef long int64_t;
typedef unsigned long uint64_t;
#else
typedef long long int64_t;
typedef unsigned long long uint64_t;
#endif
typedef uint64_t poly64_t;

// Define some valid Neon types.
typedef __attribute__((neon_vector_type(2))) int int32x2_t;
typedef __attribute__((neon_vector_type(4))) int int32x4_t;
typedef __attribute__((neon_vector_type(1))) int64_t int64x1_t;
typedef __attribute__((neon_vector_type(2))) int64_t int64x2_t;
typedef __attribute__((neon_vector_type(1))) uint64_t uint64x1_t;
typedef __attribute__((neon_vector_type(2))) uint64_t uint64x2_t;
typedef __attribute__((neon_vector_type(2))) float32_t float32x2_t;
typedef __attribute__((neon_vector_type(4))) float32_t float32x4_t;
typedef __attribute__((neon_polyvector_type(16))) poly8_t  poly8x16_t;
typedef __attribute__((neon_polyvector_type(8)))  poly16_t poly16x8_t;
typedef __attribute__((neon_polyvector_type(1)))  poly64_t poly64x1_t;
typedef __attribute__((neon_polyvector_type(2)))  poly64_t poly64x2_t;

// The attributes must have a single argument.
typedef __attribute__((neon_vector_type(2, 4))) int only_one_arg; // expected-error{{attribute takes one argument}}

// The number of elements must be an ICE.
typedef __attribute__((neon_vector_type(2.0))) int non_int_width; // expected-error{{attribute requires an integer constant}}

// Only certain element types are allowed.
typedef __attribute__((neon_vector_type(2))) double double_elt;
typedef __attribute__((neon_vector_type(4))) void* ptr_elt; // expected-error{{invalid vector element type}}
typedef __attribute__((neon_polyvector_type(4))) float32_t bad_poly_elt; // expected-error{{invalid vector element type}}
struct aggr { signed char c; };
typedef __attribute__((neon_vector_type(8))) struct aggr aggregate_elt; // expected-error{{invalid vector element type}}

// The total vector size must be 64 or 128 bits.
typedef __attribute__((neon_vector_type(1))) int int32x1_t; // expected-error{{Neon vector size must be 64 or 128 bits}}
typedef __attribute__((neon_vector_type(3))) int int32x3_t; // expected-error{{Neon vector size must be 64 or 128 bits}}

// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -ffreestanding -fsyntax-only -verify -std=c++11 -msve-vector-bits=512 -fallow-half-arguments-and-returns -Wconversion %s
// expected-no-diagnostics

#include <stdint.h>

#define N __ARM_FEATURE_SVE_BITS

typedef __SVInt8_t svint8_t;
typedef svint8_t fixed_int8_t __attribute__((arm_sve_vector_bits(N)));
typedef int8_t gnu_int8_t __attribute__((vector_size(N / 8)));

template<typename T> struct S { T var; };

S<fixed_int8_t> s;

// Test implicit casts between VLA and VLS vectors
svint8_t to_svint8_t(fixed_int8_t x) { return x; }
fixed_int8_t from_svint8_t(svint8_t x) { return x; }

// Test implicit casts between GNU and VLA vectors
svint8_t to_svint8_t__from_gnu_int8_t(gnu_int8_t x) { return x; }
gnu_int8_t from_svint8_t__to_gnu_int8_t(svint8_t x) { return x; }

// Test implicit casts between GNU and VLS vectors
fixed_int8_t to_fixed_int8_t__from_gnu_int8_t(gnu_int8_t x) { return x; }
gnu_int8_t from_fixed_int8_t__to_gnu_int8_t(fixed_int8_t x) { return x; }

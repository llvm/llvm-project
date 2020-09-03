// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fsyntax-only -verify -std=c++11 -msve-vector-bits=512 -fallow-half-arguments-and-returns %s
// expected-no-diagnostics

#define N __ARM_FEATURE_SVE_BITS

typedef __SVInt8_t svint8_t;
typedef svint8_t fixed_int8_t __attribute__((arm_sve_vector_bits(N)));

template<typename T> struct S { T var; };

S<fixed_int8_t> s;

svint8_t to_svint8_t(fixed_int8_t x) { return x; }
fixed_int8_t from_svint8_t(svint8_t x) { return x; }

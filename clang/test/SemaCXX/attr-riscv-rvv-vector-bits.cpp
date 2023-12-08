// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +zve64x -ffreestanding -fsyntax-only -verify -std=c++11 -mvscale-min=4 -mvscale-max=4 -Wconversion %s
// expected-no-diagnostics

#include <stdint.h>

typedef __rvv_int8m1_t vint8m1_t;
typedef vint8m1_t fixed_int8m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));
typedef int8_t gnu_int8m1_t __attribute__((vector_size(__riscv_v_fixed_vlen / 8)));

template<typename T> struct S { T var; };

S<fixed_int8m1_t> s;

// Test implicit casts between VLA and VLS vectors
vint8m1_t to_vint8m1_t(fixed_int8m1_t x) { return x; }
fixed_int8m1_t from_vint8m1_t(vint8m1_t x) { return x; }

// Test implicit casts between GNU and VLA vectors
vint8m1_t to_vint8m1_t__from_gnu_int8m1_t(gnu_int8m1_t x) { return x; }
gnu_int8m1_t from_vint8m1_t__to_gnu_int8m1_t(vint8m1_t x) { return x; }

// Test implicit casts between GNU and VLS vectors
fixed_int8m1_t to_fixed_int8m1_t__from_gnu_int8m1_t(gnu_int8m1_t x) { return x; }
gnu_int8m1_t from_fixed_int8m1_t__to_gnu_int8m1_t(fixed_int8m1_t x) { return x; }

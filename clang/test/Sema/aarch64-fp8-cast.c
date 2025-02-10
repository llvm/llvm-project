// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +neon -verify -emit-llvm -o - %s

// REQUIRES: aarch64-registered-target

#include <arm_neon.h>

// Bitcast between FP8 Neon vectors
mfloat8x8_t err_test_f8_f8(mfloat8x16_t x) {
    return (mfloat8x8_t) x;
// expected-error@-1 {{invalid conversion between vector type 'mfloat8x8_t' (vector of 8 'mfloat8_t' values) and 'mfloat8x16_t' (vector of 16 'mfloat8_t' values) of different size}}
}

mfloat8x16_t err_testq_f8_f8(mfloat8x8_t x) {
    return (mfloat8x16_t) x;
// expected-error@-1 {{invalid conversion between vector type 'mfloat8x16_t' (vector of 16 'mfloat8_t' values) and 'mfloat8x8_t' (vector of 8 'mfloat8_t' values) of different size}}
}

// Bitcast between FP8 and int8 Neon vectors
mfloat8x8_t err_test_f8_s8(int8x16_t x) {
    return (mfloat8x8_t) x;
// expected-error@-1 {{invalid conversion between vector type 'mfloat8x8_t' (vector of 8 'mfloat8_t' values) and 'int8x16_t' (vector of 16 'int8_t' values) of different size}}
}

int8x8_t err_test_s8_f8(mfloat8x16_t x) {
    return (int8x8_t) x;
// expected-error@-1 {{invalid conversion between vector type 'int8x8_t' (vector of 8 'int8_t' values) and 'mfloat8x16_t' (vector of 16 'mfloat8_t' values) of different size}}
}

mfloat8x16_t err_testq_f8_s8(int8x8_t x) {
    return (mfloat8x16_t) x;
// expected-error@-1 {{invalid conversion between vector type 'mfloat8x16_t' (vector of 16 'mfloat8_t' values) and 'int8x8_t' (vector of 8 'int8_t' values) of different size}}
}

int8x16_t err_testq_s8_f8(mfloat8x8_t x) {
    return (int8x16_t) x;
// expected-error@-1 {{invalid conversion between vector type 'int8x16_t' (vector of 16 'int8_t' values) and 'mfloat8x8_t' (vector of 8 'mfloat8_t' values) of different size}}
}

// Bitcast between FP8 and float32 Neon vectors
mfloat8x8_t err_test_f8_f32(float32x4_t x) {
    return (mfloat8x8_t) x;
// expected-error@-1 {{invalid conversion between vector type 'mfloat8x8_t' (vector of 8 'mfloat8_t' values) and 'float32x4_t' (vector of 4 'float32_t' values) of different size}}
}

float32x2_t err_test_f32_f8(mfloat8x16_t x) {
    return (float32x2_t) x;
// expected-error@-1 {{invalid conversion between vector type 'float32x2_t' (vector of 2 'float32_t' values) and 'mfloat8x16_t' (vector of 16 'mfloat8_t' values) of different size}}
}

mfloat8x16_t err_testq_f8_f32(float32x2_t x) {
    return (mfloat8x16_t) x;
// expected-error@-1 {{invalid conversion between vector type 'mfloat8x16_t' (vector of 16 'mfloat8_t' values) and 'float32x2_t' (vector of 2 'float32_t' values) of different size}}
}

float32x4_t err_testq_f32_f8(mfloat8x8_t x) {
    return (float32x4_t) x;
// expected-error@-1 {{invalid conversion between vector type 'float32x4_t' (vector of 4 'float32_t' values) and 'mfloat8x8_t' (vector of 8 'mfloat8_t' values) of different size}}
}

// Bitcast between FP8 and poly128_t (which is integral)
mfloat8x8_t err_testq_f8_p128(poly128_t x) {
    return (mfloat8x8_t) x;
// expected-error@-1 {{invalid conversion between vector type 'mfloat8x8_t' (vector of 8 'mfloat8_t' values) and integer type 'poly128_t' (aka 'unsigned __int128') of different size}}
}

poly128_t err_testq_p128_f8(mfloat8x8_t x) {
    return (poly128_t) x;
// expected-error@-1 {{invalid conversion between vector type 'mfloat8x8_t' (vector of 8 'mfloat8_t' values) and integer type 'poly128_t' (aka 'unsigned __int128') of different size}}
}

// Bitcast between FP8 and a non-integral type
mfloat8x8_t err_test_f8_ptr(void *p) {
    return (mfloat8x8_t) p;
// expected-error@-1 {{invalid conversion between vector type 'mfloat8x8_t' (vector of 8 'mfloat8_t' values) and scalar type 'void *'}}
}

void *err_test_ptr_f8(mfloat8x8_t v) {
    return (void *) v;
// expected-error@-1 {{invalid conversion between vector type 'mfloat8x8_t' (vector of 8 'mfloat8_t' values) and scalar type 'void *'}}
}

mfloat8x8_t err_test_f8_dbl(double v) {
    return (mfloat8x8_t) v;
// expected-error@-1 {{invalid conversion between vector type 'mfloat8x8_t' (vector of 8 'mfloat8_t' values) and scalar type 'double'}}
}

double err_test_dbl_f8(mfloat8x8_t v) {
    return (double) v;
// expected-error@-1 {{invalid conversion between vector type 'mfloat8x8_t' (vector of 8 'mfloat8_t' values) and scalar type 'double'}}
}

struct S {
    char ch[16];
};

mfloat8x16_t err_test_f8_agg(struct S s) {
    return (mfloat8x16_t) s;
// expected-error@-1 {{operand of type 'struct S' where arithmetic or pointer type is required}}
}

struct S err_test_agg_f8(mfloat8x16_t v) {
    return (struct S) v;
// expected-error@-1 {{used type 'struct S' where arithmetic or pointer type is required}}
}

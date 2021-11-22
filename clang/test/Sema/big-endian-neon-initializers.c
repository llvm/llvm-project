// RUN: %clang_cc1 %s -triple aarch64_be -target-feature +neon -verify -fsyntax-only -ffreestanding
// RUN: %clang_cc1 %s -triple armebv7 -target-cpu cortex-a8 -verify -fsyntax-only -ffreestanding

// REQUIRES: aarch64-registered-target || arm-registered-target

#include <arm_neon.h>

int32x4_t x = {1, 2, 3, 4}; // expected-warning{{vector initializers are not compatible with NEON intrinsics}} expected-note{{consider using vld1q_s32() to initialize a vector from memory, or vcombine_s32(vcreate_s32(), vcreate_s32()) to initialize from integer constants}}
int16x4_t y = {1, 2, 3, 4}; // expected-warning{{vector initializers are not compatible with NEON intrinsics}} expected-note{{consider using vld1_s16() to initialize a vector from memory, or vcreate_s16() to initialize from an integer constant}}
int64x2_t z = {1, 2}; // expected-warning{{vector initializers are not compatible with NEON intrinsics}} expected-note{{consider using vld1q_s64() to initialize a vector from memory, or vcombine_s64(vcreate_s64(), vcreate_s64()) to initialize from integer constants}}
float32x2_t b = {1, 2}; // expected-warning{{vector initializers are not compatible with NEON intrinsics}} expected-note{{consider using vld1_f32() to initialize a vector from memory, or vcreate_f32() to initialize from an integer constant}}

// No warning expected here.
typedef int v4si __attribute__ ((vector_size (16)));
v4si c = {1, 2, 3, 4};

// RUN: %clang_cc1 -triple thumbv8.1m.main-none-none-eabi -target-feature +mve.fp -flax-vector-conversions=all -Werror -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple thumbv8.1m.main-none-none-eabi -target-feature +mve.fp -flax-vector-conversions=all -verify -fsyntax-only -DERROR_CHECK %s

typedef   signed short      int16_t;
typedef   signed int        int32_t;
typedef   signed long long  int64_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;

typedef __attribute__((neon_vector_type(8), __clang_arm_mve_strict_polymorphism))  int16_t  int16x8_t;
typedef __attribute__((neon_vector_type(4), __clang_arm_mve_strict_polymorphism))  int32_t  int32x4_t;
typedef __attribute__((neon_vector_type(2), __clang_arm_mve_strict_polymorphism))  int64_t  int64x2_t;
typedef __attribute__((neon_vector_type(8), __clang_arm_mve_strict_polymorphism)) uint16_t uint16x8_t;
typedef __attribute__((neon_vector_type(4), __clang_arm_mve_strict_polymorphism)) uint32_t uint32x4_t;
typedef __attribute__((neon_vector_type(2), __clang_arm_mve_strict_polymorphism)) uint64_t uint64x2_t;

__attribute__((overloadable))
int overload(int16x8_t x, int16_t y); // expected-note {{candidate function}}
__attribute__((overloadable))
int overload(int32x4_t x, int32_t y); // expected-note {{candidate function}}
__attribute__((overloadable))
int overload(uint16x8_t x, uint16_t y); // expected-note {{candidate function}}
__attribute__((overloadable))
int overload(uint32x4_t x, uint32_t y); // expected-note {{candidate function}}

int16_t s16;
int32_t s32;
uint16_t u16;
uint32_t u32;

int16x8_t vs16;
int32x4_t vs32;
uint16x8_t vu16;
uint32x4_t vu32;

// ----------------------------------------------------------------------
// Simple cases where the types are correctly matched

// CHECK-LABEL: @test_easy_s16(
// CHECK: call i32 @_Z8overload{{[a-zA-Z0-9_]+}}_int16
int test_easy_s16(void) { return overload(vs16, s16); }

// CHECK-LABEL: @test_easy_u16(
// CHECK: call i32 @_Z8overload{{[a-zA-Z0-9_]+}}_uint16
int test_easy_u16(void) { return overload(vu16, u16); }

// CHECK-LABEL: @test_easy_s32(
// CHECK: call i32 @_Z8overload{{[a-zA-Z0-9_]+}}_int32
int test_easy_s32(void) { return overload(vs32, s32); }

// CHECK-LABEL: @test_easy_u32(
// CHECK: call i32 @_Z8overload{{[a-zA-Z0-9_]+}}_uint32
int test_easy_u32(void) { return overload(vu32, u32); }

// ----------------------------------------------------------------------
// Do arithmetic on the scalar, and it may get promoted. We still expect the
// same overloads to be selected if that happens.

// CHECK-LABEL: @test_promote_s16(
// CHECK: call i32 @_Z8overload{{[a-zA-Z0-9_]+}}_int16
int test_promote_s16(void) { return overload(vs16, s16 + 1); }

// CHECK-LABEL: @test_promote_u16(
// CHECK: call i32 @_Z8overload{{[a-zA-Z0-9_]+}}_uint16
int test_promote_u16(void) { return overload(vu16, u16 + 1); }

// CHECK-LABEL: @test_promote_s32(
// CHECK: call i32 @_Z8overload{{[a-zA-Z0-9_]+}}_int32
int test_promote_s32(void) { return overload(vs32, s32 + 1); }

// CHECK-LABEL: @test_promote_u32(
// CHECK: call i32 @_Z8overload{{[a-zA-Z0-9_]+}}_uint32
int test_promote_u32(void) { return overload(vu32, u32 + 1); }

// ----------------------------------------------------------------------
// Write a simple integer literal without qualification, and expect
// the vector type to make it unambiguous which integer type you meant
// the literal to be.

// CHECK-LABEL: @test_literal_s16(
// CHECK: call i32 @_Z8overload{{[a-zA-Z0-9_]+}}_int16
int test_literal_s16(void) { return overload(vs16, 1); }

// CHECK-LABEL: @test_literal_u16(
// CHECK: call i32 @_Z8overload{{[a-zA-Z0-9_]+}}_uint16
int test_literal_u16(void) { return overload(vu16, 1); }

// CHECK-LABEL: @test_literal_s32(
// CHECK: call i32 @_Z8overload{{[a-zA-Z0-9_]+}}_int32
int test_literal_s32(void) { return overload(vs32, 1); }

// CHECK-LABEL: @test_literal_u32(
// CHECK: call i32 @_Z8overload{{[a-zA-Z0-9_]+}}_uint32
int test_literal_u32(void) { return overload(vu32, 1); }

// ----------------------------------------------------------------------
// All of those overload resolutions are supposed to be unambiguous even when
// lax vector conversion is enabled. Check here that a lax conversion in a
// different context still works.
int16x8_t lax_conversion(void) { return vu32; }

// ----------------------------------------------------------------------
// Use a vector type that there really _isn't_ any overload for, and
// make sure that we get a fatal compile error.

#ifdef ERROR_CHECK
int expect_error(uint64x2_t v) {
  return overload(v, 2); // expected-error {{no matching function for call to 'overload'}}
}

typedef __attribute__((__clang_arm_mve_strict_polymorphism)) int i; // expected-error {{'__clang_arm_mve_strict_polymorphism' attribute can only be applied to an MVE/NEON vector type}}
typedef __attribute__((__clang_arm_mve_strict_polymorphism)) int f(); // expected-error {{'__clang_arm_mve_strict_polymorphism' attribute can only be applied to an MVE/NEON vector type}}
typedef __attribute__((__clang_arm_mve_strict_polymorphism)) struct { uint16x8_t v; } s; // expected-error {{'__clang_arm_mve_strict_polymorphism' attribute can only be applied to an MVE/NEON vector type}}
#endif

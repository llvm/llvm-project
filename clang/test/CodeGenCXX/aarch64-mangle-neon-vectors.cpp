// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon %s -emit-llvm -o - | FileCheck %s

typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef signed char int8_t;
typedef signed short int16_t;
typedef signed long int64_t;
typedef unsigned long uint64_t;
typedef unsigned char poly8_t;
typedef unsigned short poly16_t;
typedef __fp16 float16_t;
typedef float float32_t;
typedef double float64_t;

typedef __attribute__((neon_vector_type(8))) int8_t int8x8_t;
typedef __attribute__((neon_vector_type(16))) int8_t int8x16_t;
typedef __attribute__((neon_vector_type(4))) int16_t int16x4_t;
typedef __attribute__((neon_vector_type(8))) int16_t int16x8_t;
typedef __attribute__((neon_vector_type(2))) int int32x2_t;
typedef __attribute__((neon_vector_type(4))) int int32x4_t;
typedef __attribute__((neon_vector_type(2))) int64_t int64x2_t;
typedef __attribute__((neon_vector_type(8))) uint8_t uint8x8_t;
typedef __attribute__((neon_vector_type(16))) uint8_t uint8x16_t;
typedef __attribute__((neon_vector_type(4))) uint16_t uint16x4_t;
typedef __attribute__((neon_vector_type(8))) uint16_t uint16x8_t;
typedef __attribute__((neon_vector_type(2))) unsigned int uint32x2_t;
typedef __attribute__((neon_vector_type(4))) unsigned int uint32x4_t;
typedef __attribute__((neon_vector_type(2))) uint64_t uint64x2_t;
typedef __attribute__((neon_vector_type(4))) float16_t float16x4_t;
typedef __attribute__((neon_vector_type(8))) float16_t float16x8_t;
typedef __attribute__((neon_vector_type(2))) float32_t float32x2_t;
typedef __attribute__((neon_vector_type(4))) float32_t float32x4_t;
typedef __attribute__((neon_vector_type(2))) float64_t float64x2_t;
typedef __attribute__((neon_polyvector_type(8))) poly8_t poly8x8_t;
typedef __attribute__((neon_polyvector_type(16))) poly8_t poly8x16_t;
typedef __attribute__((neon_polyvector_type(4))) poly16_t poly16x4_t;
typedef __attribute__((neon_polyvector_type(8))) poly16_t poly16x8_t;

// CHECK: 10__Int8x8_t
void f1(int8x8_t) {}
// CHECK: 11__Int16x4_t
void f2(int16x4_t) {}
// CHECK: 11__Int32x2_t
void f3(int32x2_t) {}
// CHECK: 11__Uint8x8_t
void f4(uint8x8_t) {}
// CHECK: 12__Uint16x4_t
void f5(uint16x4_t) {}
// CHECK: 13__Float16x4_t
void f6(float16x4_t) {}
// CHECK: 13__Float16x8_t
void f7(float16x8_t) {}
// CHECK: 12__Uint32x2_t
void f8(uint32x2_t) {}
// CHECK: 13__Float32x2_t
void f9(float32x2_t) {}
// CHECK: 13__Float32x4_t
void f10(float32x4_t) {}
// CHECK: 11__Poly8x8_t
void f11(poly8x8_t v) {}
// CHECK: 12__Poly16x4_t
void f12(poly16x4_t v) {}
// CHECK:12__Poly8x16_t
void f13(poly8x16_t v) {}
// CHECK:12__Poly16x8_t
void f14(poly16x8_t v) {}
// CHECK: 11__Int8x16_t
void f15(int8x16_t) {}
// CHECK: 11__Int16x8_t
void f16(int16x8_t) {}
// CHECK:11__Int32x4_t
void f17(int32x4_t) {}
// CHECK: 12__Uint8x16_t
void f18(uint8x16_t) {}
// CHECK: 12__Uint16x8_t
void f19(uint16x8_t) {}
// CHECK: 12__Uint32x4_t
void f20(uint32x4_t) {}
// CHECK: 11__Int64x2_t
void f21(int64x2_t) {}
// CHECK: 12__Uint64x2_t
void f22(uint64x2_t) {}
// CHECK: 13__Float64x2_t
void f23(float64x2_t) {}

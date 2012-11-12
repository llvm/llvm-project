// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s

typedef float float32_t;
typedef signed char poly8_t;
typedef short poly16_t;
typedef unsigned long long uint64_t;

typedef __attribute__((neon_vector_type(2))) int int32x2_t;
typedef __attribute__((neon_vector_type(4))) int int32x4_t;
typedef __attribute__((neon_vector_type(1))) uint64_t uint64x1_t;
typedef __attribute__((neon_vector_type(2))) uint64_t uint64x2_t;
typedef __attribute__((neon_vector_type(2))) float32_t float32x2_t;
typedef __attribute__((neon_vector_type(4))) float32_t float32x4_t;
typedef __attribute__((neon_polyvector_type(16))) poly8_t  poly8x16_t;
typedef __attribute__((neon_polyvector_type(8)))  poly16_t poly16x8_t;

// CHECK: 16__simd64_int32_t
void f1(int32x2_t v) { }
// CHECK: 17__simd128_int32_t
void f2(int32x4_t v) { }
// CHECK: 17__simd64_uint64_t
void f3(uint64x1_t v) { }
// CHECK: 18__simd128_uint64_t
void f4(uint64x2_t v) { }
// CHECK: 18__simd64_float32_t
void f5(float32x2_t v) { }
// CHECK: 19__simd128_float32_t
void f6(float32x4_t v) { }
// CHECK: 17__simd128_poly8_t
void f7(poly8x16_t v) { }
// CHECK: 18__simd128_poly16_t
void f8(poly16x8_t v) { }

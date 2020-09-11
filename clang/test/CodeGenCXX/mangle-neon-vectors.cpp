// RUN: %clang_cc1 -triple armv7-apple-ios -target-feature +neon  %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-ios -target-feature +neon %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple arm64-linux-gnu -target-feature +neon %s -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-AARCH64
// RUN: %clang_cc1 -triple arm64-linux-gnu -target-feature +neon -target-feature +bf16 %s -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-AARCH64-BF16

typedef float float32_t;
typedef double float64_t;
typedef __fp16 float16_t;
#if defined(__aarch64__)
typedef unsigned char poly8_t;
typedef unsigned short poly16_t;
#else
typedef signed char poly8_t;
typedef short poly16_t;
#endif
typedef unsigned __INT64_TYPE__ uint64_t;

#if defined(__ARM_FEATURE_BF16)
typedef __bf16 bfloat16_t;
#endif

typedef __attribute__((neon_vector_type(2))) int int32x2_t;
typedef __attribute__((neon_vector_type(4))) int int32x4_t;
typedef __attribute__((neon_vector_type(1))) uint64_t uint64x1_t;
typedef __attribute__((neon_vector_type(2))) uint64_t uint64x2_t;
typedef __attribute__((neon_vector_type(2))) float32_t float32x2_t;
typedef __attribute__((neon_vector_type(4))) float32_t float32x4_t;
typedef __attribute__((neon_vector_type(4))) float16_t float16x4_t;
typedef __attribute__((neon_vector_type(8))) float16_t float16x8_t;
#ifdef __aarch64__
typedef __attribute__((neon_vector_type(2))) float64_t float64x2_t;
#endif
typedef __attribute__((neon_polyvector_type(16))) poly8_t  poly8x16_t;
typedef __attribute__((neon_polyvector_type(8)))  poly16_t poly16x8_t;

#if defined(__ARM_FEATURE_BF16)
typedef __attribute__((neon_vector_type(4))) __bf16 bfloat16x4_t;
#endif

// CHECK: 16__simd64_int32_t
// CHECK-AARCH64: 11__Int32x2_t
void f1(int32x2_t v) { }

// CHECK: 17__simd128_int32_t
// CHECK-AARCH64: 11__Int32x4_t
void f2(int32x4_t v) { }

// CHECK: 17__simd64_uint64_t
// CHECK-AARCH64: 12__Uint64x1_t
void f3(uint64x1_t v) { }

// CHECK: 18__simd128_uint64_t
// CHECK-AARCH64: 12__Uint64x2_t
void f4(uint64x2_t v) { }

// CHECK: 18__simd64_float32_t
// CHECK-AARCH64: 13__Float32x2_t
void f5(float32x2_t v) { }

// CHECK: 19__simd128_float32_t
// CHECK-AARCH64: 13__Float32x4_t
void f6(float32x4_t v) { }

// CHECK: 18__simd64_float16_t
// CHECK-AARCH64: 13__Float16x4_t
void f7(float16x4_t v) {}

// CHECK: 19__simd128_float16_t
// CHECK-AARCH64: 13__Float16x8_t
void f8(float16x8_t v) {}

// CHECK: 17__simd128_poly8_t
// CHECK-AARCH64: 12__Poly8x16_t
void f9(poly8x16_t v) {}

// CHECK: 18__simd128_poly16_t
// CHECK-AARCH64: 12__Poly16x8_t
void f10(poly16x8_t v) {}

#ifdef __aarch64__
// CHECK-AARCH64: 13__Float64x2_t
void f11(float64x2_t v) { }
#endif

#if defined(__ARM_FEATURE_BF16)
// CHECK-AARCH64-BF16: 14__Bfloat16x4_t
void f12(bfloat16x4_t v) {}
#endif

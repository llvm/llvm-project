// Test that NEON types are defined, even when arm_neon.h is not included.
// as required by AAPCS64 "Support for Advanced SIMD Extensions".

// RUN: %clang_cc1 -ast-dump -triple aarch64-linux-gnu %s -x c | FileCheck %s
// RUN: %clang_cc1 -ast-dump -triple aarch64-linux-gnu %s -x c++ | FileCheck %s
// RUN: %clang_cc1 -verify -verify-ignore-unexpected=note -triple x86_64 %s -x c
// RUN: %clang_cc1 -verify -verify-ignore-unexpected=note -triple x86_64 %s -x c++
// RUN: %clang_cc1 -verify -verify-ignore-unexpected=note -triple arm-linux-gnu %s -x c
// RUN: %clang_cc1 -verify -verify-ignore-unexpected=note -triple arm-linux-gnu %s -x c++

__Int8x8_t Int8x8;
// CHECK: Int8x8 '__Int8x8_t':'__attribute__((neon_vector_type(8))) signed char'
// expected-error@-2{{unknown type name '__Int8x8_t'}}

__Int16x4_t Int16x4;
// CHECK: Int16x4 '__Int16x4_t':'__attribute__((neon_vector_type(4))) short'
// expected-error@-2{{unknown type name '__Int16x4_t'}}

__Int32x2_t Int32x2;
// CHECK: Int32x2 '__Int32x2_t':'__attribute__((neon_vector_type(2))) int'
// expected-error@-2{{unknown type name '__Int32x2_t'}}

__Uint8x8_t Uint8x8;
// CHECK: Uint8x8 '__Uint8x8_t':'__attribute__((neon_vector_type(8))) unsigned char'
// expected-error@-2{{unknown type name '__Uint8x8_t'}}

__Uint16x4_t Uint16x4;
// CHECK: Uint16x4 '__Uint16x4_t':'__attribute__((neon_vector_type(4))) unsigned short'
// expected-error@-2{{unknown type name '__Uint16x4_t'}}

__Uint32x2_t Uint32x2;
// CHECK: Uint32x2 '__Uint32x2_t':'__attribute__((neon_vector_type(2))) unsigned int'
// expected-error@-2{{unknown type name '__Uint32x2_t'}}

__Float16x4_t Float16x4;
// CHECK: Float16x4 '__Float16x4_t':'__attribute__((neon_vector_type(4))) __fp16'
// expected-error@-2{{unknown type name '__Float16x4_t'}}

__Float32x2_t Float32x2;
// CHECK: Float32x2 '__Float32x2_t':'__attribute__((neon_vector_type(2))) float'
// expected-error@-2{{unknown type name '__Float32x2_t'}}

__Poly8x8_t Poly8x8;
// CHECK: Poly8x8 '__Poly8x8_t':'__attribute__((neon_polyvector_type(8))) unsigned char'
// expected-error@-2{{unknown type name '__Poly8x8_t'}}

__Poly16x4_t Poly16x4;
// CHECK: Poly16x4 '__Poly16x4_t':'__attribute__((neon_polyvector_type(4))) unsigned short'
// expected-error@-2{{unknown type name '__Poly16x4_t'}}

__Bfloat16x4_t Bfloat16x4;
// CHECK: Bfloat16x4 '__Bfloat16x4_t':'__attribute__((neon_vector_type(4))) __bf16'
// expected-error@-2{{unknown type name '__Bfloat16x4_t'}}

__Int8x16_t Int8x16;
// CHECK: Int8x16 '__Int8x16_t':'__attribute__((neon_vector_type(16))) signed char'
// expected-error@-2{{unknown type name '__Int8x16_t'}}

__Int16x8_t Int16x8;
// CHECK: Int16x8 '__Int16x8_t':'__attribute__((neon_vector_type(8))) short'
// expected-error@-2{{unknown type name '__Int16x8_t'}}

__Int32x4_t Int32x4;
// CHECK: Int32x4 '__Int32x4_t':'__attribute__((neon_vector_type(4))) int'
// expected-error@-2{{unknown type name '__Int32x4_t'}}

__Int64x2_t Int64x2;
// CHECK: Int64x2 '__Int64x2_t':'__attribute__((neon_vector_type(2))) long'
// expected-error@-2{{unknown type name '__Int64x2_t'}}

__Uint8x16_t Uint8x16;
// CHECK: Uint8x16 '__Uint8x16_t':'__attribute__((neon_vector_type(16))) unsigned char'
// expected-error@-2{{unknown type name '__Uint8x16_t'}}

__Uint16x8_t Uint16x8;
// CHECK: Uint16x8 '__Uint16x8_t':'__attribute__((neon_vector_type(8))) unsigned short'
// expected-error@-2{{unknown type name '__Uint16x8_t'}}

__Uint32x4_t Uint32x4;
// CHECK: Uint32x4 '__Uint32x4_t':'__attribute__((neon_vector_type(4))) unsigned int'
// expected-error@-2{{unknown type name '__Uint32x4_t'}}

__Uint64x2_t Uint64x2;
// CHECK: Uint64x2 '__Uint64x2_t':'__attribute__((neon_vector_type(2))) unsigned long'
// expected-error@-2{{unknown type name '__Uint64x2_t'}}

__Float16x8_t Float16x8;
// CHECK: Float16x8 '__Float16x8_t':'__attribute__((neon_vector_type(8))) __fp16'
// expected-error@-2{{unknown type name '__Float16x8_t'}}

__Float32x4_t Float32x4;
// CHECK: Float32x4 '__Float32x4_t':'__attribute__((neon_vector_type(4))) float'
// expected-error@-2{{unknown type name '__Float32x4_t'}}

__Float64x2_t Float64x2;
// CHECK: Float64x2 '__Float64x2_t':'__attribute__((neon_vector_type(2))) double'
// expected-error@-2{{unknown type name '__Float64x2_t'}}

__Poly8x16_t Poly8x16;
// CHECK: Poly8x16 '__Poly8x16_t':'__attribute__((neon_polyvector_type(16))) unsigned char'
// expected-error@-2{{unknown type name '__Poly8x16_t'}}

__Poly16x8_t Poly16x8;
// CHECK: Poly16x8 '__Poly16x8_t':'__attribute__((neon_polyvector_type(8))) unsigned short'
// expected-error@-2{{unknown type name '__Poly16x8_t'}}

__Poly64x2_t Poly64x2;
// CHECK: Poly64x2 '__Poly64x2_t':'__attribute__((neon_polyvector_type(2))) unsigned long'
// expected-error@-2{{unknown type name '__Poly64x2_t'}}

__Bfloat16x8_t Bfloat16x8;
// CHECK: Bfloat16x8 '__Bfloat16x8_t':'__attribute__((neon_vector_type(8))) __bf16'
// expected-error@-2{{unknown type name '__Bfloat16x8_t'}}

__mfp8 mfp8;
// CHECK: mfp8 '__mfp8'
// expected-error@-2{{unknown type name '__mfp8'}}

__Mfloat8x8_t Mfloat8x8;
// CHECK: Mfloat8x8 '__Mfloat8x8_t':'__attribute__((neon_vector_type(8))) __mfp8'
// expected-error@-2{{unknown type name '__Mfloat8x8_t'}}

__Mfloat8x16_t Mfloat8x16;
// CHECK: Mfloat8x16 '__Mfloat8x16_t':'__attribute__((neon_vector_type(16))) __mfp8'
// expected-error@-2{{unknown type name '__Mfloat8x16_t'}}

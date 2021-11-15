// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -mvscale-min=1 -mvscale-max=1 -fallow-half-arguments-and-returns -S -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-128
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -mvscale-min=2 -mvscale-max=2 -fallow-half-arguments-and-returns -S -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-256
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -mvscale-min=4 -mvscale-max=4 -fallow-half-arguments-and-returns -S -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-512
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -mvscale-min=8 -mvscale-max=8 -fallow-half-arguments-and-returns -S -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-1024
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -mvscale-min=16 -mvscale-max=16 -fallow-half-arguments-and-returns -S -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-2048
// RUN: %clang_cc1 -triple aarch64_32-unknown-darwin -target-feature +sve -target-feature +bf16 -mvscale-min=4 -mvscale-max=4 -fallow-half-arguments-and-returns -S -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-ILP32

// REQUIRES: aarch64-registered-target

#include <arm_sve.h>

#define N __ARM_FEATURE_SVE_BITS

typedef svint8_t fixed_int8_t __attribute__((arm_sve_vector_bits(N)));
typedef svint16_t fixed_int16_t __attribute__((arm_sve_vector_bits(N)));
typedef svint32_t fixed_int32_t __attribute__((arm_sve_vector_bits(N)));
typedef svint64_t fixed_int64_t __attribute__((arm_sve_vector_bits(N)));

typedef svuint8_t fixed_uint8_t __attribute__((arm_sve_vector_bits(N)));
typedef svuint16_t fixed_uint16_t __attribute__((arm_sve_vector_bits(N)));
typedef svuint32_t fixed_uint32_t __attribute__((arm_sve_vector_bits(N)));
typedef svuint64_t fixed_uint64_t __attribute__((arm_sve_vector_bits(N)));

typedef svfloat16_t fixed_float16_t __attribute__((arm_sve_vector_bits(N)));
typedef svfloat32_t fixed_float32_t __attribute__((arm_sve_vector_bits(N)));
typedef svfloat64_t fixed_float64_t __attribute__((arm_sve_vector_bits(N)));

typedef svbfloat16_t fixed_bfloat16_t __attribute__((arm_sve_vector_bits(N)));

typedef svbool_t fixed_bool_t __attribute__((arm_sve_vector_bits(N)));

//===----------------------------------------------------------------------===//
// Structs and unions
//===----------------------------------------------------------------------===//
#define DEFINE_STRUCT(ty) \
  struct struct_##ty {    \
    fixed_##ty##_t x;     \
  } struct_##ty;

#define DEFINE_UNION(ty) \
  union union_##ty {     \
    fixed_##ty##_t x;    \
  } union_##ty;

DEFINE_STRUCT(int8)
DEFINE_STRUCT(int16)
DEFINE_STRUCT(int32)
DEFINE_STRUCT(int64)
DEFINE_STRUCT(uint8)
DEFINE_STRUCT(uint16)
DEFINE_STRUCT(uint32)
DEFINE_STRUCT(uint64)
DEFINE_STRUCT(float16)
DEFINE_STRUCT(float32)
DEFINE_STRUCT(float64)
DEFINE_STRUCT(bfloat16)
DEFINE_STRUCT(bool)

DEFINE_UNION(int8)
DEFINE_UNION(int16)
DEFINE_UNION(int32)
DEFINE_UNION(int64)
DEFINE_UNION(uint8)
DEFINE_UNION(uint16)
DEFINE_UNION(uint32)
DEFINE_UNION(uint64)
DEFINE_UNION(float16)
DEFINE_UNION(float32)
DEFINE_UNION(float64)
DEFINE_UNION(bfloat16)
DEFINE_UNION(bool)

//===----------------------------------------------------------------------===//
// Global variables
//===----------------------------------------------------------------------===//
fixed_int8_t global_i8;
fixed_int16_t global_i16;
fixed_int32_t global_i32;
fixed_int64_t global_i64;

fixed_uint8_t global_u8;
fixed_uint16_t global_u16;
fixed_uint32_t global_u32;
fixed_uint64_t global_u64;

fixed_float16_t global_f16;
fixed_float32_t global_f32;
fixed_float64_t global_f64;

fixed_bfloat16_t global_bf16;

fixed_bool_t global_bool;

//===----------------------------------------------------------------------===//
// Global arrays
//===----------------------------------------------------------------------===//
fixed_int8_t global_arr_i8[3];
fixed_int16_t global_arr_i16[3];
fixed_int32_t global_arr_i32[3];
fixed_int64_t global_arr_i64[3];

fixed_uint8_t global_arr_u8[3];
fixed_uint16_t global_arr_u16[3];
fixed_uint32_t global_arr_u32[3];
fixed_uint64_t global_arr_u64[3];

fixed_float16_t global_arr_f16[3];
fixed_float32_t global_arr_f32[3];
fixed_float64_t global_arr_f64[3];

fixed_bfloat16_t global_arr_bf16[3];

fixed_bool_t global_arr_bool[3];

//===----------------------------------------------------------------------===//
// Locals
//===----------------------------------------------------------------------===//
void f() {
  // Variables
  fixed_int8_t local_i8;
  fixed_int16_t local_i16;
  fixed_int32_t local_i32;
  fixed_int64_t local_i64;
  fixed_uint8_t local_u8;
  fixed_uint16_t local_u16;
  fixed_uint32_t local_u32;
  fixed_uint64_t local_u64;
  fixed_float16_t local_f16;
  fixed_float32_t local_f32;
  fixed_float64_t local_f64;
  fixed_bfloat16_t local_bf16;
  fixed_bool_t local_bool;

  // Arrays
  fixed_int8_t local_arr_i8[3];
  fixed_int16_t local_arr_i16[3];
  fixed_int32_t local_arr_i32[3];
  fixed_int64_t local_arr_i64[3];
  fixed_uint8_t local_arr_u8[3];
  fixed_uint16_t local_arr_u16[3];
  fixed_uint32_t local_arr_u32[3];
  fixed_uint64_t local_arr_u64[3];
  fixed_float16_t local_arr_f16[3];
  fixed_float32_t local_arr_f32[3];
  fixed_float64_t local_arr_f64[3];
  fixed_bfloat16_t local_arr_bf16[3];
  fixed_bool_t local_arr_bool[3];
}

//===----------------------------------------------------------------------===//
// Structs and unions
//===----------------------------------------------------------------------===//
// CHECK-128:      %struct.struct_int8 = type { <16 x i8> }
// CHECK-128-NEXT: %struct.struct_int16 = type { <8 x i16> }
// CHECK-128-NEXT: %struct.struct_int32 = type { <4 x i32> }
// CHECK-128-NEXT: %struct.struct_int64 = type { <2 x i64> }
// CHECK-128-NEXT: %struct.struct_uint8 = type { <16 x i8> }
// CHECK-128-NEXT: %struct.struct_uint16 = type { <8 x i16> }
// CHECK-128-NEXT: %struct.struct_uint32 = type { <4 x i32> }
// CHECK-128-NEXT: %struct.struct_uint64 = type { <2 x i64> }
// CHECK-128-NEXT: %struct.struct_float16 = type { <8 x half> }
// CHECK-128-NEXT: %struct.struct_float32 = type { <4 x float> }
// CHECK-128-NEXT: %struct.struct_float64 = type { <2 x double> }
// CHECK-128-NEXT: %struct.struct_bfloat16 = type { <8 x bfloat> }
// CHECK-128-NEXT: %struct.struct_bool = type { <2 x i8> }

// CHECK-256:      %struct.struct_int8 = type { <32 x i8> }
// CHECK-256-NEXT: %struct.struct_int16 = type { <16 x i16> }
// CHECK-256-NEXT: %struct.struct_int32 = type { <8 x i32> }
// CHECK-256-NEXT: %struct.struct_int64 = type { <4 x i64> }
// CHECK-256-NEXT: %struct.struct_uint8 = type { <32 x i8> }
// CHECK-256-NEXT: %struct.struct_uint16 = type { <16 x i16> }
// CHECK-256-NEXT: %struct.struct_uint32 = type { <8 x i32> }
// CHECK-256-NEXT: %struct.struct_uint64 = type { <4 x i64> }
// CHECK-256-NEXT: %struct.struct_float16 = type { <16 x half> }
// CHECK-256-NEXT: %struct.struct_float32 = type { <8 x float> }
// CHECK-256-NEXT: %struct.struct_float64 = type { <4 x double> }
// CHECK-256-NEXT: %struct.struct_bfloat16 = type { <16 x bfloat> }
// CHECK-256-NEXT: %struct.struct_bool = type { <4 x i8> }

// CHECK-512:      %struct.struct_int8 = type { <64 x i8> }
// CHECK-512-NEXT: %struct.struct_int16 = type { <32 x i16> }
// CHECK-512-NEXT: %struct.struct_int32 = type { <16 x i32> }
// CHECK-512-NEXT: %struct.struct_int64 = type { <8 x i64> }
// CHECK-512-NEXT: %struct.struct_uint8 = type { <64 x i8> }
// CHECK-512-NEXT: %struct.struct_uint16 = type { <32 x i16> }
// CHECK-512-NEXT: %struct.struct_uint32 = type { <16 x i32> }
// CHECK-512-NEXT: %struct.struct_uint64 = type { <8 x i64> }
// CHECK-512-NEXT: %struct.struct_float16 = type { <32 x half> }
// CHECK-512-NEXT: %struct.struct_float32 = type { <16 x float> }
// CHECK-512-NEXT: %struct.struct_float64 = type { <8 x double> }
// CHECK-512-NEXT: %struct.struct_bfloat16 = type { <32 x bfloat> }
// CHECK-512-NEXT: %struct.struct_bool = type { <8 x i8> }

// CHECK-1024:      %struct.struct_int8 = type { <128 x i8> }
// CHECK-1024-NEXT: %struct.struct_int16 = type { <64 x i16> }
// CHECK-1024-NEXT: %struct.struct_int32 = type { <32 x i32> }
// CHECK-1024-NEXT: %struct.struct_int64 = type { <16 x i64> }
// CHECK-1024-NEXT: %struct.struct_uint8 = type { <128 x i8> }
// CHECK-1024-NEXT: %struct.struct_uint16 = type { <64 x i16> }
// CHECK-1024-NEXT: %struct.struct_uint32 = type { <32 x i32> }
// CHECK-1024-NEXT: %struct.struct_uint64 = type { <16 x i64> }
// CHECK-1024-NEXT: %struct.struct_float16 = type { <64 x half> }
// CHECK-1024-NEXT: %struct.struct_float32 = type { <32 x float> }
// CHECK-1024-NEXT: %struct.struct_float64 = type { <16 x double> }
// CHECK-1024-NEXT: %struct.struct_bfloat16 = type { <64 x bfloat> }
// CHECK-1024-NEXT: %struct.struct_bool = type { <16 x i8> }

// CHECK-2048:      %struct.struct_int8 = type { <256 x i8> }
// CHECK-2048-NEXT: %struct.struct_int16 = type { <128 x i16> }
// CHECK-2048-NEXT: %struct.struct_int32 = type { <64 x i32> }
// CHECK-2048-NEXT: %struct.struct_int64 = type { <32 x i64> }
// CHECK-2048-NEXT: %struct.struct_uint8 = type { <256 x i8> }
// CHECK-2048-NEXT: %struct.struct_uint16 = type { <128 x i16> }
// CHECK-2048-NEXT: %struct.struct_uint32 = type { <64 x i32> }
// CHECK-2048-NEXT: %struct.struct_uint64 = type { <32 x i64> }
// CHECK-2048-NEXT: %struct.struct_float16 = type { <128 x half> }
// CHECK-2048-NEXT: %struct.struct_float32 = type { <64 x float> }
// CHECK-2048-NEXT: %struct.struct_float64 = type { <32 x double> }
// CHECK-2048-NEXT: %struct.struct_bfloat16 = type { <128 x bfloat> }
// CHECK-2048-NEXT: %struct.struct_bool = type { <32 x i8> }

// CHECK-128:      %union.union_int8 = type { <16 x i8> }
// CHECK-128-NEXT: %union.union_int16 = type { <8 x i16> }
// CHECK-128-NEXT: %union.union_int32 = type { <4 x i32> }
// CHECK-128-NEXT: %union.union_int64 = type { <2 x i64> }
// CHECK-128-NEXT: %union.union_uint8 = type { <16 x i8> }
// CHECK-128-NEXT: %union.union_uint16 = type { <8 x i16> }
// CHECK-128-NEXT: %union.union_uint32 = type { <4 x i32> }
// CHECK-128-NEXT: %union.union_uint64 = type { <2 x i64> }
// CHECK-128-NEXT: %union.union_float16 = type { <8 x half> }
// CHECK-128-NEXT: %union.union_float32 = type { <4 x float> }
// CHECK-128-NEXT: %union.union_float64 = type { <2 x double> }
// CHECK-128-NEXT: %union.union_bfloat16 = type { <8 x bfloat> }
// CHECK-128-NEXT: %union.union_bool = type { <2 x i8> }

// CHECK-256:      %union.union_int8 = type { <32 x i8> }
// CHECK-256-NEXT: %union.union_int16 = type { <16 x i16> }
// CHECK-256-NEXT: %union.union_int32 = type { <8 x i32> }
// CHECK-256-NEXT: %union.union_int64 = type { <4 x i64> }
// CHECK-256-NEXT: %union.union_uint8 = type { <32 x i8> }
// CHECK-256-NEXT: %union.union_uint16 = type { <16 x i16> }
// CHECK-256-NEXT: %union.union_uint32 = type { <8 x i32> }
// CHECK-256-NEXT: %union.union_uint64 = type { <4 x i64> }
// CHECK-256-NEXT: %union.union_float16 = type { <16 x half> }
// CHECK-256-NEXT: %union.union_float32 = type { <8 x float> }
// CHECK-256-NEXT: %union.union_float64 = type { <4 x double> }
// CHECK-256-NEXT: %union.union_bfloat16 = type { <16 x bfloat> }
// CHECK-256-NEXT: %union.union_bool = type { <4 x i8> }

// CHECK-512:      %union.union_int8 = type { <64 x i8> }
// CHECK-512-NEXT: %union.union_int16 = type { <32 x i16> }
// CHECK-512-NEXT: %union.union_int32 = type { <16 x i32> }
// CHECK-512-NEXT: %union.union_int64 = type { <8 x i64> }
// CHECK-512-NEXT: %union.union_uint8 = type { <64 x i8> }
// CHECK-512-NEXT: %union.union_uint16 = type { <32 x i16> }
// CHECK-512-NEXT: %union.union_uint32 = type { <16 x i32> }
// CHECK-512-NEXT: %union.union_uint64 = type { <8 x i64> }
// CHECK-512-NEXT: %union.union_float16 = type { <32 x half> }
// CHECK-512-NEXT: %union.union_float32 = type { <16 x float> }
// CHECK-512-NEXT: %union.union_float64 = type { <8 x double> }
// CHECK-512-NEXT: %union.union_bfloat16 = type { <32 x bfloat> }
// CHECK-512-NEXT: %union.union_bool = type { <8 x i8> }

// CHECK-1024:      %union.union_int8 = type { <128 x i8> }
// CHECK-1024-NEXT: %union.union_int16 = type { <64 x i16> }
// CHECK-1024-NEXT: %union.union_int32 = type { <32 x i32> }
// CHECK-1024-NEXT: %union.union_int64 = type { <16 x i64> }
// CHECK-1024-NEXT: %union.union_uint8 = type { <128 x i8> }
// CHECK-1024-NEXT: %union.union_uint16 = type { <64 x i16> }
// CHECK-1024-NEXT: %union.union_uint32 = type { <32 x i32> }
// CHECK-1024-NEXT: %union.union_uint64 = type { <16 x i64> }
// CHECK-1024-NEXT: %union.union_float16 = type { <64 x half> }
// CHECK-1024-NEXT: %union.union_float32 = type { <32 x float> }
// CHECK-1024-NEXT: %union.union_float64 = type { <16 x double> }
// CHECK-1024-NEXT: %union.union_bfloat16 = type { <64 x bfloat> }
// CHECK-1024-NEXT: %union.union_bool = type { <16 x i8> }

// CHECK-2048:      %union.union_int8 = type { <256 x i8> }
// CHECK-2048-NEXT: %union.union_int16 = type { <128 x i16> }
// CHECK-2048-NEXT: %union.union_int32 = type { <64 x i32> }
// CHECK-2048-NEXT: %union.union_int64 = type { <32 x i64> }
// CHECK-2048-NEXT: %union.union_uint8 = type { <256 x i8> }
// CHECK-2048-NEXT: %union.union_uint16 = type { <128 x i16> }
// CHECK-2048-NEXT: %union.union_uint32 = type { <64 x i32> }
// CHECK-2048-NEXT: %union.union_uint64 = type { <32 x i64> }
// CHECK-2048-NEXT: %union.union_float16 = type { <128 x half> }
// CHECK-2048-NEXT: %union.union_float32 = type { <64 x float> }
// CHECK-2048-NEXT: %union.union_float64 = type { <32 x double> }
// CHECK-2048-NEXT: %union.union_bfloat16 = type { <128 x bfloat> }
// CHECK-2048-NEXT: %union.union_bool = type { <32 x i8> }

//===----------------------------------------------------------------------===//
// Global variables
//===----------------------------------------------------------------------===//
// CHECK-128:      @global_i8 ={{.*}} global <16 x i8> zeroinitializer, align 16
// CHECK-128-NEXT: @global_i16 ={{.*}} global <8 x i16> zeroinitializer, align 16
// CHECK-128-NEXT: @global_i32 ={{.*}} global <4 x i32> zeroinitializer, align 16
// CHECK-128-NEXT: @global_i64 ={{.*}} global <2 x i64> zeroinitializer, align 16
// CHECK-128-NEXT: @global_u8 ={{.*}} global <16 x i8> zeroinitializer, align 16
// CHECK-128-NEXT: @global_u16 ={{.*}} global <8 x i16> zeroinitializer, align 16
// CHECK-128-NEXT: @global_u32 ={{.*}} global <4 x i32> zeroinitializer, align 16
// CHECK-128-NEXT: @global_u64 ={{.*}} global <2 x i64> zeroinitializer, align 16
// CHECK-128-NEXT: @global_f16 ={{.*}} global <8 x half> zeroinitializer, align 16
// CHECK-128-NEXT: @global_f32 ={{.*}} global <4 x float> zeroinitializer, align 16
// CHECK-128-NEXT: @global_f64 ={{.*}} global <2 x double> zeroinitializer, align 16
// CHECK-128-NEXT: @global_bf16 ={{.*}} global <8 x bfloat> zeroinitializer, align 16
// CHECK-128-NEXT: @global_bool ={{.*}} global <2 x i8> zeroinitializer, align 2

// CHECK-256:      @global_i8 ={{.*}} global <32 x i8> zeroinitializer, align 16
// CHECK-NEXT-256: @global_i16 ={{.*}} global <16 x i16> zeroinitializer, align 16
// CHECK-NEXT-256: @global_i32 ={{.*}} global <8 x i32> zeroinitializer, align 16
// CHECK-NEXT-256: @global_i64 ={{.*}} global <4 x i64> zeroinitializer, align 16
// CHECK-NEXT-256: @global_u8 ={{.*}} global <32 x i8> zeroinitializer, align 16
// CHECK-NEXT-256: @global_u16 ={{.*}} global <16 x i16> zeroinitializer, align 16
// CHECK-NEXT-256: @global_u32 ={{.*}} global <8 x i32> zeroinitializer, align 16
// CHECK-NEXT-256: @global_u64 ={{.*}} global <4 x i64> zeroinitializer, align 16
// CHECK-NEXT-256: @global_f16 ={{.*}} global <16 x half> zeroinitializer, align 16
// CHECK-NEXT-256: @global_f32 ={{.*}} global <8 x float> zeroinitializer, align 16
// CHECK-NEXT-256: @global_f64 ={{.*}} global <4 x double> zeroinitializer, align 16
// CHECK-NEXT-256: @global_bf16 ={{.*}} global <16 x bfloat> zeroinitializer, align 16
// CHECK-NEXT-256: @global_bool ={{.*}} global <4 x i8> zeroinitializer, align 2

// CHECK-512:      @global_i8 ={{.*}} global <64 x i8> zeroinitializer, align 16
// CHECK-NEXT-512: @global_i16 ={{.*}} global <32 x i16> zeroinitializer, align 16
// CHECK-NEXT-512: @global_i32 ={{.*}} global <16 x i32> zeroinitializer, align 16
// CHECK-NEXT-512: @global_i64 ={{.*}} global <8 x i64> zeroinitializer, align 16
// CHECK-NEXT-512: @global_u8 ={{.*}} global <64 x i8> zeroinitializer, align 16
// CHECK-NEXT-512: @global_u16 ={{.*}} global <32 x i16> zeroinitializer, align 16
// CHECK-NEXT-512: @global_u32 ={{.*}} global <16 x i32> zeroinitializer, align 16
// CHECK-NEXT-512: @global_u64 ={{.*}} global <8 x i64> zeroinitializer, align 16
// CHECK-NEXT-512: @global_f16 ={{.*}} global <32 x half> zeroinitializer, align 16
// CHECK-NEXT-512: @global_f32 ={{.*}} global <16 x float> zeroinitializer, align 16
// CHECK-NEXT-512: @global_f64 ={{.*}} global <8 x double> zeroinitializer, align 16
// CHECK-NEXT-512: @global_bf16 ={{.*}} global <32 x bfloat> zeroinitializer, align 16
// CHECK-NEXT-512: @global_bool ={{.*}} global <8 x i8> zeroinitializer, align 2

// CHECK-1024:      @global_i8 ={{.*}} global <128 x i8> zeroinitializer, align 16
// CHECK-NEXT-1024: @global_i16 ={{.*}} global <64 x i16> zeroinitializer, align 16
// CHECK-NEXT-1024: @global_i32 ={{.*}} global <32 x i32> zeroinitializer, align 16
// CHECK-NEXT-1024: @global_i64 ={{.*}} global <16 x i64> zeroinitializer, align 16
// CHECK-NEXT-1024: @global_u8 ={{.*}} global <128 x i8> zeroinitializer, align 16
// CHECK-NEXT-1024: @global_u16 ={{.*}} global <64 x i16> zeroinitializer, align 16
// CHECK-NEXT-1024: @global_u32 ={{.*}} global <32 x i32> zeroinitializer, align 16
// CHECK-NEXT-1024: @global_u64 ={{.*}} global <16 x i64> zeroinitializer, align 16
// CHECK-NEXT-1024: @global_f16 ={{.*}} global <64 x half> zeroinitializer, align 16
// CHECK-NEXT-1024: @global_f32 ={{.*}} global <32 x float> zeroinitializer, align 16
// CHECK-NEXT-1024: @global_f64 ={{.*}} global <16 x double> zeroinitializer, align 16
// CHECK-NEXT-1024: @global_bf16 ={{.*}} global <64 x bfloat> zeroinitializer, align 16
// CHECK-NEXT-1024: @global_bool ={{.*}} global <16 x i8> zeroinitializer, align 2

// CHECK-2048:      @global_i8 ={{.*}} global <256 x i8> zeroinitializer, align 16
// CHECK-NEXT-2048: @global_i16 ={{.*}} global <128 x i16> zeroinitializer, align 16
// CHECK-NEXT-2048: @global_i32 ={{.*}} global <64 x i32> zeroinitializer, align 16
// CHECK-NEXT-2048: @global_i64 ={{.*}} global <32 x i64> zeroinitializer, align 16
// CHECK-NEXT-2048: @global_u8 ={{.*}} global <256 x i8> zeroinitializer, align 16
// CHECK-NEXT-2048: @global_u16 ={{.*}} global <128 x i16> zeroinitializer, align 16
// CHECK-NEXT-2048: @global_u32 ={{.*}} global <64 x i32> zeroinitializer, align 16
// CHECK-NEXT-2048: @global_u64 ={{.*}} global <32 x i64> zeroinitializer, align 16
// CHECK-NEXT-2048: @global_f16 ={{.*}} global <128 x half> zeroinitializer, align 16
// CHECK-NEXT-2048: @global_f32 ={{.*}} global <64 x float> zeroinitializer, align 16
// CHECK-NEXT-2048: @global_f64 ={{.*}} global <32 x double> zeroinitializer, align 16
// CHECK-NEXT-2048: @global_bf16 ={{.*}} global <128 x bfloat> zeroinitializer, align 16
// CHECK-NEXT-2048: @global_bool ={{.*}} global <32 x i8> zeroinitializer, align 2

//===----------------------------------------------------------------------===//
// Global arrays
//===----------------------------------------------------------------------===//
// CHECK-128:      @global_arr_i8 ={{.*}} global [3 x <16 x i8>] zeroinitializer, align 16
// CHECK-128-NEXT: @global_arr_i16 ={{.*}} global [3 x <8 x i16>] zeroinitializer, align 16
// CHECK-128-NEXT: @global_arr_i32 ={{.*}} global [3 x <4 x i32>] zeroinitializer, align 16
// CHECK-128-NEXT: @global_arr_i64 ={{.*}} global [3 x <2 x i64>] zeroinitializer, align 16
// CHECK-128-NEXT: @global_arr_u8 ={{.*}} global [3 x <16 x i8>] zeroinitializer, align 16
// CHECK-128-NEXT: @global_arr_u16 ={{.*}} global [3 x <8 x i16>] zeroinitializer, align 16
// CHECK-128-NEXT: @global_arr_u32 ={{.*}} global [3 x <4 x i32>] zeroinitializer, align 16
// CHECK-128-NEXT: @global_arr_u64 ={{.*}} global [3 x <2 x i64>] zeroinitializer, align 16
// CHECK-128-NEXT: @global_arr_f16 ={{.*}} global [3 x <8 x half>] zeroinitializer, align 16
// CHECK-128-NEXT: @global_arr_f32 ={{.*}} global [3 x <4 x float>] zeroinitializer, align 16
// CHECK-128-NEXT: @global_arr_f64 ={{.*}} global [3 x <2 x double>] zeroinitializer, align 16
// CHECK-128-NEXT: @global_arr_bf16 ={{.*}} global [3 x <8 x bfloat>] zeroinitializer, align 16
// CHECK-128-NEXT: @global_arr_bool ={{.*}} global [3 x <2 x i8>] zeroinitializer, align 2

// CHECK-256:      @global_arr_i8 ={{.*}} global [3 x <32 x i8>] zeroinitializer, align 16
// CHECK-NEXT-256: @global_arr_i16 ={{.*}} global [3 x <16 x i16>] zeroinitializer, align 16
// CHECK-NEXT-256: @global_arr_i32 ={{.*}} global [3 x <8 x i32>] zeroinitializer, align 16
// CHECK-NEXT-256: @global_arr_i64 ={{.*}} global [3 x <4 x i64>] zeroinitializer, align 16
// CHECK-NEXT-256: @global_arr_u8 ={{.*}} global [3 x <32 x i8>] zeroinitializer, align 16
// CHECK-NEXT-256: @global_arr_u16 ={{.*}} global [3 x <16 x i16>] zeroinitializer, align 16
// CHECK-NEXT-256: @global_arr_u32 ={{.*}} global [3 x <8 x i32>] zeroinitializer, align 16
// CHECK-NEXT-256: @global_arr_u64 ={{.*}} global [3 x <4 x i64>] zeroinitializer, align 16
// CHECK-NEXT-256: @global_arr_f16 ={{.*}} global [3 x <16 x half>] zeroinitializer, align 16
// CHECK-NEXT-256: @global_arr_f32 ={{.*}} global [3 x <8 x float>] zeroinitializer, align 16
// CHECK-NEXT-256: @global_arr_f64 ={{.*}} global [3 x <4 x double>] zeroinitializer, align 16
// CHECK-NEXT-256: @global_arr_bf16 ={{.*}} global [3 x <16 x bfloat>] zeroinitializer, align 16
// CHECK-NEXT-256: @global_arr_bool ={{.*}} global [3 x <4 x i8>] zeroinitializer, align 2

// CHECK-512:      @global_arr_i8 ={{.*}} global [3 x <64 x i8>] zeroinitializer, align 16
// CHECK-NEXT-512: @global_arr_i16 ={{.*}} global [3 x <32 x i16>] zeroinitializer, align 16
// CHECK-NEXT-512: @global_arr_i32 ={{.*}} global [3 x <16 x i32>] zeroinitializer, align 16
// CHECK-NEXT-512: @global_arr_i64 ={{.*}} global [3 x <8 x i64>] zeroinitializer, align 16
// CHECK-NEXT-512: @global_arr_u8 ={{.*}} global [3 x <64 x i8>] zeroinitializer, align 16
// CHECK-NEXT-512: @global_arr_u16 ={{.*}} global [3 x <32 x i16>] zeroinitializer, align 16
// CHECK-NEXT-512: @global_arr_u32 ={{.*}} global [3 x <16 x i32>] zeroinitializer, align 16
// CHECK-NEXT-512: @global_arr_u64 ={{.*}} global [3 x <8 x i64>] zeroinitializer, align 16
// CHECK-NEXT-512: @global_arr_f16 ={{.*}} global [3 x <32 x half>] zeroinitializer, align 16
// CHECK-NEXT-512: @global_arr_f32 ={{.*}} global [3 x <16 x float>] zeroinitializer, align 16
// CHECK-NEXT-512: @global_arr_f64 ={{.*}} global [3 x <8 x double>] zeroinitializer, align 16
// CHECK-NEXT-512: @global_arr_bf16 ={{.*}} global [3 x <32 x bfloat>] zeroinitializer, align 16
// CHECK-NEXT-512: @global_arr_bool ={{.*}} global [3 x <8 x i8>] zeroinitializer, align 2

// CHECK-1024:      @global_arr_i8 ={{.*}} global [3 x <128 x i8>] zeroinitializer, align 16
// CHECK-NEXT-1024: @global_arr_i16 ={{.*}} global [3 x <64 x i16>] zeroinitializer, align 16
// CHECK-NEXT-1024: @global_arr_i32 ={{.*}} global [3 x <32 x i32>] zeroinitializer, align 16
// CHECK-NEXT-1024: @global_arr_i64 ={{.*}} global [3 x <16 x i64>] zeroinitializer, align 16
// CHECK-NEXT-1024: @global_arr_u8 ={{.*}} global [3 x <128 x i8>] zeroinitializer, align 16
// CHECK-NEXT-1024: @global_arr_u16 ={{.*}} global [3 x <64 x i16>] zeroinitializer, align 16
// CHECK-NEXT-1024: @global_arr_u32 ={{.*}} global [3 x <32 x i32>] zeroinitializer, align 16
// CHECK-NEXT-1024: @global_arr_u64 ={{.*}} global [3 x <16 x i64>] zeroinitializer, align 16
// CHECK-NEXT-1024: @global_arr_f16 ={{.*}} global [3 x <64 x half>] zeroinitializer, align 16
// CHECK-NEXT-1024: @global_arr_f32 ={{.*}} global [3 x <32 x float>] zeroinitializer, align 16
// CHECK-NEXT-1024: @global_arr_f64 ={{.*}} global [3 x <16 x double>] zeroinitializer, align 16
// CHECK-NEXT-1024: @global_arr_bf16 ={{.*}} global [3 x <64 x bfloat>] zeroinitializer, align 16
// CHECK-NEXT-1024: @global_arr_bool ={{.*}} global [3 x <16 x i8>] zeroinitializer, align 2

// CHECK-2048:      @global_arr_i8 ={{.*}} global [3 x <256 x i8>] zeroinitializer, align 16
// CHECK-NEXT-2048: @global_arr_i16 ={{.*}} global [3 x <128 x i16>] zeroinitializer, align 16
// CHECK-NEXT-2048: @global_arr_i32 ={{.*}} global [3 x <64 x i32>] zeroinitializer, align 16
// CHECK-NEXT-2048: @global_arr_i64 ={{.*}} global [3 x <32 x i64>] zeroinitializer, align 16
// CHECK-NEXT-2048: @global_arr_u8 ={{.*}} global [3 x <256 x i8>] zeroinitializer, align 16
// CHECK-NEXT-2048: @global_arr_u16 ={{.*}} global [3 x <128 x i16>] zeroinitializer, align 16
// CHECK-NEXT-2048: @global_arr_u32 ={{.*}} global [3 x <64 x i32>] zeroinitializer, align 16
// CHECK-NEXT-2048: @global_arr_u64 ={{.*}} global [3 x <32 x i64>] zeroinitializer, align 16
// CHECK-NEXT-2048: @global_arr_f16 ={{.*}} global [3 x <128 x half>] zeroinitializer, align 16
// CHECK-NEXT-2048: @global_arr_f32 ={{.*}} global [3 x <64 x float>] zeroinitializer, align 16
// CHECK-NEXT-2048: @global_arr_f64 ={{.*}} global [3 x <32 x double>] zeroinitializer, align 16
// CHECK-NEXT-2048: @global_arr_bf16 ={{.*}} global [3 x <128 x bfloat>] zeroinitializer, align 16
// CHECK-NEXT-2048: @global_arr_bool ={{.*}} global [3 x <32 x i8>] zeroinitializer, align 2

//===----------------------------------------------------------------------===//
// Local variables
//===----------------------------------------------------------------------===//
// CHECK-128:      %local_i8 = alloca <16 x i8>, align 16
// CHECK-128-NEXT: %local_i16 = alloca <8 x i16>, align 16
// CHECK-128-NEXT: %local_i32 = alloca <4 x i32>, align 16
// CHECK-128-NEXT: %local_i64 = alloca <2 x i64>, align 16
// CHECK-128-NEXT: %local_u8 = alloca <16 x i8>, align 16
// CHECK-128-NEXT: %local_u16 = alloca <8 x i16>, align 16
// CHECK-128-NEXT: %local_u32 = alloca <4 x i32>, align 16
// CHECK-128-NEXT: %local_u64 = alloca <2 x i64>, align 16
// CHECK-128-NEXT: %local_f16 = alloca <8 x half>, align 16
// CHECK-128-NEXT: %local_f32 = alloca <4 x float>, align 16
// CHECK-128-NEXT: %local_f64 = alloca <2 x double>, align 16
// CHECK-128-NEXT: %local_bf16 = alloca <8 x bfloat>, align 16
// CHECK-128-NEXT: %local_bool = alloca <2 x i8>, align 2

// CHECK-256:      %local_i8 = alloca <32 x i8>, align 16
// CHECK-256-NEXT: %local_i16 = alloca <16 x i16>, align 16
// CHECK-256-NEXT: %local_i32 = alloca <8 x i32>, align 16
// CHECK-256-NEXT: %local_i64 = alloca <4 x i64>, align 16
// CHECK-256-NEXT: %local_u8 = alloca <32 x i8>, align 16
// CHECK-256-NEXT: %local_u16 = alloca <16 x i16>, align 16
// CHECK-256-NEXT: %local_u32 = alloca <8 x i32>, align 16
// CHECK-256-NEXT: %local_u64 = alloca <4 x i64>, align 16
// CHECK-256-NEXT: %local_f16 = alloca <16 x half>, align 16
// CHECK-256-NEXT: %local_f32 = alloca <8 x float>, align 16
// CHECK-256-NEXT: %local_f64 = alloca <4 x double>, align 16
// CHECK-256-NEXT: %local_bf16 = alloca <16 x bfloat>, align 16
// CHECK-256-NEXT: %local_bool = alloca <4 x i8>, align 2

// CHECK-512:      %local_i8 = alloca <64 x i8>, align 16
// CHECK-512-NEXT: %local_i16 = alloca <32 x i16>, align 16
// CHECK-512-NEXT: %local_i32 = alloca <16 x i32>, align 16
// CHECK-512-NEXT: %local_i64 = alloca <8 x i64>, align 16
// CHECK-512-NEXT: %local_u8 = alloca <64 x i8>, align 16
// CHECK-512-NEXT: %local_u16 = alloca <32 x i16>, align 16
// CHECK-512-NEXT: %local_u32 = alloca <16 x i32>, align 16
// CHECK-512-NEXT: %local_u64 = alloca <8 x i64>, align 16
// CHECK-512-NEXT: %local_f16 = alloca <32 x half>, align 16
// CHECK-512-NEXT: %local_f32 = alloca <16 x float>, align 16
// CHECK-512-NEXT: %local_f64 = alloca <8 x double>, align 16
// CHECK-512-NEXT: %local_bf16 = alloca <32 x bfloat>, align 16
// CHECK-512-NEXT: %local_bool = alloca <8 x i8>, align 2

// CHECK-1024:       %local_i8 = alloca <128 x i8>, align 16
// CHECK-1024-NEXT:  %local_i16 = alloca <64 x i16>, align 16
// CHECK-1024-NEXT:  %local_i32 = alloca <32 x i32>, align 16
// CHECK-1024-NEXT:  %local_i64 = alloca <16 x i64>, align 16
// CHECK-1024-NEXT:  %local_u8 = alloca <128 x i8>, align 16
// CHECK-1024-NEXT:  %local_u16 = alloca <64 x i16>, align 16
// CHECK-1024-NEXT:  %local_u32 = alloca <32 x i32>, align 16
// CHECK-1024-NEXT:  %local_u64 = alloca <16 x i64>, align 16
// CHECK-1024-NEXT:  %local_f16 = alloca <64 x half>, align 16
// CHECK-1024-NEXT:  %local_f32 = alloca <32 x float>, align 16
// CHECK-1024-NEXT:  %local_f64 = alloca <16 x double>, align 16
// CHECK-1024-NEXT:  %local_bf16 = alloca <64 x bfloat>, align 16
// CHECK-1024-NEXT:  %local_bool = alloca <16 x i8>, align 2

// CHECK-2048:       %local_i8 = alloca <256 x i8>, align 16
// CHECK-2048-NEXT:  %local_i16 = alloca <128 x i16>, align 16
// CHECK-2048-NEXT:  %local_i32 = alloca <64 x i32>, align 16
// CHECK-2048-NEXT:  %local_i64 = alloca <32 x i64>, align 16
// CHECK-2048-NEXT:  %local_u8 = alloca <256 x i8>, align 16
// CHECK-2048-NEXT:  %local_u16 = alloca <128 x i16>, align 16
// CHECK-2048-NEXT:  %local_u32 = alloca <64 x i32>, align 16
// CHECK-2048-NEXT:  %local_u64 = alloca <32 x i64>, align 16
// CHECK-2048-NEXT:  %local_f16 = alloca <128 x half>, align 16
// CHECK-2048-NEXT:  %local_f32 = alloca <64 x float>, align 16
// CHECK-2048-NEXT:  %local_f64 = alloca <32 x double>, align 16
// CHECK-2048-NEXT:  %local_bf16 = alloca <128 x bfloat>, align 16
// CHECK-2048-NEXT:  %local_bool = alloca <32 x i8>, align 2

//===----------------------------------------------------------------------===//
// Local arrays
//===----------------------------------------------------------------------===//
// CHECK-128:      %local_arr_i8 = alloca [3 x <16 x i8>], align 16
// CHECK-128-NEXT: %local_arr_i16 = alloca [3 x <8 x i16>], align 16
// CHECK-128-NEXT: %local_arr_i32 = alloca [3 x <4 x i32>], align 16
// CHECK-128-NEXT: %local_arr_i64 = alloca [3 x <2 x i64>], align 16
// CHECK-128-NEXT: %local_arr_u8 = alloca [3 x <16 x i8>], align 16
// CHECK-128-NEXT: %local_arr_u16 = alloca [3 x <8 x i16>], align 16
// CHECK-128-NEXT: %local_arr_u32 = alloca [3 x <4 x i32>], align 16
// CHECK-128-NEXT: %local_arr_u64 = alloca [3 x <2 x i64>], align 16
// CHECK-128-NEXT: %local_arr_f16 = alloca [3 x <8 x half>], align 16
// CHECK-128-NEXT: %local_arr_f32 = alloca [3 x <4 x float>], align 16
// CHECK-128-NEXT: %local_arr_f64 = alloca [3 x <2 x double>], align 16
// CHECK-128-NEXT: %local_arr_bf16 = alloca [3 x <8 x bfloat>], align 16
// CHECK-128-NEXT: %local_arr_bool = alloca [3 x <2 x i8>], align 2

// CHECK-256:      %local_arr_i8 = alloca [3 x <32 x i8>], align 16
// CHECK-256-NEXT: %local_arr_i16 = alloca [3 x <16 x i16>], align 16
// CHECK-256-NEXT: %local_arr_i32 = alloca [3 x <8 x i32>], align 16
// CHECK-256-NEXT: %local_arr_i64 = alloca [3 x <4 x i64>], align 16
// CHECK-256-NEXT: %local_arr_u8 = alloca [3 x <32 x i8>], align 16
// CHECK-256-NEXT: %local_arr_u16 = alloca [3 x <16 x i16>], align 16
// CHECK-256-NEXT: %local_arr_u32 = alloca [3 x <8 x i32>], align 16
// CHECK-256-NEXT: %local_arr_u64 = alloca [3 x <4 x i64>], align 16
// CHECK-256-NEXT: %local_arr_f16 = alloca [3 x <16 x half>], align 16
// CHECK-256-NEXT: %local_arr_f32 = alloca [3 x <8 x float>], align 16
// CHECK-256-NEXT: %local_arr_f64 = alloca [3 x <4 x double>], align 16
// CHECK-256-NEXT: %local_arr_bf16 = alloca [3 x <16 x bfloat>], align 16
// CHECK-256-NEXT: %local_arr_bool = alloca [3 x <4 x i8>], align 2

// CHECK-512:      %local_arr_i8 = alloca [3 x <64 x i8>], align 16
// CHECK-512-NEXT: %local_arr_i16 = alloca [3 x <32 x i16>], align 16
// CHECK-512-NEXT: %local_arr_i32 = alloca [3 x <16 x i32>], align 16
// CHECK-512-NEXT: %local_arr_i64 = alloca [3 x <8 x i64>], align 16
// CHECK-512-NEXT: %local_arr_u8 = alloca [3 x <64 x i8>], align 16
// CHECK-512-NEXT: %local_arr_u16 = alloca [3 x <32 x i16>], align 16
// CHECK-512-NEXT: %local_arr_u32 = alloca [3 x <16 x i32>], align 16
// CHECK-512-NEXT: %local_arr_u64 = alloca [3 x <8 x i64>], align 16
// CHECK-512-NEXT: %local_arr_f16 = alloca [3 x <32 x half>], align 16
// CHECK-512-NEXT: %local_arr_f32 = alloca [3 x <16 x float>], align 16
// CHECK-512-NEXT: %local_arr_f64 = alloca [3 x <8 x double>], align 16
// CHECK-512-NEXT: %local_arr_bf16 = alloca [3 x <32 x bfloat>], align 16
// CHECK-512-NEXT: %local_arr_bool = alloca [3 x <8 x i8>], align 2

// CHECK-1024:       %local_arr_i8 = alloca [3 x <128 x i8>], align 16
// CHECK-1024-NEXT:  %local_arr_i16 = alloca [3 x <64 x i16>], align 16
// CHECK-1024-NEXT:  %local_arr_i32 = alloca [3 x <32 x i32>], align 16
// CHECK-1024-NEXT:  %local_arr_i64 = alloca [3 x <16 x i64>], align 16
// CHECK-1024-NEXT:  %local_arr_u8 = alloca [3 x <128 x i8>], align 16
// CHECK-1024-NEXT:  %local_arr_u16 = alloca [3 x <64 x i16>], align 16
// CHECK-1024-NEXT:  %local_arr_u32 = alloca [3 x <32 x i32>], align 16
// CHECK-1024-NEXT:  %local_arr_u64 = alloca [3 x <16 x i64>], align 16
// CHECK-1024-NEXT:  %local_arr_f16 = alloca [3 x <64 x half>], align 16
// CHECK-1024-NEXT:  %local_arr_f32 = alloca [3 x <32 x float>], align 16
// CHECK-1024-NEXT:  %local_arr_f64 = alloca [3 x <16 x double>], align 16
// CHECK-1024-NEXT:  %local_arr_bf16 = alloca [3 x <64 x bfloat>], align 16
// CHECK-1024-NEXT:  %local_arr_bool = alloca [3 x <16 x i8>], align 2

// CHECK-2048:       %local_arr_i8 = alloca [3 x <256 x i8>], align 16
// CHECK-2048-NEXT:  %local_arr_i16 = alloca [3 x <128 x i16>], align 16
// CHECK-2048-NEXT:  %local_arr_i32 = alloca [3 x <64 x i32>], align 16
// CHECK-2048-NEXT:  %local_arr_i64 = alloca [3 x <32 x i64>], align 16
// CHECK-2048-NEXT:  %local_arr_u8 = alloca [3 x <256 x i8>], align 16
// CHECK-2048-NEXT:  %local_arr_u16 = alloca [3 x <128 x i16>], align 16
// CHECK-2048-NEXT:  %local_arr_u32 = alloca [3 x <64 x i32>], align 16
// CHECK-2048-NEXT:  %local_arr_u64 = alloca [3 x <32 x i64>], align 16
// CHECK-2048-NEXT:  %local_arr_f16 = alloca [3 x <128 x half>], align 16
// CHECK-2048-NEXT:  %local_arr_f32 = alloca [3 x <64 x float>], align 16
// CHECK-2048-NEXT:  %local_arr_f64 = alloca [3 x <32 x double>], align 16
// CHECK-2048-NEXT:  %local_arr_bf16 = alloca [3 x <128 x bfloat>], align 16
// CHECK-2048-NEXT:  %local_arr_bool = alloca [3 x <32 x i8>], align 2

//===----------------------------------------------------------------------===//
// ILP32 ABI
//===----------------------------------------------------------------------===//
// CHECK-ILP32: @global_i32 ={{.*}} global <16 x i32> zeroinitializer, align 16
// CHECK-ILP32: @global_i64 ={{.*}} global <8 x i64> zeroinitializer, align 16
// CHECK-ILP32: @global_u32 ={{.*}} global <16 x i32> zeroinitializer, align 16
// CHECK-ILP32: @global_u64 ={{.*}} global <8 x i64> zeroinitializer, align 16

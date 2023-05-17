// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +f -target-feature +d -target-feature +zve64d -mvscale-min=1 -mvscale-max=1 -S -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-64
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +f -target-feature +d -target-feature +zve64d -mvscale-min=2 -mvscale-max=2 -S -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-128
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +f -target-feature +d -target-feature +zve64d -mvscale-min=4 -mvscale-max=4 -S -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-256
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +f -target-feature +d -target-feature +zve64d -mvscale-min=8 -mvscale-max=8 -S -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-512
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +f -target-feature +d -target-feature +zve64d -mvscale-min=16 -mvscale-max=16 -S -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-1024

// REQUIRES: riscv-registered-target

#include <stdint.h>

typedef __rvv_int8m1_t vint8m1_t;
typedef __rvv_uint8m1_t vuint8m1_t;
typedef __rvv_int16m1_t vint16m1_t;
typedef __rvv_uint16m1_t vuint16m1_t;
typedef __rvv_int32m1_t vint32m1_t;
typedef __rvv_uint32m1_t vuint32m1_t;
typedef __rvv_int64m1_t vint64m1_t;
typedef __rvv_uint64m1_t vuint64m1_t;
typedef __rvv_float32m1_t vfloat32m1_t;
typedef __rvv_float64m1_t vfloat64m1_t;

// Define valid fixed-width RVV types
typedef vint8m1_t fixed_int8m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));
typedef vint16m1_t fixed_int16m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));
typedef vint32m1_t fixed_int32m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));
typedef vint64m1_t fixed_int64m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));

typedef vuint8m1_t fixed_uint8m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));
typedef vuint16m1_t fixed_uint16m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));
typedef vuint32m1_t fixed_uint32m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));
typedef vuint64m1_t fixed_uint64m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));

typedef vfloat32m1_t fixed_float32m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));
typedef vfloat64m1_t fixed_float64m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));

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

DEFINE_STRUCT(int8m1)
DEFINE_STRUCT(int16m1)
DEFINE_STRUCT(int32m1)
DEFINE_STRUCT(int64m1)
DEFINE_STRUCT(uint8m1)
DEFINE_STRUCT(uint16m1)
DEFINE_STRUCT(uint32m1)
DEFINE_STRUCT(uint64m1)
DEFINE_STRUCT(float32m1)
DEFINE_STRUCT(float64m1)

DEFINE_UNION(int8m1)
DEFINE_UNION(int16m1)
DEFINE_UNION(int32m1)
DEFINE_UNION(int64m1)
DEFINE_UNION(uint8m1)
DEFINE_UNION(uint16m1)
DEFINE_UNION(uint32m1)
DEFINE_UNION(uint64m1)
DEFINE_UNION(float32m1)
DEFINE_UNION(float64m1)

//===----------------------------------------------------------------------===//
// Global variables
//===----------------------------------------------------------------------===//
fixed_int8m1_t global_i8;
fixed_int16m1_t global_i16;
fixed_int32m1_t global_i32;
fixed_int64m1_t global_i64;

fixed_uint8m1_t global_u8;
fixed_uint16m1_t global_u16;
fixed_uint32m1_t global_u32;
fixed_uint64m1_t global_u64;

fixed_float32m1_t global_f32;
fixed_float64m1_t global_f64;

//===----------------------------------------------------------------------===//
// Global arrays
//===----------------------------------------------------------------------===//
fixed_int8m1_t global_arr_i8[3];
fixed_int16m1_t global_arr_i16[3];
fixed_int32m1_t global_arr_i32[3];
fixed_int64m1_t global_arr_i64[3];

fixed_uint8m1_t global_arr_u8[3];
fixed_uint16m1_t global_arr_u16[3];
fixed_uint32m1_t global_arr_u32[3];
fixed_uint64m1_t global_arr_u64[3];

fixed_float32m1_t global_arr_f32[3];
fixed_float64m1_t global_arr_f64[3];

//===----------------------------------------------------------------------===//
// Locals
//===----------------------------------------------------------------------===//
void f() {
  // Variables
  fixed_int8m1_t local_i8;
  fixed_int16m1_t local_i16;
  fixed_int32m1_t local_i32;
  fixed_int64m1_t local_i64;
  fixed_uint8m1_t local_u8;
  fixed_uint16m1_t local_u16;
  fixed_uint32m1_t local_u32;
  fixed_uint64m1_t local_u64;
  fixed_float32m1_t local_f32;
  fixed_float64m1_t local_f64;

  // Arrays
  fixed_int8m1_t local_arr_i8[3];
  fixed_int16m1_t local_arr_i16[3];
  fixed_int32m1_t local_arr_i32[3];
  fixed_int64m1_t local_arr_i64[3];
  fixed_uint8m1_t local_arr_u8[3];
  fixed_uint16m1_t local_arr_u16[3];
  fixed_uint32m1_t local_arr_u32[3];
  fixed_uint64m1_t local_arr_u64[3];
  fixed_float32m1_t local_arr_f32[3];
  fixed_float64m1_t local_arr_f64[3];
}

//===----------------------------------------------------------------------===//
// Structs and unions
//===----------------------------------------------------------------------===//
// CHECK-64:      %struct.struct_int8m1 = type { <8 x i8> }
// CHECK-64-NEXT: %struct.struct_int16m1 = type { <4 x i16> }
// CHECK-64-NEXT: %struct.struct_int32m1 = type { <2 x i32> }
// CHECK-64-NEXT: %struct.struct_int64m1 = type { <1 x i64> }
// CHECK-64-NEXT: %struct.struct_uint8m1 = type { <8 x i8> }
// CHECK-64-NEXT: %struct.struct_uint16m1 = type { <4 x i16> }
// CHECK-64-NEXT: %struct.struct_uint32m1 = type { <2 x i32> }
// CHECK-64-NEXT: %struct.struct_uint64m1 = type { <1 x i64> }
// CHECK-64-NEXT: %struct.struct_float32m1 = type { <2 x float> }
// CHECK-64-NEXT: %struct.struct_float64m1 = type { <1 x double> }

// CHECK-128:      %struct.struct_int8m1 = type { <16 x i8> }
// CHECK-128-NEXT: %struct.struct_int16m1 = type { <8 x i16> }
// CHECK-128-NEXT: %struct.struct_int32m1 = type { <4 x i32> }
// CHECK-128-NEXT: %struct.struct_int64m1 = type { <2 x i64> }
// CHECK-128-NEXT: %struct.struct_uint8m1 = type { <16 x i8> }
// CHECK-128-NEXT: %struct.struct_uint16m1 = type { <8 x i16> }
// CHECK-128-NEXT: %struct.struct_uint32m1 = type { <4 x i32> }
// CHECK-128-NEXT: %struct.struct_uint64m1 = type { <2 x i64> }
// CHECK-128-NEXT: %struct.struct_float32m1 = type { <4 x float> }
// CHECK-128-NEXT: %struct.struct_float64m1 = type { <2 x double> }

// CHECK-256:      %struct.struct_int8m1 = type { <32 x i8> }
// CHECK-256-NEXT: %struct.struct_int16m1 = type { <16 x i16> }
// CHECK-256-NEXT: %struct.struct_int32m1 = type { <8 x i32> }
// CHECK-256-NEXT: %struct.struct_int64m1 = type { <4 x i64> }
// CHECK-256-NEXT: %struct.struct_uint8m1 = type { <32 x i8> }
// CHECK-256-NEXT: %struct.struct_uint16m1 = type { <16 x i16> }
// CHECK-256-NEXT: %struct.struct_uint32m1 = type { <8 x i32> }
// CHECK-256-NEXT: %struct.struct_uint64m1 = type { <4 x i64> }
// CHECK-256-NEXT: %struct.struct_float32m1 = type { <8 x float> }
// CHECK-256-NEXT: %struct.struct_float64m1 = type { <4 x double> }

// CHECK-512:      %struct.struct_int8m1 = type { <64 x i8> }
// CHECK-512-NEXT: %struct.struct_int16m1 = type { <32 x i16> }
// CHECK-512-NEXT: %struct.struct_int32m1 = type { <16 x i32> }
// CHECK-512-NEXT: %struct.struct_int64m1 = type { <8 x i64> }
// CHECK-512-NEXT: %struct.struct_uint8m1 = type { <64 x i8> }
// CHECK-512-NEXT: %struct.struct_uint16m1 = type { <32 x i16> }
// CHECK-512-NEXT: %struct.struct_uint32m1 = type { <16 x i32> }
// CHECK-512-NEXT: %struct.struct_uint64m1 = type { <8 x i64> }
// CHECK-512-NEXT: %struct.struct_float32m1 = type { <16 x float> }
// CHECK-512-NEXT: %struct.struct_float64m1 = type { <8 x double> }

// CHECK-1024:      %struct.struct_int8m1 = type { <128 x i8> }
// CHECK-1024-NEXT: %struct.struct_int16m1 = type { <64 x i16> }
// CHECK-1024-NEXT: %struct.struct_int32m1 = type { <32 x i32> }
// CHECK-1024-NEXT: %struct.struct_int64m1 = type { <16 x i64> }
// CHECK-1024-NEXT: %struct.struct_uint8m1 = type { <128 x i8> }
// CHECK-1024-NEXT: %struct.struct_uint16m1 = type { <64 x i16> }
// CHECK-1024-NEXT: %struct.struct_uint32m1 = type { <32 x i32> }
// CHECK-1024-NEXT: %struct.struct_uint64m1 = type { <16 x i64> }
// CHECK-1024-NEXT: %struct.struct_float32m1 = type { <32 x float> }
// CHECK-1024-NEXT: %struct.struct_float64m1 = type { <16 x double> }

// CHECK-64:      %union.union_int8m1 = type { <8 x i8> }
// CHECK-64-NEXT: %union.union_int16m1 = type { <4 x i16> }
// CHECK-64-NEXT: %union.union_int32m1 = type { <2 x i32> }
// CHECK-64-NEXT: %union.union_int64m1 = type { <1 x i64> }
// CHECK-64-NEXT: %union.union_uint8m1 = type { <8 x i8> }
// CHECK-64-NEXT: %union.union_uint16m1 = type { <4 x i16> }
// CHECK-64-NEXT: %union.union_uint32m1 = type { <2 x i32> }
// CHECK-64-NEXT: %union.union_uint64m1 = type { <1 x i64> }
// CHECK-64-NEXT: %union.union_float32m1 = type { <2 x float> }
// CHECK-64-NEXT: %union.union_float64m1 = type { <1 x double> }

// CHECK-128:      %union.union_int8m1 = type { <16 x i8> }
// CHECK-128-NEXT: %union.union_int16m1 = type { <8 x i16> }
// CHECK-128-NEXT: %union.union_int32m1 = type { <4 x i32> }
// CHECK-128-NEXT: %union.union_int64m1 = type { <2 x i64> }
// CHECK-128-NEXT: %union.union_uint8m1 = type { <16 x i8> }
// CHECK-128-NEXT: %union.union_uint16m1 = type { <8 x i16> }
// CHECK-128-NEXT: %union.union_uint32m1 = type { <4 x i32> }
// CHECK-128-NEXT: %union.union_uint64m1 = type { <2 x i64> }
// CHECK-128-NEXT: %union.union_float32m1 = type { <4 x float> }
// CHECK-128-NEXT: %union.union_float64m1 = type { <2 x double> }

// CHECK-256:      %union.union_int8m1 = type { <32 x i8> }
// CHECK-256-NEXT: %union.union_int16m1 = type { <16 x i16> }
// CHECK-256-NEXT: %union.union_int32m1 = type { <8 x i32> }
// CHECK-256-NEXT: %union.union_int64m1 = type { <4 x i64> }
// CHECK-256-NEXT: %union.union_uint8m1 = type { <32 x i8> }
// CHECK-256-NEXT: %union.union_uint16m1 = type { <16 x i16> }
// CHECK-256-NEXT: %union.union_uint32m1 = type { <8 x i32> }
// CHECK-256-NEXT: %union.union_uint64m1 = type { <4 x i64> }
// CHECK-256-NEXT: %union.union_float32m1 = type { <8 x float> }
// CHECK-256-NEXT: %union.union_float64m1 = type { <4 x double> }

// CHECK-512:      %union.union_int8m1 = type { <64 x i8> }
// CHECK-512-NEXT: %union.union_int16m1 = type { <32 x i16> }
// CHECK-512-NEXT: %union.union_int32m1 = type { <16 x i32> }
// CHECK-512-NEXT: %union.union_int64m1 = type { <8 x i64> }
// CHECK-512-NEXT: %union.union_uint8m1 = type { <64 x i8> }
// CHECK-512-NEXT: %union.union_uint16m1 = type { <32 x i16> }
// CHECK-512-NEXT: %union.union_uint32m1 = type { <16 x i32> }
// CHECK-512-NEXT: %union.union_uint64m1 = type { <8 x i64> }
// CHECK-512-NEXT: %union.union_float32m1 = type { <16 x float> }
// CHECK-512-NEXT: %union.union_float64m1 = type { <8 x double> }

// CHECK-1024:      %union.union_int8m1 = type { <128 x i8> }
// CHECK-1024-NEXT: %union.union_int16m1 = type { <64 x i16> }
// CHECK-1024-NEXT: %union.union_int32m1 = type { <32 x i32> }
// CHECK-1024-NEXT: %union.union_int64m1 = type { <16 x i64> }
// CHECK-1024-NEXT: %union.union_uint8m1 = type { <128 x i8> }
// CHECK-1024-NEXT: %union.union_uint16m1 = type { <64 x i16> }
// CHECK-1024-NEXT: %union.union_uint32m1 = type { <32 x i32> }
// CHECK-1024-NEXT: %union.union_uint64m1 = type { <16 x i64> }
// CHECK-1024-NEXT: %union.union_float32m1 = type { <32 x float> }
// CHECK-1024-NEXT: %union.union_float64m1 = type { <16 x double> }

//===----------------------------------------------------------------------===//
// Global variables
//===----------------------------------------------------------------------===//
// CHECK-64:      @global_i8 ={{.*}} global <8 x i8> zeroinitializer, align 8
// CHECK-64-NEXT: @global_i16 ={{.*}} global <4 x i16> zeroinitializer, align 8
// CHECK-64-NEXT: @global_i32 ={{.*}} global <2 x i32> zeroinitializer, align 8
// CHECK-64-NEXT: @global_i64 ={{.*}} global <1 x i64> zeroinitializer, align 8
// CHECK-64-NEXT: @global_u8 ={{.*}} global <8 x i8> zeroinitializer, align 8
// CHECK-64-NEXT: @global_u16 ={{.*}} global <4 x i16> zeroinitializer, align 8
// CHECK-64-NEXT: @global_u32 ={{.*}} global <2 x i32> zeroinitializer, align 8
// CHECK-64-NEXT: @global_u64 ={{.*}} global <1 x i64> zeroinitializer, align 8
// CHECK-64-NEXT: @global_f32 ={{.*}} global <2 x float> zeroinitializer, align 8
// CHECK-64-NEXT: @global_f64 ={{.*}} global <1 x double> zeroinitializer, align 8

// CHECK-128:      @global_i8 ={{.*}} global <16 x i8> zeroinitializer, align 8
// CHECK-128-NEXT: @global_i16 ={{.*}} global <8 x i16> zeroinitializer, align 8
// CHECK-128-NEXT: @global_i32 ={{.*}} global <4 x i32> zeroinitializer, align 8
// CHECK-128-NEXT: @global_i64 ={{.*}} global <2 x i64> zeroinitializer, align 8
// CHECK-128-NEXT: @global_u8 ={{.*}} global <16 x i8> zeroinitializer, align 8
// CHECK-128-NEXT: @global_u16 ={{.*}} global <8 x i16> zeroinitializer, align 8
// CHECK-128-NEXT: @global_u32 ={{.*}} global <4 x i32> zeroinitializer, align 8
// CHECK-128-NEXT: @global_u64 ={{.*}} global <2 x i64> zeroinitializer, align 8
// CHECK-128-NEXT: @global_f32 ={{.*}} global <4 x float> zeroinitializer, align 8
// CHECK-128-NEXT: @global_f64 ={{.*}} global <2 x double> zeroinitializer, align 8

// CHECK-256:      @global_i8 ={{.*}} global <32 x i8> zeroinitializer, align 8
// CHECK-NEXT-256: @global_i16 ={{.*}} global <16 x i16> zeroinitializer, align 8
// CHECK-NEXT-256: @global_i32 ={{.*}} global <8 x i32> zeroinitializer, align 8
// CHECK-NEXT-256: @global_i64 ={{.*}} global <4 x i64> zeroinitializer, align 8
// CHECK-NEXT-256: @global_u8 ={{.*}} global <32 x i8> zeroinitializer, align 8
// CHECK-NEXT-256: @global_u16 ={{.*}} global <16 x i16> zeroinitializer, align 8
// CHECK-NEXT-256: @global_u32 ={{.*}} global <8 x i32> zeroinitializer, align 8
// CHECK-NEXT-256: @global_u64 ={{.*}} global <4 x i64> zeroinitializer, align 8
// CHECK-NEXT-256: @global_f32 ={{.*}} global <8 x float> zeroinitializer, align 8
// CHECK-NEXT-256: @global_f64 ={{.*}} global <4 x double> zeroinitializer, align 8

// CHECK-512:      @global_i8 ={{.*}} global <64 x i8> zeroinitializer, align 8
// CHECK-NEXT-512: @global_i16 ={{.*}} global <32 x i16> zeroinitializer, align 8
// CHECK-NEXT-512: @global_i32 ={{.*}} global <16 x i32> zeroinitializer, align 8
// CHECK-NEXT-512: @global_i64 ={{.*}} global <8 x i64> zeroinitializer, align 8
// CHECK-NEXT-512: @global_u8 ={{.*}} global <64 x i8> zeroinitializer, align 8
// CHECK-NEXT-512: @global_u16 ={{.*}} global <32 x i16> zeroinitializer, align 8
// CHECK-NEXT-512: @global_u32 ={{.*}} global <16 x i32> zeroinitializer, align 8
// CHECK-NEXT-512: @global_u64 ={{.*}} global <8 x i64> zeroinitializer, align 8
// CHECK-NEXT-512: @global_f32 ={{.*}} global <16 x float> zeroinitializer, align 8
// CHECK-NEXT-512: @global_f64 ={{.*}} global <8 x double> zeroinitializer, align 8

// CHECK-1024:      @global_i8 ={{.*}} global <128 x i8> zeroinitializer, align 8
// CHECK-NEXT-1024: @global_i16 ={{.*}} global <64 x i16> zeroinitializer, align 8
// CHECK-NEXT-1024: @global_i32 ={{.*}} global <32 x i32> zeroinitializer, align 8
// CHECK-NEXT-1024: @global_i64 ={{.*}} global <16 x i64> zeroinitializer, align 8
// CHECK-NEXT-1024: @global_u8 ={{.*}} global <128 x i8> zeroinitializer, align 8
// CHECK-NEXT-1024: @global_u16 ={{.*}} global <64 x i16> zeroinitializer, align 8
// CHECK-NEXT-1024: @global_u32 ={{.*}} global <32 x i32> zeroinitializer, align 8
// CHECK-NEXT-1024: @global_u64 ={{.*}} global <16 x i64> zeroinitializer, align 8
// CHECK-NEXT-1024: @global_f32 ={{.*}} global <32 x float> zeroinitializer, align 8
// CHECK-NEXT-1024: @global_f64 ={{.*}} global <16 x double> zeroinitializer, align 8

//===----------------------------------------------------------------------===//
// Global arrays
//===----------------------------------------------------------------------===//
// CHECK-64:      @global_arr_i8 ={{.*}} global [3 x <8 x i8>] zeroinitializer, align 8
// CHECK-64-NEXT: @global_arr_i16 ={{.*}} global [3 x <4 x i16>] zeroinitializer, align 8
// CHECK-64-NEXT: @global_arr_i32 ={{.*}} global [3 x <2 x i32>] zeroinitializer, align 8
// CHECK-64-NEXT: @global_arr_i64 ={{.*}} global [3 x <1 x i64>] zeroinitializer, align 8
// CHECK-64-NEXT: @global_arr_u8 ={{.*}} global [3 x <8 x i8>] zeroinitializer, align 8
// CHECK-64-NEXT: @global_arr_u16 ={{.*}} global [3 x <4 x i16>] zeroinitializer, align 8
// CHECK-64-NEXT: @global_arr_u32 ={{.*}} global [3 x <2 x i32>] zeroinitializer, align 8
// CHECK-64-NEXT: @global_arr_u64 ={{.*}} global [3 x <1 x i64>] zeroinitializer, align 8
// CHECK-64-NEXT: @global_arr_f32 ={{.*}} global [3 x <2 x float>] zeroinitializer, align 8
// CHECK-64-NEXT: @global_arr_f64 ={{.*}} global [3 x <1 x double>] zeroinitializer, align 8

// CHECK-128:      @global_arr_i8 ={{.*}} global [3 x <16 x i8>] zeroinitializer, align 8
// CHECK-128-NEXT: @global_arr_i16 ={{.*}} global [3 x <8 x i16>] zeroinitializer, align 8
// CHECK-128-NEXT: @global_arr_i32 ={{.*}} global [3 x <4 x i32>] zeroinitializer, align 8
// CHECK-128-NEXT: @global_arr_i64 ={{.*}} global [3 x <2 x i64>] zeroinitializer, align 8
// CHECK-128-NEXT: @global_arr_u8 ={{.*}} global [3 x <16 x i8>] zeroinitializer, align 8
// CHECK-128-NEXT: @global_arr_u16 ={{.*}} global [3 x <8 x i16>] zeroinitializer, align 8
// CHECK-128-NEXT: @global_arr_u32 ={{.*}} global [3 x <4 x i32>] zeroinitializer, align 8
// CHECK-128-NEXT: @global_arr_u64 ={{.*}} global [3 x <2 x i64>] zeroinitializer, align 8
// CHECK-128-NEXT: @global_arr_f32 ={{.*}} global [3 x <4 x float>] zeroinitializer, align 8
// CHECK-128-NEXT: @global_arr_f64 ={{.*}} global [3 x <2 x double>] zeroinitializer, align 8

// CHECK-256:      @global_arr_i8 ={{.*}} global [3 x <32 x i8>] zeroinitializer, align 8
// CHECK-NEXT-256: @global_arr_i16 ={{.*}} global [3 x <16 x i16>] zeroinitializer, align 8
// CHECK-NEXT-256: @global_arr_i32 ={{.*}} global [3 x <8 x i32>] zeroinitializer, align 8
// CHECK-NEXT-256: @global_arr_i64 ={{.*}} global [3 x <4 x i64>] zeroinitializer, align 8
// CHECK-NEXT-256: @global_arr_u8 ={{.*}} global [3 x <32 x i8>] zeroinitializer, align 8
// CHECK-NEXT-256: @global_arr_u16 ={{.*}} global [3 x <16 x i16>] zeroinitializer, align 8
// CHECK-NEXT-256: @global_arr_u32 ={{.*}} global [3 x <8 x i32>] zeroinitializer, align 8
// CHECK-NEXT-256: @global_arr_u64 ={{.*}} global [3 x <4 x i64>] zeroinitializer, align 8
// CHECK-NEXT-256: @global_arr_f32 ={{.*}} global [3 x <8 x float>] zeroinitializer, align 8
// CHECK-NEXT-256: @global_arr_f64 ={{.*}} global [3 x <4 x double>] zeroinitializer, align 8

// CHECK-512:      @global_arr_i8 ={{.*}} global [3 x <64 x i8>] zeroinitializer, align 8
// CHECK-NEXT-512: @global_arr_i16 ={{.*}} global [3 x <32 x i16>] zeroinitializer, align 8
// CHECK-NEXT-512: @global_arr_i32 ={{.*}} global [3 x <16 x i32>] zeroinitializer, align 8
// CHECK-NEXT-512: @global_arr_i64 ={{.*}} global [3 x <8 x i64>] zeroinitializer, align 8
// CHECK-NEXT-512: @global_arr_u8 ={{.*}} global [3 x <64 x i8>] zeroinitializer, align 8
// CHECK-NEXT-512: @global_arr_u16 ={{.*}} global [3 x <32 x i16>] zeroinitializer, align 8
// CHECK-NEXT-512: @global_arr_u32 ={{.*}} global [3 x <16 x i32>] zeroinitializer, align 8
// CHECK-NEXT-512: @global_arr_u64 ={{.*}} global [3 x <8 x i64>] zeroinitializer, align 8
// CHECK-NEXT-512: @global_arr_f32 ={{.*}} global [3 x <16 x float>] zeroinitializer, align 8
// CHECK-NEXT-512: @global_arr_f64 ={{.*}} global [3 x <8 x double>] zeroinitializer, align 8

// CHECK-1024:      @global_arr_i8 ={{.*}} global [3 x <128 x i8>] zeroinitializer, align 8
// CHECK-NEXT-1024: @global_arr_i16 ={{.*}} global [3 x <64 x i16>] zeroinitializer, align 8
// CHECK-NEXT-1024: @global_arr_i32 ={{.*}} global [3 x <32 x i32>] zeroinitializer, align 8
// CHECK-NEXT-1024: @global_arr_i64 ={{.*}} global [3 x <16 x i64>] zeroinitializer, align 8
// CHECK-NEXT-1024: @global_arr_u8 ={{.*}} global [3 x <128 x i8>] zeroinitializer, align 8
// CHECK-NEXT-1024: @global_arr_u16 ={{.*}} global [3 x <64 x i16>] zeroinitializer, align 8
// CHECK-NEXT-1024: @global_arr_u32 ={{.*}} global [3 x <32 x i32>] zeroinitializer, align 8
// CHECK-NEXT-1024: @global_arr_u64 ={{.*}} global [3 x <16 x i64>] zeroinitializer, align 8
// CHECK-NEXT-1024: @global_arr_f32 ={{.*}} global [3 x <32 x float>] zeroinitializer, align 8
// CHECK-NEXT-1024: @global_arr_f64 ={{.*}} global [3 x <16 x double>] zeroinitializer, align 8

//===----------------------------------------------------------------------===//
// Local variables
//===----------------------------------------------------------------------===//
// CHECK-64:      %local_i8 = alloca <8 x i8>, align 8
// CHECK-64-NEXT: %local_i16 = alloca <4 x i16>, align 8
// CHECK-64-NEXT: %local_i32 = alloca <2 x i32>, align 8
// CHECK-64-NEXT: %local_i64 = alloca <1 x i64>, align 8
// CHECK-64-NEXT: %local_u8 = alloca <8 x i8>, align 8
// CHECK-64-NEXT: %local_u16 = alloca <4 x i16>, align 8
// CHECK-64-NEXT: %local_u32 = alloca <2 x i32>, align 8
// CHECK-64-NEXT: %local_u64 = alloca <1 x i64>, align 8
// CHECK-64-NEXT: %local_f32 = alloca <2 x float>, align 8
// CHECK-64-NEXT: %local_f64 = alloca <1 x double>, align 8

// CHECK-128:      %local_i8 = alloca <16 x i8>, align 8
// CHECK-128-NEXT: %local_i16 = alloca <8 x i16>, align 8
// CHECK-128-NEXT: %local_i32 = alloca <4 x i32>, align 8
// CHECK-128-NEXT: %local_i64 = alloca <2 x i64>, align 8
// CHECK-128-NEXT: %local_u8 = alloca <16 x i8>, align 8
// CHECK-128-NEXT: %local_u16 = alloca <8 x i16>, align 8
// CHECK-128-NEXT: %local_u32 = alloca <4 x i32>, align 8
// CHECK-128-NEXT: %local_u64 = alloca <2 x i64>, align 8
// CHECK-128-NEXT: %local_f32 = alloca <4 x float>, align 8
// CHECK-128-NEXT: %local_f64 = alloca <2 x double>, align 8

// CHECK-256:      %local_i8 = alloca <32 x i8>, align 8
// CHECK-256-NEXT: %local_i16 = alloca <16 x i16>, align 8
// CHECK-256-NEXT: %local_i32 = alloca <8 x i32>, align 8
// CHECK-256-NEXT: %local_i64 = alloca <4 x i64>, align 8
// CHECK-256-NEXT: %local_u8 = alloca <32 x i8>, align 8
// CHECK-256-NEXT: %local_u16 = alloca <16 x i16>, align 8
// CHECK-256-NEXT: %local_u32 = alloca <8 x i32>, align 8
// CHECK-256-NEXT: %local_u64 = alloca <4 x i64>, align 8
// CHECK-256-NEXT: %local_f32 = alloca <8 x float>, align 8
// CHECK-256-NEXT: %local_f64 = alloca <4 x double>, align 8

// CHECK-512:      %local_i8 = alloca <64 x i8>, align 8
// CHECK-512-NEXT: %local_i16 = alloca <32 x i16>, align 8
// CHECK-512-NEXT: %local_i32 = alloca <16 x i32>, align 8
// CHECK-512-NEXT: %local_i64 = alloca <8 x i64>, align 8
// CHECK-512-NEXT: %local_u8 = alloca <64 x i8>, align 8
// CHECK-512-NEXT: %local_u16 = alloca <32 x i16>, align 8
// CHECK-512-NEXT: %local_u32 = alloca <16 x i32>, align 8
// CHECK-512-NEXT: %local_u64 = alloca <8 x i64>, align 8
// CHECK-512-NEXT: %local_f32 = alloca <16 x float>, align 8
// CHECK-512-NEXT: %local_f64 = alloca <8 x double>, align 8

// CHECK-1024:       %local_i8 = alloca <128 x i8>, align 8
// CHECK-1024-NEXT:  %local_i16 = alloca <64 x i16>, align 8
// CHECK-1024-NEXT:  %local_i32 = alloca <32 x i32>, align 8
// CHECK-1024-NEXT:  %local_i64 = alloca <16 x i64>, align 8
// CHECK-1024-NEXT:  %local_u8 = alloca <128 x i8>, align 8
// CHECK-1024-NEXT:  %local_u16 = alloca <64 x i16>, align 8
// CHECK-1024-NEXT:  %local_u32 = alloca <32 x i32>, align 8
// CHECK-1024-NEXT:  %local_u64 = alloca <16 x i64>, align 8
// CHECK-1024-NEXT:  %local_f32 = alloca <32 x float>, align 8
// CHECK-1024-NEXT:  %local_f64 = alloca <16 x double>, align 8

//===----------------------------------------------------------------------===//
// Local arrays
//===----------------------------------------------------------------------===//
// CHECK-64:      %local_arr_i8 = alloca [3 x <8 x i8>], align 8
// CHECK-64-NEXT: %local_arr_i16 = alloca [3 x <4 x i16>], align 8
// CHECK-64-NEXT: %local_arr_i32 = alloca [3 x <2 x i32>], align 8
// CHECK-64-NEXT: %local_arr_i64 = alloca [3 x <1 x i64>], align 8
// CHECK-64-NEXT: %local_arr_u8 = alloca [3 x <8 x i8>], align 8
// CHECK-64-NEXT: %local_arr_u16 = alloca [3 x <4 x i16>], align 8
// CHECK-64-NEXT: %local_arr_u32 = alloca [3 x <2 x i32>], align 8
// CHECK-64-NEXT: %local_arr_u64 = alloca [3 x <1 x i64>], align 8
// CHECK-64-NEXT: %local_arr_f32 = alloca [3 x <2 x float>], align 8
// CHECK-64-NEXT: %local_arr_f64 = alloca [3 x <1 x double>], align 8

// CHECK-128:      %local_arr_i8 = alloca [3 x <16 x i8>], align 8
// CHECK-128-NEXT: %local_arr_i16 = alloca [3 x <8 x i16>], align 8
// CHECK-128-NEXT: %local_arr_i32 = alloca [3 x <4 x i32>], align 8
// CHECK-128-NEXT: %local_arr_i64 = alloca [3 x <2 x i64>], align 8
// CHECK-128-NEXT: %local_arr_u8 = alloca [3 x <16 x i8>], align 8
// CHECK-128-NEXT: %local_arr_u16 = alloca [3 x <8 x i16>], align 8
// CHECK-128-NEXT: %local_arr_u32 = alloca [3 x <4 x i32>], align 8
// CHECK-128-NEXT: %local_arr_u64 = alloca [3 x <2 x i64>], align 8
// CHECK-128-NEXT: %local_arr_f32 = alloca [3 x <4 x float>], align 8
// CHECK-128-NEXT: %local_arr_f64 = alloca [3 x <2 x double>], align 8

// CHECK-256:      %local_arr_i8 = alloca [3 x <32 x i8>], align 8
// CHECK-256-NEXT: %local_arr_i16 = alloca [3 x <16 x i16>], align 8
// CHECK-256-NEXT: %local_arr_i32 = alloca [3 x <8 x i32>], align 8
// CHECK-256-NEXT: %local_arr_i64 = alloca [3 x <4 x i64>], align 8
// CHECK-256-NEXT: %local_arr_u8 = alloca [3 x <32 x i8>], align 8
// CHECK-256-NEXT: %local_arr_u16 = alloca [3 x <16 x i16>], align 8
// CHECK-256-NEXT: %local_arr_u32 = alloca [3 x <8 x i32>], align 8
// CHECK-256-NEXT: %local_arr_u64 = alloca [3 x <4 x i64>], align 8
// CHECK-256-NEXT: %local_arr_f32 = alloca [3 x <8 x float>], align 8
// CHECK-256-NEXT: %local_arr_f64 = alloca [3 x <4 x double>], align 8

// CHECK-512:      %local_arr_i8 = alloca [3 x <64 x i8>], align 8
// CHECK-512-NEXT: %local_arr_i16 = alloca [3 x <32 x i16>], align 8
// CHECK-512-NEXT: %local_arr_i32 = alloca [3 x <16 x i32>], align 8
// CHECK-512-NEXT: %local_arr_i64 = alloca [3 x <8 x i64>], align 8
// CHECK-512-NEXT: %local_arr_u8 = alloca [3 x <64 x i8>], align 8
// CHECK-512-NEXT: %local_arr_u16 = alloca [3 x <32 x i16>], align 8
// CHECK-512-NEXT: %local_arr_u32 = alloca [3 x <16 x i32>], align 8
// CHECK-512-NEXT: %local_arr_u64 = alloca [3 x <8 x i64>], align 8
// CHECK-512-NEXT: %local_arr_f32 = alloca [3 x <16 x float>], align 8
// CHECK-512-NEXT: %local_arr_f64 = alloca [3 x <8 x double>], align 8

// CHECK-1024:       %local_arr_i8 = alloca [3 x <128 x i8>], align 8
// CHECK-1024-NEXT:  %local_arr_i16 = alloca [3 x <64 x i16>], align 8
// CHECK-1024-NEXT:  %local_arr_i32 = alloca [3 x <32 x i32>], align 8
// CHECK-1024-NEXT:  %local_arr_i64 = alloca [3 x <16 x i64>], align 8
// CHECK-1024-NEXT:  %local_arr_u8 = alloca [3 x <128 x i8>], align 8
// CHECK-1024-NEXT:  %local_arr_u16 = alloca [3 x <64 x i16>], align 8
// CHECK-1024-NEXT:  %local_arr_u32 = alloca [3 x <32 x i32>], align 8
// CHECK-1024-NEXT:  %local_arr_u64 = alloca [3 x <16 x i64>], align 8
// CHECK-1024-NEXT:  %local_arr_f32 = alloca [3 x <32 x float>], align 8
// CHECK-1024-NEXT:  %local_arr_f64 = alloca [3 x <16 x double>], align 8

//===----------------------------------------------------------------------===//
// ILP32 ABI
//===----------------------------------------------------------------------===//
// CHECK-ILP32: @global_i32 ={{.*}} global <16 x i32> zeroinitializer, align 8
// CHECK-ILP32: @global_i64 ={{.*}} global <8 x i64> zeroinitializer, align 8
// CHECK-ILP32: @global_u32 ={{.*}} global <16 x i32> zeroinitializer, align 8
// CHECK-ILP32: @global_u64 ={{.*}} global <8 x i64> zeroinitializer, align 8

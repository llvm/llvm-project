// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +f -target-feature +d -target-feature +zve64d -mvscale-min=1 -mvscale-max=1 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-64
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +f -target-feature +d -target-feature +zve64d -mvscale-min=2 -mvscale-max=2 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-128
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +f -target-feature +d -target-feature +zve64d -mvscale-min=4 -mvscale-max=4 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-256
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +f -target-feature +d -target-feature +zve64d -mvscale-min=8 -mvscale-max=8 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-512
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +f -target-feature +d -target-feature +zve64d -mvscale-min=16 -mvscale-max=16 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-1024

// REQUIRES: riscv-registered-target

#include <stdint.h>

typedef __rvv_bool64_t vbool64_t;
typedef __rvv_bool32_t vbool32_t;
typedef __rvv_bool16_t vbool16_t;
typedef __rvv_bool8_t vbool8_t;
typedef __rvv_bool4_t vbool4_t;
typedef __rvv_bool2_t vbool2_t;
typedef __rvv_bool1_t vbool1_t;

typedef __rvv_int8mf8_t vint8mf8_t;
typedef __rvv_uint8mf8_t vuint8mf8_t;

typedef __rvv_int8mf4_t vint8mf4_t;
typedef __rvv_uint8mf4_t vuint8mf4_t;
typedef __rvv_int16mf4_t vint16mf4_t;
typedef __rvv_uint16mf4_t vuint16mf4_t;

typedef __rvv_int8mf2_t vint8mf2_t;
typedef __rvv_uint8mf2_t vuint8mf2_t;
typedef __rvv_int16mf2_t vint16mf2_t;
typedef __rvv_uint16mf2_t vuint16mf2_t;
typedef __rvv_int32mf2_t vint32mf2_t;
typedef __rvv_uint32mf2_t vuint32mf2_t;
typedef __rvv_float32mf2_t vfloat32mf2_t;

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

typedef __rvv_int8m2_t vint8m2_t;
typedef __rvv_uint8m2_t vuint8m2_t;
typedef __rvv_int16m2_t vint16m2_t;
typedef __rvv_uint16m2_t vuint16m2_t;
typedef __rvv_int32m2_t vint32m2_t;
typedef __rvv_uint32m2_t vuint32m2_t;
typedef __rvv_int64m2_t vint64m2_t;
typedef __rvv_uint64m2_t vuint64m2_t;
typedef __rvv_float32m2_t vfloat32m2_t;
typedef __rvv_float64m2_t vfloat64m2_t;

typedef __rvv_int8m4_t vint8m4_t;
typedef __rvv_uint8m4_t vuint8m4_t;
typedef __rvv_int16m4_t vint16m4_t;
typedef __rvv_uint16m4_t vuint16m4_t;
typedef __rvv_int32m4_t vint32m4_t;
typedef __rvv_uint32m4_t vuint32m4_t;
typedef __rvv_int64m4_t vint64m4_t;
typedef __rvv_uint64m4_t vuint64m4_t;
typedef __rvv_float32m4_t vfloat32m4_t;
typedef __rvv_float64m4_t vfloat64m4_t;

typedef __rvv_int8m8_t vint8m8_t;
typedef __rvv_uint8m8_t vuint8m8_t;
typedef __rvv_int16m8_t vint16m8_t;
typedef __rvv_uint16m8_t vuint16m8_t;
typedef __rvv_int32m8_t vint32m8_t;
typedef __rvv_uint32m8_t vuint32m8_t;
typedef __rvv_int64m8_t vint64m8_t;
typedef __rvv_uint64m8_t vuint64m8_t;
typedef __rvv_float32m8_t vfloat32m8_t;
typedef __rvv_float64m8_t vfloat64m8_t;

// Define valid fixed-width RVV types
typedef vint8mf8_t fixed_int8mf8_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen / 8)));

typedef vuint8mf8_t fixed_uint8mf8_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen / 8)));

typedef vint8mf4_t fixed_int8mf4_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen / 4)));
typedef vint16mf4_t fixed_int16mf4_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen / 4)));

typedef vuint8mf4_t fixed_uint8mf4_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen / 4)));
typedef vuint16mf4_t fixed_uint16mf4_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen / 4)));

typedef vint8mf2_t fixed_int8mf2_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen / 2)));
typedef vint16mf2_t fixed_int16mf2_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen / 2)));
typedef vint32mf2_t fixed_int32mf2_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen / 2)));

typedef vuint8mf2_t fixed_uint8mf2_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen / 2)));
typedef vuint16mf2_t fixed_uint16mf2_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen / 2)));
typedef vuint32mf2_t fixed_uint32mf2_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen / 2)));

typedef vfloat32mf2_t fixed_float32mf2_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen / 2)));

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

typedef vint8m2_t fixed_int8m2_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen * 2)));
typedef vint16m2_t fixed_int16m2_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen * 2)));
typedef vint32m2_t fixed_int32m2_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen * 2)));
typedef vint64m2_t fixed_int64m2_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen * 2)));

typedef vuint8m2_t fixed_uint8m2_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen * 2)));
typedef vuint16m2_t fixed_uint16m2_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen * 2)));
typedef vuint32m2_t fixed_uint32m2_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen * 2)));
typedef vuint64m2_t fixed_uint64m2_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen * 2)));

typedef vfloat32m2_t fixed_float32m2_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen * 2)));
typedef vfloat64m2_t fixed_float64m2_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen * 2)));

typedef vint8m4_t fixed_int8m4_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen * 4)));
typedef vint16m4_t fixed_int16m4_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen * 4)));
typedef vint32m4_t fixed_int32m4_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen * 4)));
typedef vint64m4_t fixed_int64m4_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen * 4)));

typedef vuint8m4_t fixed_uint8m4_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen * 4)));
typedef vuint16m4_t fixed_uint16m4_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen * 4)));
typedef vuint32m4_t fixed_uint32m4_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen * 4)));
typedef vuint64m4_t fixed_uint64m4_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen * 4)));

typedef vfloat32m4_t fixed_float32m4_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen * 4)));
typedef vfloat64m4_t fixed_float64m4_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen * 4)));

typedef vint8m8_t fixed_int8m8_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen * 8)));
typedef vint16m8_t fixed_int16m8_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen * 8)));
typedef vint32m8_t fixed_int32m8_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen * 8)));
typedef vint64m8_t fixed_int64m8_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen * 8)));

typedef vuint8m8_t fixed_uint8m8_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen * 8)));
typedef vuint16m8_t fixed_uint16m8_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen * 8)));
typedef vuint32m8_t fixed_uint32m8_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen * 8)));
typedef vuint64m8_t fixed_uint64m8_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen * 8)));

typedef vfloat32m8_t fixed_float32m8_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen * 8)));
typedef vfloat64m8_t fixed_float64m8_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen * 8)));

#if __riscv_v_fixed_vlen / 64 >= 8
typedef vbool64_t fixed_bool64_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen / 64)));
#endif
#if __riscv_v_fixed_vlen / 32 >= 8
typedef vbool32_t fixed_bool32_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen / 32)));
#endif
#if __riscv_v_fixed_vlen / 16 >= 8
typedef vbool16_t fixed_bool16_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen / 16)));
#endif
typedef vbool8_t fixed_bool8_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen / 8)));
typedef vbool4_t fixed_bool4_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen / 4)));
typedef vbool2_t fixed_bool2_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen / 2)));
typedef vbool1_t fixed_bool1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));

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

DEFINE_STRUCT(int8m2)
DEFINE_STRUCT(int16m2)
DEFINE_STRUCT(int32m2)
DEFINE_STRUCT(int64m2)
DEFINE_STRUCT(uint8m2)
DEFINE_STRUCT(uint16m2)
DEFINE_STRUCT(uint32m2)
DEFINE_STRUCT(uint64m2)
DEFINE_STRUCT(float32m2)
DEFINE_STRUCT(float64m2)

DEFINE_STRUCT(int8m4)
DEFINE_STRUCT(int16m4)
DEFINE_STRUCT(int32m4)
DEFINE_STRUCT(int64m4)
DEFINE_STRUCT(uint8m4)
DEFINE_STRUCT(uint16m4)
DEFINE_STRUCT(uint32m4)
DEFINE_STRUCT(uint64m4)
DEFINE_STRUCT(float32m4)
DEFINE_STRUCT(float64m4)

DEFINE_STRUCT(int8m8)
DEFINE_STRUCT(int16m8)
DEFINE_STRUCT(int32m8)
DEFINE_STRUCT(int64m8)
DEFINE_STRUCT(uint8m8)
DEFINE_STRUCT(uint16m8)
DEFINE_STRUCT(uint32m8)
DEFINE_STRUCT(uint64m8)
DEFINE_STRUCT(float32m8)
DEFINE_STRUCT(float64m8)

DEFINE_STRUCT(bool1)
DEFINE_STRUCT(bool2)
DEFINE_STRUCT(bool4)
DEFINE_STRUCT(bool8)
#if __riscv_v_fixed_vlen / 16 >= 8
DEFINE_STRUCT(bool16)
#endif
#if __riscv_v_fixed_vlen / 32 >= 8
DEFINE_STRUCT(bool32)
#endif
#if __riscv_v_fixed_vlen / 64 >= 8
DEFINE_STRUCT(bool64)
#endif

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

DEFINE_UNION(int8m2)
DEFINE_UNION(int16m2)
DEFINE_UNION(int32m2)
DEFINE_UNION(int64m2)
DEFINE_UNION(uint8m2)
DEFINE_UNION(uint16m2)
DEFINE_UNION(uint32m2)
DEFINE_UNION(uint64m2)
DEFINE_UNION(float32m2)
DEFINE_UNION(float64m2)

DEFINE_UNION(int8m4)
DEFINE_UNION(int16m4)
DEFINE_UNION(int32m4)
DEFINE_UNION(int64m4)
DEFINE_UNION(uint8m4)
DEFINE_UNION(uint16m4)
DEFINE_UNION(uint32m4)
DEFINE_UNION(uint64m4)
DEFINE_UNION(float32m4)
DEFINE_UNION(float64m4)

DEFINE_UNION(int8m8)
DEFINE_UNION(int16m8)
DEFINE_UNION(int32m8)
DEFINE_UNION(int64m8)
DEFINE_UNION(uint8m8)
DEFINE_UNION(uint16m8)
DEFINE_UNION(uint32m8)
DEFINE_UNION(uint64m8)
DEFINE_UNION(float32m8)
DEFINE_UNION(float64m8)

DEFINE_UNION(bool1)
DEFINE_UNION(bool2)
DEFINE_UNION(bool4)
DEFINE_UNION(bool8)
#if __riscv_v_fixed_vlen / 16 >= 8
DEFINE_UNION(bool16)
#endif
#if __riscv_v_fixed_vlen / 32 >= 8
DEFINE_UNION(bool32)
#endif
#if __riscv_v_fixed_vlen / 64 >= 8
DEFINE_UNION(bool64)
#endif

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

fixed_int8m2_t global_i8m2;
fixed_int16m2_t global_i16m2;
fixed_int32m2_t global_i32m2;
fixed_int64m2_t global_i64m2;

fixed_uint8m2_t global_u8m2;
fixed_uint16m2_t global_u16m2;
fixed_uint32m2_t global_u32m2;
fixed_uint64m2_t global_u64m2;

fixed_float32m2_t global_f32m2;
fixed_float64m2_t global_f64m2;

fixed_int8m4_t global_i8m4;
fixed_int16m4_t global_i16m4;
fixed_int32m4_t global_i32m4;
fixed_int64m4_t global_i64m4;

fixed_uint8m4_t global_u8m4;
fixed_uint16m4_t global_u16m4;
fixed_uint32m4_t global_u32m4;
fixed_uint64m4_t global_u64m4;

fixed_float32m4_t global_f32m4;
fixed_float64m4_t global_f64m4;

fixed_int8m8_t global_i8m8;
fixed_int16m8_t global_i16m8;
fixed_int32m8_t global_i32m8;
fixed_int64m8_t global_i64m8;

fixed_uint8m8_t global_u8m8;
fixed_uint16m8_t global_u16m8;
fixed_uint32m8_t global_u32m8;
fixed_uint64m8_t global_u64m8;

fixed_float32m8_t global_f32m8;
fixed_float64m8_t global_f64m8;

fixed_bool1_t global_bool1;
fixed_bool2_t global_bool2;
fixed_bool4_t global_bool4;
fixed_bool8_t global_bool8;
#if __riscv_v_fixed_vlen / 16 >= 8
fixed_bool16_t global_bool16;
#endif
#if __riscv_v_fixed_vlen / 32 >= 8
fixed_bool32_t global_bool32;
#endif
#if __riscv_v_fixed_vlen / 64 >= 8
fixed_bool64_t global_bool64;
#endif

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

fixed_int8m2_t global_arr_i8m2[3];
fixed_int16m2_t global_arr_i16m2[3];
fixed_int32m2_t global_arr_i32m2[3];
fixed_int64m2_t global_arr_i64m2[3];

fixed_uint8m2_t global_arr_u8m2[3];
fixed_uint16m2_t global_arr_u16m2[3];
fixed_uint32m2_t global_arr_u32m2[3];
fixed_uint64m2_t global_arr_u64m2[3];

fixed_float32m2_t global_arr_f32m2[3];
fixed_float64m2_t global_arr_f64m2[3];

fixed_int8m4_t global_arr_i8m4[3];
fixed_int16m4_t global_arr_i16m4[3];
fixed_int32m4_t global_arr_i32m4[3];
fixed_int64m4_t global_arr_i64m4[3];

fixed_uint8m4_t global_arr_u8m4[3];
fixed_uint16m4_t global_arr_u16m4[3];
fixed_uint32m4_t global_arr_u32m4[3];
fixed_uint64m4_t global_arr_u64m4[3];

fixed_float32m4_t global_arr_f32m4[3];
fixed_float64m4_t global_arr_f64m4[3];

fixed_int8m8_t global_arr_i8m8[3];
fixed_int16m8_t global_arr_i16m8[3];
fixed_int32m8_t global_arr_i32m8[3];
fixed_int64m8_t global_arr_i64m8[3];

fixed_uint8m8_t global_arr_u8m8[3];
fixed_uint16m8_t global_arr_u16m8[3];
fixed_uint32m8_t global_arr_u32m8[3];
fixed_uint64m8_t global_arr_u64m8[3];

fixed_float32m8_t global_arr_f32m8[3];
fixed_float64m8_t global_arr_f64m8[3];

fixed_bool1_t global_arr_bool1[3];
fixed_bool2_t global_arr_bool2[3];
fixed_bool4_t global_arr_bool4[3];
fixed_bool8_t global_arr_bool8[3];
#if __riscv_v_fixed_vlen / 16 >= 8
fixed_bool16_t global_arr_bool16[3];
#endif
#if __riscv_v_fixed_vlen / 32 >= 8
fixed_bool32_t global_arr_bool32[3];
#endif
#if __riscv_v_fixed_vlen / 64 >= 8
fixed_bool64_t global_arr_bool64[3];
#endif

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

  fixed_int8m2_t local_i8m2;
  fixed_int16m2_t local_i16m2;
  fixed_int32m2_t local_i32m2;
  fixed_int64m2_t local_i64m2;
  fixed_uint8m2_t local_u8m2;
  fixed_uint16m2_t local_u16m2;
  fixed_uint32m2_t local_u32m2;
  fixed_uint64m2_t local_u64m2;
  fixed_float32m2_t local_f32m2;
  fixed_float64m2_t local_f64m2;

  fixed_int8m4_t local_i8m4;
  fixed_int16m4_t local_i16m4;
  fixed_int32m4_t local_i32m4;
  fixed_int64m4_t local_i64m4;
  fixed_uint8m4_t local_u8m4;
  fixed_uint16m4_t local_u16m4;
  fixed_uint32m4_t local_u32m4;
  fixed_uint64m4_t local_u64m4;
  fixed_float32m4_t local_f32m4;
  fixed_float64m4_t local_f64m4;

  fixed_int8m8_t local_i8m8;
  fixed_int16m8_t local_i16m8;
  fixed_int32m8_t local_i32m8;
  fixed_int64m8_t local_i64m8;
  fixed_uint8m8_t local_u8m8;
  fixed_uint16m8_t local_u16m8;
  fixed_uint32m8_t local_u32m8;
  fixed_uint64m8_t local_u64m8;
  fixed_float32m8_t local_f32m8;
  fixed_float64m8_t local_f64m8;

  fixed_bool1_t local_bool1;
  fixed_bool2_t local_bool2;
  fixed_bool4_t local_bool4;
  fixed_bool8_t local_bool8;
#if __riscv_v_fixed_vlen / 16 >= 8
  fixed_bool16_t local_bool16;
#endif
#if __riscv_v_fixed_vlen / 32 >= 8
  fixed_bool32_t local_bool32;
#endif
#if __riscv_v_fixed_vlen / 64 >= 8
  fixed_bool64_t local_bool64;
#endif

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

  fixed_int8m2_t local_arr_i8m2[3];
  fixed_int16m2_t local_arr_i16m2[3];
  fixed_int32m2_t local_arr_i32m2[3];
  fixed_int64m2_t local_arr_i64m2[3];
  fixed_uint8m2_t local_arr_u8m2[3];
  fixed_uint16m2_t local_arr_u16m2[3];
  fixed_uint32m2_t local_arr_u32m2[3];
  fixed_uint64m2_t local_arr_u64m2[3];
  fixed_float32m2_t local_arr_f32m2[3];
  fixed_float64m2_t local_arr_f64m2[3];

  fixed_int8m4_t local_arr_i8m4[3];
  fixed_int16m4_t local_arr_i16m4[3];
  fixed_int32m4_t local_arr_i32m4[3];
  fixed_int64m4_t local_arr_i64m4[3];
  fixed_uint8m4_t local_arr_u8m4[3];
  fixed_uint16m4_t local_arr_u16m4[3];
  fixed_uint32m4_t local_arr_u32m4[3];
  fixed_uint64m4_t local_arr_u64m4[3];
  fixed_float32m4_t local_arr_f32m4[3];
  fixed_float64m4_t local_arr_f64m4[3];

  fixed_int8m8_t local_arr_i8m8[3];
  fixed_int16m8_t local_arr_i16m8[3];
  fixed_int32m8_t local_arr_i32m8[3];
  fixed_int64m8_t local_arr_i64m8[3];
  fixed_uint8m8_t local_arr_u8m8[3];
  fixed_uint16m8_t local_arr_u16m8[3];
  fixed_uint32m8_t local_arr_u32m8[3];
  fixed_uint64m8_t local_arr_u64m8[3];
  fixed_float32m8_t local_arr_f32m8[3];
  fixed_float64m8_t local_arr_f64m8[3];

  fixed_int8mf2_t local_arr_i8mf2[3];
  fixed_int16mf2_t local_arr_i16mf2[3];
  fixed_int32mf2_t local_arr_i32mf2[3];
  fixed_uint8mf2_t local_arr_u8mf2[3];
  fixed_uint16mf2_t local_arr_u16mf2[3];
  fixed_uint32mf2_t local_arr_u32mf2[3];
  fixed_float32mf2_t local_arr_f32mf2[3];

  fixed_int8mf4_t local_arr_i8mf4[3];
  fixed_int16mf4_t local_arr_i16mf4[3];
  fixed_uint8mf4_t local_arr_u8mf4[3];
  fixed_uint16mf4_t local_arr_u16mf4[3];

  fixed_int8mf8_t local_arr_i8mf8[3];
  fixed_uint8mf8_t local_arr_u8mf8[3];

  fixed_bool1_t local_arr_bool1[3];
  fixed_bool2_t local_arr_bool2[3];
  fixed_bool4_t local_arr_bool4[3];
  fixed_bool8_t local_arr_bool8[3];
#if __riscv_v_fixed_vlen / 16 >= 8
  fixed_bool16_t local_arr_bool16[3];
#endif
#if __riscv_v_fixed_vlen / 32 >= 8
  fixed_bool32_t local_arr_bool32[3];
#endif
#if __riscv_v_fixed_vlen / 64 >= 8
  fixed_bool64_t local_arr_bool64[3];
#endif
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
// CHECK-64-NEXT: %struct.struct_int8m2 = type { <16 x i8> }
// CHECK-64-NEXT: %struct.struct_int16m2 = type { <8 x i16> }
// CHECK-64-NEXT: %struct.struct_int32m2 = type { <4 x i32> }
// CHECK-64-NEXT: %struct.struct_int64m2 = type { <2 x i64> }
// CHECK-64-NEXT: %struct.struct_uint8m2 = type { <16 x i8> }
// CHECK-64-NEXT: %struct.struct_uint16m2 = type { <8 x i16> }
// CHECK-64-NEXT: %struct.struct_uint32m2 = type { <4 x i32> }
// CHECK-64-NEXT: %struct.struct_uint64m2 = type { <2 x i64> }
// CHECK-64-NEXT: %struct.struct_float32m2 = type { <4 x float> }
// CHECK-64-NEXT: %struct.struct_float64m2 = type { <2 x double> }
// CHECK-64-NEXT: %struct.struct_int8m4 = type { <32 x i8> }
// CHECK-64-NEXT: %struct.struct_int16m4 = type { <16 x i16> }
// CHECK-64-NEXT: %struct.struct_int32m4 = type { <8 x i32> }
// CHECK-64-NEXT: %struct.struct_int64m4 = type { <4 x i64> }
// CHECK-64-NEXT: %struct.struct_uint8m4 = type { <32 x i8> }
// CHECK-64-NEXT: %struct.struct_uint16m4 = type { <16 x i16> }
// CHECK-64-NEXT: %struct.struct_uint32m4 = type { <8 x i32> }
// CHECK-64-NEXT: %struct.struct_uint64m4 = type { <4 x i64> }
// CHECK-64-NEXT: %struct.struct_float32m4 = type { <8 x float> }
// CHECK-64-NEXT: %struct.struct_float64m4 = type { <4 x double> }
// CHECK-64-NEXT: %struct.struct_int8m8 = type { <64 x i8> }
// CHECK-64-NEXT: %struct.struct_int16m8 = type { <32 x i16> }
// CHECK-64-NEXT: %struct.struct_int32m8 = type { <16 x i32> }
// CHECK-64-NEXT: %struct.struct_int64m8 = type { <8 x i64> }
// CHECK-64-NEXT: %struct.struct_uint8m8 = type { <64 x i8> }
// CHECK-64-NEXT: %struct.struct_uint16m8 = type { <32 x i16> }
// CHECK-64-NEXT: %struct.struct_uint32m8 = type { <16 x i32> }
// CHECK-64-NEXT: %struct.struct_uint64m8 = type { <8 x i64> }
// CHECK-64-NEXT: %struct.struct_float32m8 = type { <16 x float> }
// CHECK-64-NEXT: %struct.struct_float64m8 = type { <8 x double> }
// CHECK-64-NEXT: %struct.struct_bool1 = type { <8 x i8> }
// CHECK-64-NEXT: %struct.struct_bool2 = type { <4 x i8> }
// CHECK-64-NEXT: %struct.struct_bool4 = type { <2 x i8> }
// CHECK-64-NEXT: %struct.struct_bool8 = type { <1 x i8> }

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
// CHECK-128-NEXT: %struct.struct_int8m2 = type { <32 x i8> }
// CHECK-128-NEXT: %struct.struct_int16m2 = type { <16 x i16> }
// CHECK-128-NEXT: %struct.struct_int32m2 = type { <8 x i32> }
// CHECK-128-NEXT: %struct.struct_int64m2 = type { <4 x i64> }
// CHECK-128-NEXT: %struct.struct_uint8m2 = type { <32 x i8> }
// CHECK-128-NEXT: %struct.struct_uint16m2 = type { <16 x i16> }
// CHECK-128-NEXT: %struct.struct_uint32m2 = type { <8 x i32> }
// CHECK-128-NEXT: %struct.struct_uint64m2 = type { <4 x i64> }
// CHECK-128-NEXT: %struct.struct_float32m2 = type { <8 x float> }
// CHECK-128-NEXT: %struct.struct_float64m2 = type { <4 x double> }
// CHECK-128-NEXT: %struct.struct_int8m4 = type { <64 x i8> }
// CHECK-128-NEXT: %struct.struct_int16m4 = type { <32 x i16> }
// CHECK-128-NEXT: %struct.struct_int32m4 = type { <16 x i32> }
// CHECK-128-NEXT: %struct.struct_int64m4 = type { <8 x i64> }
// CHECK-128-NEXT: %struct.struct_uint8m4 = type { <64 x i8> }
// CHECK-128-NEXT: %struct.struct_uint16m4 = type { <32 x i16> }
// CHECK-128-NEXT: %struct.struct_uint32m4 = type { <16 x i32> }
// CHECK-128-NEXT: %struct.struct_uint64m4 = type { <8 x i64> }
// CHECK-128-NEXT: %struct.struct_float32m4 = type { <16 x float> }
// CHECK-128-NEXT: %struct.struct_float64m4 = type { <8 x double> }
// CHECK-128-NEXT: %struct.struct_int8m8 = type { <128 x i8> }
// CHECK-128-NEXT: %struct.struct_int16m8 = type { <64 x i16> }
// CHECK-128-NEXT: %struct.struct_int32m8 = type { <32 x i32> }
// CHECK-128-NEXT: %struct.struct_int64m8 = type { <16 x i64> }
// CHECK-128-NEXT: %struct.struct_uint8m8 = type { <128 x i8> }
// CHECK-128-NEXT: %struct.struct_uint16m8 = type { <64 x i16> }
// CHECK-128-NEXT: %struct.struct_uint32m8 = type { <32 x i32> }
// CHECK-128-NEXT: %struct.struct_uint64m8 = type { <16 x i64> }
// CHECK-128-NEXT: %struct.struct_float32m8 = type { <32 x float> }
// CHECK-128-NEXT: %struct.struct_float64m8 = type { <16 x double> }
// CHECK-128-NEXT: %struct.struct_bool1 = type { <16 x i8> }
// CHECK-128-NEXT: %struct.struct_bool2 = type { <8 x i8> }
// CHECK-128-NEXT: %struct.struct_bool4 = type { <4 x i8> }
// CHECK-128-NEXT: %struct.struct_bool8 = type { <2 x i8> }
// CHECK-128-NEXT: %struct.struct_bool16 = type { <1 x i8> }

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
// CHECK-256-NEXT: %struct.struct_int8m2 = type { <64 x i8> }
// CHECK-256-NEXT: %struct.struct_int16m2 = type { <32 x i16> }
// CHECK-256-NEXT: %struct.struct_int32m2 = type { <16 x i32> }
// CHECK-256-NEXT: %struct.struct_int64m2 = type { <8 x i64> }
// CHECK-256-NEXT: %struct.struct_uint8m2 = type { <64 x i8> }
// CHECK-256-NEXT: %struct.struct_uint16m2 = type { <32 x i16> }
// CHECK-256-NEXT: %struct.struct_uint32m2 = type { <16 x i32> }
// CHECK-256-NEXT: %struct.struct_uint64m2 = type { <8 x i64> }
// CHECK-256-NEXT: %struct.struct_float32m2 = type { <16 x float> }
// CHECK-256-NEXT: %struct.struct_float64m2 = type { <8 x double> }
// CHECK-256-NEXT: %struct.struct_int8m4 = type { <128 x i8> }
// CHECK-256-NEXT: %struct.struct_int16m4 = type { <64 x i16> }
// CHECK-256-NEXT: %struct.struct_int32m4 = type { <32 x i32> }
// CHECK-256-NEXT: %struct.struct_int64m4 = type { <16 x i64> }
// CHECK-256-NEXT: %struct.struct_uint8m4 = type { <128 x i8> }
// CHECK-256-NEXT: %struct.struct_uint16m4 = type { <64 x i16> }
// CHECK-256-NEXT: %struct.struct_uint32m4 = type { <32 x i32> }
// CHECK-256-NEXT: %struct.struct_uint64m4 = type { <16 x i64> }
// CHECK-256-NEXT: %struct.struct_float32m4 = type { <32 x float> }
// CHECK-256-NEXT: %struct.struct_float64m4 = type { <16 x double> }
// CHECK-256-NEXT: %struct.struct_int8m8 = type { <256 x i8> }
// CHECK-256-NEXT: %struct.struct_int16m8 = type { <128 x i16> }
// CHECK-256-NEXT: %struct.struct_int32m8 = type { <64 x i32> }
// CHECK-256-NEXT: %struct.struct_int64m8 = type { <32 x i64> }
// CHECK-256-NEXT: %struct.struct_uint8m8 = type { <256 x i8> }
// CHECK-256-NEXT: %struct.struct_uint16m8 = type { <128 x i16> }
// CHECK-256-NEXT: %struct.struct_uint32m8 = type { <64 x i32> }
// CHECK-256-NEXT: %struct.struct_uint64m8 = type { <32 x i64> }
// CHECK-256-NEXT: %struct.struct_float32m8 = type { <64 x float> }
// CHECK-256-NEXT: %struct.struct_float64m8 = type { <32 x double> }
// CHECK-256-NEXT: %struct.struct_bool1 = type { <32 x i8> }
// CHECK-256-NEXT: %struct.struct_bool2 = type { <16 x i8> }
// CHECK-256-NEXT: %struct.struct_bool4 = type { <8 x i8> }
// CHECK-256-NEXT: %struct.struct_bool8 = type { <4 x i8> }
// CHECK-256-NEXT: %struct.struct_bool16 = type { <2 x i8> }
// CHECK-256-NEXT: %struct.struct_bool32 = type { <1 x i8> }

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
// CHECK-512-NEXT: %struct.struct_int8m2 = type { <128 x i8> }
// CHECK-512-NEXT: %struct.struct_int16m2 = type { <64 x i16> }
// CHECK-512-NEXT: %struct.struct_int32m2 = type { <32 x i32> }
// CHECK-512-NEXT: %struct.struct_int64m2 = type { <16 x i64> }
// CHECK-512-NEXT: %struct.struct_uint8m2 = type { <128 x i8> }
// CHECK-512-NEXT: %struct.struct_uint16m2 = type { <64 x i16> }
// CHECK-512-NEXT: %struct.struct_uint32m2 = type { <32 x i32> }
// CHECK-512-NEXT: %struct.struct_uint64m2 = type { <16 x i64> }
// CHECK-512-NEXT: %struct.struct_float32m2 = type { <32 x float> }
// CHECK-512-NEXT: %struct.struct_float64m2 = type { <16 x double> }
// CHECK-512-NEXT: %struct.struct_int8m4 = type { <256 x i8> }
// CHECK-512-NEXT: %struct.struct_int16m4 = type { <128 x i16> }
// CHECK-512-NEXT: %struct.struct_int32m4 = type { <64 x i32> }
// CHECK-512-NEXT: %struct.struct_int64m4 = type { <32 x i64> }
// CHECK-512-NEXT: %struct.struct_uint8m4 = type { <256 x i8> }
// CHECK-512-NEXT: %struct.struct_uint16m4 = type { <128 x i16> }
// CHECK-512-NEXT: %struct.struct_uint32m4 = type { <64 x i32> }
// CHECK-512-NEXT: %struct.struct_uint64m4 = type { <32 x i64> }
// CHECK-512-NEXT: %struct.struct_float32m4 = type { <64 x float> }
// CHECK-512-NEXT: %struct.struct_float64m4 = type { <32 x double> }
// CHECK-512-NEXT: %struct.struct_int8m8 = type { <512 x i8> }
// CHECK-512-NEXT: %struct.struct_int16m8 = type { <256 x i16> }
// CHECK-512-NEXT: %struct.struct_int32m8 = type { <128 x i32> }
// CHECK-512-NEXT: %struct.struct_int64m8 = type { <64 x i64> }
// CHECK-512-NEXT: %struct.struct_uint8m8 = type { <512 x i8> }
// CHECK-512-NEXT: %struct.struct_uint16m8 = type { <256 x i16> }
// CHECK-512-NEXT: %struct.struct_uint32m8 = type { <128 x i32> }
// CHECK-512-NEXT: %struct.struct_uint64m8 = type { <64 x i64> }
// CHECK-512-NEXT: %struct.struct_float32m8 = type { <128 x float> }
// CHECK-512-NEXT: %struct.struct_float64m8 = type { <64 x double> }
// CHECK-512-NEXT: %struct.struct_bool1 = type { <64 x i8> }
// CHECK-512-NEXT: %struct.struct_bool2 = type { <32 x i8> }
// CHECK-512-NEXT: %struct.struct_bool4 = type { <16 x i8> }
// CHECK-512-NEXT: %struct.struct_bool8 = type { <8 x i8> }
// CHECK-512-NEXT: %struct.struct_bool16 = type { <4 x i8> }
// CHECK-512-NEXT: %struct.struct_bool32 = type { <2 x i8> }
// CHECK-512-NEXT: %struct.struct_bool64 = type { <1 x i8> }

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
// CHECK-1024-NEXT: %struct.struct_int8m2 = type { <256 x i8> }
// CHECK-1024-NEXT: %struct.struct_int16m2 = type { <128 x i16> }
// CHECK-1024-NEXT: %struct.struct_int32m2 = type { <64 x i32> }
// CHECK-1024-NEXT: %struct.struct_int64m2 = type { <32 x i64> }
// CHECK-1024-NEXT: %struct.struct_uint8m2 = type { <256 x i8> }
// CHECK-1024-NEXT: %struct.struct_uint16m2 = type { <128 x i16> }
// CHECK-1024-NEXT: %struct.struct_uint32m2 = type { <64 x i32> }
// CHECK-1024-NEXT: %struct.struct_uint64m2 = type { <32 x i64> }
// CHECK-1024-NEXT: %struct.struct_float32m2 = type { <64 x float> }
// CHECK-1024-NEXT: %struct.struct_float64m2 = type { <32 x double> }
// CHECK-1024-NEXT: %struct.struct_int8m4 = type { <512 x i8> }
// CHECK-1024-NEXT: %struct.struct_int16m4 = type { <256 x i16> }
// CHECK-1024-NEXT: %struct.struct_int32m4 = type { <128 x i32> }
// CHECK-1024-NEXT: %struct.struct_int64m4 = type { <64 x i64> }
// CHECK-1024-NEXT: %struct.struct_uint8m4 = type { <512 x i8> }
// CHECK-1024-NEXT: %struct.struct_uint16m4 = type { <256 x i16> }
// CHECK-1024-NEXT: %struct.struct_uint32m4 = type { <128 x i32> }
// CHECK-1024-NEXT: %struct.struct_uint64m4 = type { <64 x i64> }
// CHECK-1024-NEXT: %struct.struct_float32m4 = type { <128 x float> }
// CHECK-1024-NEXT: %struct.struct_float64m4 = type { <64 x double> }
// CHECK-1024-NEXT: %struct.struct_int8m8 = type { <1024 x i8> }
// CHECK-1024-NEXT: %struct.struct_int16m8 = type { <512 x i16> }
// CHECK-1024-NEXT: %struct.struct_int32m8 = type { <256 x i32> }
// CHECK-1024-NEXT: %struct.struct_int64m8 = type { <128 x i64> }
// CHECK-1024-NEXT: %struct.struct_uint8m8 = type { <1024 x i8> }
// CHECK-1024-NEXT: %struct.struct_uint16m8 = type { <512 x i16> }
// CHECK-1024-NEXT: %struct.struct_uint32m8 = type { <256 x i32> }
// CHECK-1024-NEXT: %struct.struct_uint64m8 = type { <128 x i64> }
// CHECK-1024-NEXT: %struct.struct_float32m8 = type { <256 x float> }
// CHECK-1024-NEXT: %struct.struct_float64m8 = type { <128 x double> }
// CHECK-1024-NEXT: %struct.struct_bool1 = type { <128 x i8> }
// CHECK-1024-NEXT: %struct.struct_bool2 = type { <64 x i8> }
// CHECK-1024-NEXT: %struct.struct_bool4 = type { <32 x i8> }
// CHECK-1024-NEXT: %struct.struct_bool8 = type { <16 x i8> }
// CHECK-1024-NEXT: %struct.struct_bool16 = type { <8 x i8> }
// CHECK-1024-NEXT: %struct.struct_bool32 = type { <4 x i8> }
// CHECK-1024-NEXT: %struct.struct_bool64 = type { <2 x i8> }

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
// CHECK-64-NEXT: %union.union_int8m2 = type { <16 x i8> }
// CHECK-64-NEXT: %union.union_int16m2 = type { <8 x i16> }
// CHECK-64-NEXT: %union.union_int32m2 = type { <4 x i32> }
// CHECK-64-NEXT: %union.union_int64m2 = type { <2 x i64> }
// CHECK-64-NEXT: %union.union_uint8m2 = type { <16 x i8> }
// CHECK-64-NEXT: %union.union_uint16m2 = type { <8 x i16> }
// CHECK-64-NEXT: %union.union_uint32m2 = type { <4 x i32> }
// CHECK-64-NEXT: %union.union_uint64m2 = type { <2 x i64> }
// CHECK-64-NEXT: %union.union_float32m2 = type { <4 x float> }
// CHECK-64-NEXT: %union.union_float64m2 = type { <2 x double> }
// CHECK-64-NEXT: %union.union_int8m4 = type { <32 x i8> }
// CHECK-64-NEXT: %union.union_int16m4 = type { <16 x i16> }
// CHECK-64-NEXT: %union.union_int32m4 = type { <8 x i32> }
// CHECK-64-NEXT: %union.union_int64m4 = type { <4 x i64> }
// CHECK-64-NEXT: %union.union_uint8m4 = type { <32 x i8> }
// CHECK-64-NEXT: %union.union_uint16m4 = type { <16 x i16> }
// CHECK-64-NEXT: %union.union_uint32m4 = type { <8 x i32> }
// CHECK-64-NEXT: %union.union_uint64m4 = type { <4 x i64> }
// CHECK-64-NEXT: %union.union_float32m4 = type { <8 x float> }
// CHECK-64-NEXT: %union.union_float64m4 = type { <4 x double> }
// CHECK-64-NEXT: %union.union_int8m8 = type { <64 x i8> }
// CHECK-64-NEXT: %union.union_int16m8 = type { <32 x i16> }
// CHECK-64-NEXT: %union.union_int32m8 = type { <16 x i32> }
// CHECK-64-NEXT: %union.union_int64m8 = type { <8 x i64> }
// CHECK-64-NEXT: %union.union_uint8m8 = type { <64 x i8> }
// CHECK-64-NEXT: %union.union_uint16m8 = type { <32 x i16> }
// CHECK-64-NEXT: %union.union_uint32m8 = type { <16 x i32> }
// CHECK-64-NEXT: %union.union_uint64m8 = type { <8 x i64> }
// CHECK-64-NEXT: %union.union_float32m8 = type { <16 x float> }
// CHECK-64-NEXT: %union.union_float64m8 = type { <8 x double> }
// CHECK-64-NEXT: %union.union_bool1 = type { <8 x i8> }
// CHECK-64-NEXT: %union.union_bool2 = type { <4 x i8> }
// CHECK-64-NEXT: %union.union_bool4 = type { <2 x i8> }
// CHECK-64-NEXT: %union.union_bool8 = type { <1 x i8> }

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
// CHECK-128-NEXT: %union.union_int8m2 = type { <32 x i8> }
// CHECK-128-NEXT: %union.union_int16m2 = type { <16 x i16> }
// CHECK-128-NEXT: %union.union_int32m2 = type { <8 x i32> }
// CHECK-128-NEXT: %union.union_int64m2 = type { <4 x i64> }
// CHECK-128-NEXT: %union.union_uint8m2 = type { <32 x i8> }
// CHECK-128-NEXT: %union.union_uint16m2 = type { <16 x i16> }
// CHECK-128-NEXT: %union.union_uint32m2 = type { <8 x i32> }
// CHECK-128-NEXT: %union.union_uint64m2 = type { <4 x i64> }
// CHECK-128-NEXT: %union.union_float32m2 = type { <8 x float> }
// CHECK-128-NEXT: %union.union_float64m2 = type { <4 x double> }
// CHECK-128-NEXT: %union.union_int8m4 = type { <64 x i8> }
// CHECK-128-NEXT: %union.union_int16m4 = type { <32 x i16> }
// CHECK-128-NEXT: %union.union_int32m4 = type { <16 x i32> }
// CHECK-128-NEXT: %union.union_int64m4 = type { <8 x i64> }
// CHECK-128-NEXT: %union.union_uint8m4 = type { <64 x i8> }
// CHECK-128-NEXT: %union.union_uint16m4 = type { <32 x i16> }
// CHECK-128-NEXT: %union.union_uint32m4 = type { <16 x i32> }
// CHECK-128-NEXT: %union.union_uint64m4 = type { <8 x i64> }
// CHECK-128-NEXT: %union.union_float32m4 = type { <16 x float> }
// CHECK-128-NEXT: %union.union_float64m4 = type { <8 x double> }
// CHECK-128-NEXT: %union.union_int8m8 = type { <128 x i8> }
// CHECK-128-NEXT: %union.union_int16m8 = type { <64 x i16> }
// CHECK-128-NEXT: %union.union_int32m8 = type { <32 x i32> }
// CHECK-128-NEXT: %union.union_int64m8 = type { <16 x i64> }
// CHECK-128-NEXT: %union.union_uint8m8 = type { <128 x i8> }
// CHECK-128-NEXT: %union.union_uint16m8 = type { <64 x i16> }
// CHECK-128-NEXT: %union.union_uint32m8 = type { <32 x i32> }
// CHECK-128-NEXT: %union.union_uint64m8 = type { <16 x i64> }
// CHECK-128-NEXT: %union.union_float32m8 = type { <32 x float> }
// CHECK-128-NEXT: %union.union_float64m8 = type { <16 x double> }
// CHECK-128-NEXT: %union.union_bool1 = type { <16 x i8> }
// CHECK-128-NEXT: %union.union_bool2 = type { <8 x i8> }
// CHECK-128-NEXT: %union.union_bool4 = type { <4 x i8> }
// CHECK-128-NEXT: %union.union_bool8 = type { <2 x i8> }
// CHECK-128-NEXT: %union.union_bool16 = type { <1 x i8> }

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
// CHECK-256-NEXT: %union.union_int8m2 = type { <64 x i8> }
// CHECK-256-NEXT: %union.union_int16m2 = type { <32 x i16> }
// CHECK-256-NEXT: %union.union_int32m2 = type { <16 x i32> }
// CHECK-256-NEXT: %union.union_int64m2 = type { <8 x i64> }
// CHECK-256-NEXT: %union.union_uint8m2 = type { <64 x i8> }
// CHECK-256-NEXT: %union.union_uint16m2 = type { <32 x i16> }
// CHECK-256-NEXT: %union.union_uint32m2 = type { <16 x i32> }
// CHECK-256-NEXT: %union.union_uint64m2 = type { <8 x i64> }
// CHECK-256-NEXT: %union.union_float32m2 = type { <16 x float> }
// CHECK-256-NEXT: %union.union_float64m2 = type { <8 x double> }
// CHECK-256-NEXT: %union.union_int8m4 = type { <128 x i8> }
// CHECK-256-NEXT: %union.union_int16m4 = type { <64 x i16> }
// CHECK-256-NEXT: %union.union_int32m4 = type { <32 x i32> }
// CHECK-256-NEXT: %union.union_int64m4 = type { <16 x i64> }
// CHECK-256-NEXT: %union.union_uint8m4 = type { <128 x i8> }
// CHECK-256-NEXT: %union.union_uint16m4 = type { <64 x i16> }
// CHECK-256-NEXT: %union.union_uint32m4 = type { <32 x i32> }
// CHECK-256-NEXT: %union.union_uint64m4 = type { <16 x i64> }
// CHECK-256-NEXT: %union.union_float32m4 = type { <32 x float> }
// CHECK-256-NEXT: %union.union_float64m4 = type { <16 x double> }
// CHECK-256-NEXT: %union.union_int8m8 = type { <256 x i8> }
// CHECK-256-NEXT: %union.union_int16m8 = type { <128 x i16> }
// CHECK-256-NEXT: %union.union_int32m8 = type { <64 x i32> }
// CHECK-256-NEXT: %union.union_int64m8 = type { <32 x i64> }
// CHECK-256-NEXT: %union.union_uint8m8 = type { <256 x i8> }
// CHECK-256-NEXT: %union.union_uint16m8 = type { <128 x i16> }
// CHECK-256-NEXT: %union.union_uint32m8 = type { <64 x i32> }
// CHECK-256-NEXT: %union.union_uint64m8 = type { <32 x i64> }
// CHECK-256-NEXT: %union.union_float32m8 = type { <64 x float> }
// CHECK-256-NEXT: %union.union_float64m8 = type { <32 x double> }
// CHECK-256-NEXT: %union.union_bool1 = type { <32 x i8> }
// CHECK-256-NEXT: %union.union_bool2 = type { <16 x i8> }
// CHECK-256-NEXT: %union.union_bool4 = type { <8 x i8> }
// CHECK-256-NEXT: %union.union_bool8 = type { <4 x i8> }
// CHECK-256-NEXT: %union.union_bool16 = type { <2 x i8> }
// CHECK-256-NEXT: %union.union_bool32 = type { <1 x i8> }

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
// CHECK-512-NEXT: %union.union_int8m2 = type { <128 x i8> }
// CHECK-512-NEXT: %union.union_int16m2 = type { <64 x i16> }
// CHECK-512-NEXT: %union.union_int32m2 = type { <32 x i32> }
// CHECK-512-NEXT: %union.union_int64m2 = type { <16 x i64> }
// CHECK-512-NEXT: %union.union_uint8m2 = type { <128 x i8> }
// CHECK-512-NEXT: %union.union_uint16m2 = type { <64 x i16> }
// CHECK-512-NEXT: %union.union_uint32m2 = type { <32 x i32> }
// CHECK-512-NEXT: %union.union_uint64m2 = type { <16 x i64> }
// CHECK-512-NEXT: %union.union_float32m2 = type { <32 x float> }
// CHECK-512-NEXT: %union.union_float64m2 = type { <16 x double> }
// CHECK-512-NEXT: %union.union_int8m4 = type { <256 x i8> }
// CHECK-512-NEXT: %union.union_int16m4 = type { <128 x i16> }
// CHECK-512-NEXT: %union.union_int32m4 = type { <64 x i32> }
// CHECK-512-NEXT: %union.union_int64m4 = type { <32 x i64> }
// CHECK-512-NEXT: %union.union_uint8m4 = type { <256 x i8> }
// CHECK-512-NEXT: %union.union_uint16m4 = type { <128 x i16> }
// CHECK-512-NEXT: %union.union_uint32m4 = type { <64 x i32> }
// CHECK-512-NEXT: %union.union_uint64m4 = type { <32 x i64> }
// CHECK-512-NEXT: %union.union_float32m4 = type { <64 x float> }
// CHECK-512-NEXT: %union.union_float64m4 = type { <32 x double> }
// CHECK-512-NEXT: %union.union_int8m8 = type { <512 x i8> }
// CHECK-512-NEXT: %union.union_int16m8 = type { <256 x i16> }
// CHECK-512-NEXT: %union.union_int32m8 = type { <128 x i32> }
// CHECK-512-NEXT: %union.union_int64m8 = type { <64 x i64> }
// CHECK-512-NEXT: %union.union_uint8m8 = type { <512 x i8> }
// CHECK-512-NEXT: %union.union_uint16m8 = type { <256 x i16> }
// CHECK-512-NEXT: %union.union_uint32m8 = type { <128 x i32> }
// CHECK-512-NEXT: %union.union_uint64m8 = type { <64 x i64> }
// CHECK-512-NEXT: %union.union_float32m8 = type { <128 x float> }
// CHECK-512-NEXT: %union.union_float64m8 = type { <64 x double> }
// CHECK-512-NEXT: %union.union_bool1 = type { <64 x i8> }
// CHECK-512-NEXT: %union.union_bool2 = type { <32 x i8> }
// CHECK-512-NEXT: %union.union_bool4 = type { <16 x i8> }
// CHECK-512-NEXT: %union.union_bool8 = type { <8 x i8> }
// CHECK-512-NEXT: %union.union_bool16 = type { <4 x i8> }
// CHECK-512-NEXT: %union.union_bool32 = type { <2 x i8> }
// CHECK-512-NEXT: %union.union_bool64 = type { <1 x i8> }

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
// CHECK-1024-NEXT: %union.union_int8m2 = type { <256 x i8> }
// CHECK-1024-NEXT: %union.union_int16m2 = type { <128 x i16> }
// CHECK-1024-NEXT: %union.union_int32m2 = type { <64 x i32> }
// CHECK-1024-NEXT: %union.union_int64m2 = type { <32 x i64> }
// CHECK-1024-NEXT: %union.union_uint8m2 = type { <256 x i8> }
// CHECK-1024-NEXT: %union.union_uint16m2 = type { <128 x i16> }
// CHECK-1024-NEXT: %union.union_uint32m2 = type { <64 x i32> }
// CHECK-1024-NEXT: %union.union_uint64m2 = type { <32 x i64> }
// CHECK-1024-NEXT: %union.union_float32m2 = type { <64 x float> }
// CHECK-1024-NEXT: %union.union_float64m2 = type { <32 x double> }
// CHECK-1024-NEXT: %union.union_int8m4 = type { <512 x i8> }
// CHECK-1024-NEXT: %union.union_int16m4 = type { <256 x i16> }
// CHECK-1024-NEXT: %union.union_int32m4 = type { <128 x i32> }
// CHECK-1024-NEXT: %union.union_int64m4 = type { <64 x i64> }
// CHECK-1024-NEXT: %union.union_uint8m4 = type { <512 x i8> }
// CHECK-1024-NEXT: %union.union_uint16m4 = type { <256 x i16> }
// CHECK-1024-NEXT: %union.union_uint32m4 = type { <128 x i32> }
// CHECK-1024-NEXT: %union.union_uint64m4 = type { <64 x i64> }
// CHECK-1024-NEXT: %union.union_float32m4 = type { <128 x float> }
// CHECK-1024-NEXT: %union.union_float64m4 = type { <64 x double> }
// CHECK-1024-NEXT: %union.union_int8m8 = type { <1024 x i8> }
// CHECK-1024-NEXT: %union.union_int16m8 = type { <512 x i16> }
// CHECK-1024-NEXT: %union.union_int32m8 = type { <256 x i32> }
// CHECK-1024-NEXT: %union.union_int64m8 = type { <128 x i64> }
// CHECK-1024-NEXT: %union.union_uint8m8 = type { <1024 x i8> }
// CHECK-1024-NEXT: %union.union_uint16m8 = type { <512 x i16> }
// CHECK-1024-NEXT: %union.union_uint32m8 = type { <256 x i32> }
// CHECK-1024-NEXT: %union.union_uint64m8 = type { <128 x i64> }
// CHECK-1024-NEXT: %union.union_float32m8 = type { <256 x float> }
// CHECK-1024-NEXT: %union.union_float64m8 = type { <128 x double> }
// CHECK-1024-NEXT: %union.union_bool1 = type { <128 x i8> }
// CHECK-1024-NEXT: %union.union_bool2 = type { <64 x i8> }
// CHECK-1024-NEXT: %union.union_bool4 = type { <32 x i8> }
// CHECK-1024-NEXT: %union.union_bool8 = type { <16 x i8> }
// CHECK-1024-NEXT: %union.union_bool16 = type { <8 x i8> }
// CHECK-1024-NEXT: %union.union_bool32 = type { <4 x i8> }
// CHECK-1024-NEXT: %union.union_bool64 = type { <2 x i8> }

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
// CHECK-64-NEXT: @global_i8m2 ={{.*}} global <16 x i8> zeroinitializer, align 8
// CHECK-64-NEXT: @global_i16m2 ={{.*}} global <8 x i16> zeroinitializer, align 8
// CHECK-64-NEXT: @global_i32m2 ={{.*}} global <4 x i32> zeroinitializer, align 8
// CHECK-64-NEXT: @global_i64m2 ={{.*}} global <2 x i64> zeroinitializer, align 8
// CHECK-64-NEXT: @global_u8m2 ={{.*}} global <16 x i8> zeroinitializer, align 8
// CHECK-64-NEXT: @global_u16m2 ={{.*}} global <8 x i16> zeroinitializer, align 8
// CHECK-64-NEXT: @global_u32m2 ={{.*}} global <4 x i32> zeroinitializer, align 8
// CHECK-64-NEXT: @global_u64m2 ={{.*}} global <2 x i64> zeroinitializer, align 8
// CHECK-64-NEXT: @global_f32m2 ={{.*}} global <4 x float> zeroinitializer, align 8
// CHECK-64-NEXT: @global_f64m2 ={{.*}} global <2 x double> zeroinitializer, align 8
// CHECK-64-NEXT: @global_i8m4 ={{.*}} global <32 x i8> zeroinitializer, align 8
// CHECK-64-NEXT: @global_i16m4 ={{.*}} global <16 x i16> zeroinitializer, align 8
// CHECK-64-NEXT: @global_i32m4 ={{.*}} global <8 x i32> zeroinitializer, align 8
// CHECK-64-NEXT: @global_i64m4 ={{.*}} global <4 x i64> zeroinitializer, align 8
// CHECK-64-NEXT: @global_u8m4 ={{.*}} global <32 x i8> zeroinitializer, align 8
// CHECK-64-NEXT: @global_u16m4 ={{.*}} global <16 x i16> zeroinitializer, align 8
// CHECK-64-NEXT: @global_u32m4 ={{.*}} global <8 x i32> zeroinitializer, align 8
// CHECK-64-NEXT: @global_u64m4 ={{.*}} global <4 x i64> zeroinitializer, align 8
// CHECK-64-NEXT: @global_f32m4 ={{.*}} global <8 x float> zeroinitializer, align 8
// CHECK-64-NEXT: @global_f64m4 ={{.*}} global <4 x double> zeroinitializer, align 8
// CHECK-64-NEXT: @global_i8m8 ={{.*}} global <64 x i8> zeroinitializer, align 8
// CHECK-64-NEXT: @global_i16m8 ={{.*}} global <32 x i16> zeroinitializer, align 8
// CHECK-64-NEXT: @global_i32m8 ={{.*}} global <16 x i32> zeroinitializer, align 8
// CHECK-64-NEXT: @global_i64m8 ={{.*}} global <8 x i64> zeroinitializer, align 8
// CHECK-64-NEXT: @global_u8m8 ={{.*}} global <64 x i8> zeroinitializer, align 8
// CHECK-64-NEXT: @global_u16m8 ={{.*}} global <32 x i16> zeroinitializer, align 8
// CHECK-64-NEXT: @global_u32m8 ={{.*}} global <16 x i32> zeroinitializer, align 8
// CHECK-64-NEXT: @global_u64m8 ={{.*}} global <8 x i64> zeroinitializer, align 8
// CHECK-64-NEXT: @global_f32m8 ={{.*}} global <16 x float> zeroinitializer, align 8
// CHECK-64-NEXT: @global_f64m8 ={{.*}} global <8 x double> zeroinitializer, align 8
// CHECK-64-NEXT: @global_bool1 ={{.*}} global <8 x i8> zeroinitializer, align 8
// CHECK-64-NEXT: @global_bool2 ={{.*}} global <4 x i8> zeroinitializer, align 4
// CHECK-64-NEXT: @global_bool4 ={{.*}} global <2 x i8> zeroinitializer, align 2
// CHECK-64-NEXT: @global_bool8 ={{.*}} global <1 x i8> zeroinitializer, align 1

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
// CHECK-128-NEXT: @global_i8m2 ={{.*}} global <32 x i8> zeroinitializer, align 8
// CHECK-128-NEXT: @global_i16m2 ={{.*}} global <16 x i16> zeroinitializer, align 8
// CHECK-128-NEXT: @global_i32m2 ={{.*}} global <8 x i32> zeroinitializer, align 8
// CHECK-128-NEXT: @global_i64m2 ={{.*}} global <4 x i64> zeroinitializer, align 8
// CHECK-128-NEXT: @global_u8m2 ={{.*}} global <32 x i8> zeroinitializer, align 8
// CHECK-128-NEXT: @global_u16m2 ={{.*}} global <16 x i16> zeroinitializer, align 8
// CHECK-128-NEXT: @global_u32m2 ={{.*}} global <8 x i32> zeroinitializer, align 8
// CHECK-128-NEXT: @global_u64m2 ={{.*}} global <4 x i64> zeroinitializer, align 8
// CHECK-128-NEXT: @global_f32m2 ={{.*}} global <8 x float> zeroinitializer, align 8
// CHECK-128-NEXT: @global_f64m2 ={{.*}} global <4 x double> zeroinitializer, align 8
// CHECK-128-NEXT: @global_i8m4 ={{.*}} global <64 x i8> zeroinitializer, align 8
// CHECK-128-NEXT: @global_i16m4 ={{.*}} global <32 x i16> zeroinitializer, align 8
// CHECK-128-NEXT: @global_i32m4 ={{.*}} global <16 x i32> zeroinitializer, align 8
// CHECK-128-NEXT: @global_i64m4 ={{.*}} global <8 x i64> zeroinitializer, align 8
// CHECK-128-NEXT: @global_u8m4 ={{.*}} global <64 x i8> zeroinitializer, align 8
// CHECK-128-NEXT: @global_u16m4 ={{.*}} global <32 x i16> zeroinitializer, align 8
// CHECK-128-NEXT: @global_u32m4 ={{.*}} global <16 x i32> zeroinitializer, align 8
// CHECK-128-NEXT: @global_u64m4 ={{.*}} global <8 x i64> zeroinitializer, align 8
// CHECK-128-NEXT: @global_f32m4 ={{.*}} global <16 x float> zeroinitializer, align 8
// CHECK-128-NEXT: @global_f64m4 ={{.*}} global <8 x double> zeroinitializer, align 8
// CHECK-128-NEXT: @global_i8m8 ={{.*}} global <128 x i8> zeroinitializer, align 8
// CHECK-128-NEXT: @global_i16m8 ={{.*}} global <64 x i16> zeroinitializer, align 8
// CHECK-128-NEXT: @global_i32m8 ={{.*}} global <32 x i32> zeroinitializer, align 8
// CHECK-128-NEXT: @global_i64m8 ={{.*}} global <16 x i64> zeroinitializer, align 8
// CHECK-128-NEXT: @global_u8m8 ={{.*}} global <128 x i8> zeroinitializer, align 8
// CHECK-128-NEXT: @global_u16m8 ={{.*}} global <64 x i16> zeroinitializer, align 8
// CHECK-128-NEXT: @global_u32m8 ={{.*}} global <32 x i32> zeroinitializer, align 8
// CHECK-128-NEXT: @global_u64m8 ={{.*}} global <16 x i64> zeroinitializer, align 8
// CHECK-128-NEXT: @global_f32m8 ={{.*}} global <32 x float> zeroinitializer, align 8
// CHECK-128-NEXT: @global_f64m8 ={{.*}} global <16 x double> zeroinitializer, align 8
// CHECK-128-NEXT: @global_bool1 ={{.*}} global <16 x i8> zeroinitializer, align 8
// CHECK-128-NEXT: @global_bool2 ={{.*}} global <8 x i8> zeroinitializer, align 8
// CHECK-128-NEXT: @global_bool4 ={{.*}} global <4 x i8> zeroinitializer, align 4
// CHECK-128-NEXT: @global_bool8 ={{.*}} global <2 x i8> zeroinitializer, align 2
// CHECK-128-NEXT: @global_bool16 ={{.*}} global <1 x i8> zeroinitializer, align 1

// CHECK-256:      @global_i8 ={{.*}} global <32 x i8> zeroinitializer, align 8
// CHECK-256-NEXT: @global_i16 ={{.*}} global <16 x i16> zeroinitializer, align 8
// CHECK-256-NEXT: @global_i32 ={{.*}} global <8 x i32> zeroinitializer, align 8
// CHECK-256-NEXT: @global_i64 ={{.*}} global <4 x i64> zeroinitializer, align 8
// CHECK-256-NEXT: @global_u8 ={{.*}} global <32 x i8> zeroinitializer, align 8
// CHECK-256-NEXT: @global_u16 ={{.*}} global <16 x i16> zeroinitializer, align 8
// CHECK-256-NEXT: @global_u32 ={{.*}} global <8 x i32> zeroinitializer, align 8
// CHECK-256-NEXT: @global_u64 ={{.*}} global <4 x i64> zeroinitializer, align 8
// CHECK-256-NEXT: @global_f32 ={{.*}} global <8 x float> zeroinitializer, align 8
// CHECK-256-NEXT: @global_f64 ={{.*}} global <4 x double> zeroinitializer, align 8
// CHECK-256-NEXT: @global_i8m2 ={{.*}} global <64 x i8> zeroinitializer, align 8
// CHECK-256-NEXT: @global_i16m2 ={{.*}} global <32 x i16> zeroinitializer, align 8
// CHECK-256-NEXT: @global_i32m2 ={{.*}} global <16 x i32> zeroinitializer, align 8
// CHECK-256-NEXT: @global_i64m2 ={{.*}} global <8 x i64> zeroinitializer, align 8
// CHECK-256-NEXT: @global_u8m2 ={{.*}} global <64 x i8> zeroinitializer, align 8
// CHECK-256-NEXT: @global_u16m2 ={{.*}} global <32 x i16> zeroinitializer, align 8
// CHECK-256-NEXT: @global_u32m2 ={{.*}} global <16 x i32> zeroinitializer, align 8
// CHECK-256-NEXT: @global_u64m2 ={{.*}} global <8 x i64> zeroinitializer, align 8
// CHECK-256-NEXT: @global_f32m2 ={{.*}} global <16 x float> zeroinitializer, align 8
// CHECK-256-NEXT: @global_f64m2 ={{.*}} global <8 x double> zeroinitializer, align 8
// CHECK-256-NEXT: @global_i8m4 ={{.*}} global <128 x i8> zeroinitializer, align 8
// CHECK-256-NEXT: @global_i16m4 ={{.*}} global <64 x i16> zeroinitializer, align 8
// CHECK-256-NEXT: @global_i32m4 ={{.*}} global <32 x i32> zeroinitializer, align 8
// CHECK-256-NEXT: @global_i64m4 ={{.*}} global <16 x i64> zeroinitializer, align 8
// CHECK-256-NEXT: @global_u8m4 ={{.*}} global <128 x i8> zeroinitializer, align 8
// CHECK-256-NEXT: @global_u16m4 ={{.*}} global <64 x i16> zeroinitializer, align 8
// CHECK-256-NEXT: @global_u32m4 ={{.*}} global <32 x i32> zeroinitializer, align 8
// CHECK-256-NEXT: @global_u64m4 ={{.*}} global <16 x i64> zeroinitializer, align 8
// CHECK-256-NEXT: @global_f32m4 ={{.*}} global <32 x float> zeroinitializer, align 8
// CHECK-256-NEXT: @global_f64m4 ={{.*}} global <16 x double> zeroinitializer, align 8
// CHECK-256-NEXT: @global_i8m8 ={{.*}} global <256 x i8> zeroinitializer, align 8
// CHECK-256-NEXT: @global_i16m8 ={{.*}} global <128 x i16> zeroinitializer, align 8
// CHECK-256-NEXT: @global_i32m8 ={{.*}} global <64 x i32> zeroinitializer, align 8
// CHECK-256-NEXT: @global_i64m8 ={{.*}} global <32 x i64> zeroinitializer, align 8
// CHECK-256-NEXT: @global_u8m8 ={{.*}} global <256 x i8> zeroinitializer, align 8
// CHECK-256-NEXT: @global_u16m8 ={{.*}} global <128 x i16> zeroinitializer, align 8
// CHECK-256-NEXT: @global_u32m8 ={{.*}} global <64 x i32> zeroinitializer, align 8
// CHECK-256-NEXT: @global_u64m8 ={{.*}} global <32 x i64> zeroinitializer, align 8
// CHECK-256-NEXT: @global_f32m8 ={{.*}} global <64 x float> zeroinitializer, align 8
// CHECK-256-NEXT: @global_f64m8 ={{.*}} global <32 x double> zeroinitializer, align 8
// CHECK-256-NEXT: @global_bool1 ={{.*}} global <32 x i8> zeroinitializer, align 8
// CHECK-256-NEXT: @global_bool2 ={{.*}} global <16 x i8> zeroinitializer, align 8
// CHECK-256-NEXT: @global_bool4 ={{.*}} global <8 x i8> zeroinitializer, align 8
// CHECK-256-NEXT: @global_bool8 ={{.*}} global <4 x i8> zeroinitializer, align 4
// CHECK-256-NEXT: @global_bool16 ={{.*}} global <2 x i8> zeroinitializer, align 2
// CHECK-256-NEXT: @global_bool32 ={{.*}} global <1 x i8> zeroinitializer, align 1

// CHECK-512:      @global_i8 ={{.*}} global <64 x i8> zeroinitializer, align 8
// CHECK-512-NEXT: @global_i16 ={{.*}} global <32 x i16> zeroinitializer, align 8
// CHECK-512-NEXT: @global_i32 ={{.*}} global <16 x i32> zeroinitializer, align 8
// CHECK-512-NEXT: @global_i64 ={{.*}} global <8 x i64> zeroinitializer, align 8
// CHECK-512-NEXT: @global_u8 ={{.*}} global <64 x i8> zeroinitializer, align 8
// CHECK-512-NEXT: @global_u16 ={{.*}} global <32 x i16> zeroinitializer, align 8
// CHECK-512-NEXT: @global_u32 ={{.*}} global <16 x i32> zeroinitializer, align 8
// CHECK-512-NEXT: @global_u64 ={{.*}} global <8 x i64> zeroinitializer, align 8
// CHECK-512-NEXT: @global_f32 ={{.*}} global <16 x float> zeroinitializer, align 8
// CHECK-512-NEXT: @global_f64 ={{.*}} global <8 x double> zeroinitializer, align 8
// CHECK-512-NEXT: @global_i8m2 ={{.*}} global <128 x i8> zeroinitializer, align 8
// CHECK-512-NEXT: @global_i16m2 ={{.*}} global <64 x i16> zeroinitializer, align 8
// CHECK-512-NEXT: @global_i32m2 ={{.*}} global <32 x i32> zeroinitializer, align 8
// CHECK-512-NEXT: @global_i64m2 ={{.*}} global <16 x i64> zeroinitializer, align 8
// CHECK-512-NEXT: @global_u8m2 ={{.*}} global <128 x i8> zeroinitializer, align 8
// CHECK-512-NEXT: @global_u16m2 ={{.*}} global <64 x i16> zeroinitializer, align 8
// CHECK-512-NEXT: @global_u32m2 ={{.*}} global <32 x i32> zeroinitializer, align 8
// CHECK-512-NEXT: @global_u64m2 ={{.*}} global <16 x i64> zeroinitializer, align 8
// CHECK-512-NEXT: @global_f32m2 ={{.*}} global <32 x float> zeroinitializer, align 8
// CHECK-512-NEXT: @global_f64m2 ={{.*}} global <16 x double> zeroinitializer, align 8
// CHECK-512-NEXT: @global_i8m4 ={{.*}} global <256 x i8> zeroinitializer, align 8
// CHECK-512-NEXT: @global_i16m4 ={{.*}} global <128 x i16> zeroinitializer, align 8
// CHECK-512-NEXT: @global_i32m4 ={{.*}} global <64 x i32> zeroinitializer, align 8
// CHECK-512-NEXT: @global_i64m4 ={{.*}} global <32 x i64> zeroinitializer, align 8
// CHECK-512-NEXT: @global_u8m4 ={{.*}} global <256 x i8> zeroinitializer, align 8
// CHECK-512-NEXT: @global_u16m4 ={{.*}} global <128 x i16> zeroinitializer, align 8
// CHECK-512-NEXT: @global_u32m4 ={{.*}} global <64 x i32> zeroinitializer, align 8
// CHECK-512-NEXT: @global_u64m4 ={{.*}} global <32 x i64> zeroinitializer, align 8
// CHECK-512-NEXT: @global_f32m4 ={{.*}} global <64 x float> zeroinitializer, align 8
// CHECK-512-NEXT: @global_f64m4 ={{.*}} global <32 x double> zeroinitializer, align 8
// CHECK-512-NEXT: @global_i8m8 ={{.*}} global <512 x i8> zeroinitializer, align 8
// CHECK-512-NEXT: @global_i16m8 ={{.*}} global <256 x i16> zeroinitializer, align 8
// CHECK-512-NEXT: @global_i32m8 ={{.*}} global <128 x i32> zeroinitializer, align 8
// CHECK-512-NEXT: @global_i64m8 ={{.*}} global <64 x i64> zeroinitializer, align 8
// CHECK-512-NEXT: @global_u8m8 ={{.*}} global <512 x i8> zeroinitializer, align 8
// CHECK-512-NEXT: @global_u16m8 ={{.*}} global <256 x i16> zeroinitializer, align 8
// CHECK-512-NEXT: @global_u32m8 ={{.*}} global <128 x i32> zeroinitializer, align 8
// CHECK-512-NEXT: @global_u64m8 ={{.*}} global <64 x i64> zeroinitializer, align 8
// CHECK-512-NEXT: @global_f32m8 ={{.*}} global <128 x float> zeroinitializer, align 8
// CHECK-512-NEXT: @global_f64m8 ={{.*}} global <64 x double> zeroinitializer, align 8
// CHECK-512-NEXT: @global_bool1 ={{.*}} global <64 x i8> zeroinitializer, align 8
// CHECK-512-NEXT: @global_bool2 ={{.*}} global <32 x i8> zeroinitializer, align 8
// CHECK-512-NEXT: @global_bool4 ={{.*}} global <16 x i8> zeroinitializer, align 8
// CHECK-512-NEXT: @global_bool8 ={{.*}} global <8 x i8> zeroinitializer, align 8
// CHECK-512-NEXT: @global_bool16 ={{.*}} global <4 x i8> zeroinitializer, align 4
// CHECK-512-NEXT: @global_bool32 ={{.*}} global <2 x i8> zeroinitializer, align 2
// CHECK-512-NEXT: @global_bool64 ={{.*}} global <1 x i8> zeroinitializer, align 1

// CHECK-1024:      @global_i8 ={{.*}} global <128 x i8> zeroinitializer, align 8
// CHECK-1024-NEXT: @global_i16 ={{.*}} global <64 x i16> zeroinitializer, align 8
// CHECK-1024-NEXT: @global_i32 ={{.*}} global <32 x i32> zeroinitializer, align 8
// CHECK-1024-NEXT: @global_i64 ={{.*}} global <16 x i64> zeroinitializer, align 8
// CHECK-1024-NEXT: @global_u8 ={{.*}} global <128 x i8> zeroinitializer, align 8
// CHECK-1024-NEXT: @global_u16 ={{.*}} global <64 x i16> zeroinitializer, align 8
// CHECK-1024-NEXT: @global_u32 ={{.*}} global <32 x i32> zeroinitializer, align 8
// CHECK-1024-NEXT: @global_u64 ={{.*}} global <16 x i64> zeroinitializer, align 8
// CHECK-1024-NEXT: @global_f32 ={{.*}} global <32 x float> zeroinitializer, align 8
// CHECK-1024-NEXT: @global_f64 ={{.*}} global <16 x double> zeroinitializer, align 8
// CHECK-1024-NEXT: @global_i8m2 ={{.*}} global <256 x i8> zeroinitializer, align 8
// CHECK-1024-NEXT: @global_i16m2 ={{.*}} global <128 x i16> zeroinitializer, align 8
// CHECK-1024-NEXT: @global_i32m2 ={{.*}} global <64 x i32> zeroinitializer, align 8
// CHECK-1024-NEXT: @global_i64m2 ={{.*}} global <32 x i64> zeroinitializer, align 8
// CHECK-1024-NEXT: @global_u8m2 ={{.*}} global <256 x i8> zeroinitializer, align 8
// CHECK-1024-NEXT: @global_u16m2 ={{.*}} global <128 x i16> zeroinitializer, align 8
// CHECK-1024-NEXT: @global_u32m2 ={{.*}} global <64 x i32> zeroinitializer, align 8
// CHECK-1024-NEXT: @global_u64m2 ={{.*}} global <32 x i64> zeroinitializer, align 8
// CHECK-1024-NEXT: @global_f32m2 ={{.*}} global <64 x float> zeroinitializer, align 8
// CHECK-1024-NEXT: @global_f64m2 ={{.*}} global <32 x double> zeroinitializer, align 8
// CHECK-1024-NEXT: @global_i8m4 ={{.*}} global <512 x i8> zeroinitializer, align 8
// CHECK-1024-NEXT: @global_i16m4 ={{.*}} global <256 x i16> zeroinitializer, align 8
// CHECK-1024-NEXT: @global_i32m4 ={{.*}} global <128 x i32> zeroinitializer, align 8
// CHECK-1024-NEXT: @global_i64m4 ={{.*}} global <64 x i64> zeroinitializer, align 8
// CHECK-1024-NEXT: @global_u8m4 ={{.*}} global <512 x i8> zeroinitializer, align 8
// CHECK-1024-NEXT: @global_u16m4 ={{.*}} global <256 x i16> zeroinitializer, align 8
// CHECK-1024-NEXT: @global_u32m4 ={{.*}} global <128 x i32> zeroinitializer, align 8
// CHECK-1024-NEXT: @global_u64m4 ={{.*}} global <64 x i64> zeroinitializer, align 8
// CHECK-1024-NEXT: @global_f32m4 ={{.*}} global <128 x float> zeroinitializer, align 8
// CHECK-1024-NEXT: @global_f64m4 ={{.*}} global <64 x double> zeroinitializer, align 8
// CHECK-1024-NEXT: @global_i8m8 ={{.*}} global <1024 x i8> zeroinitializer, align 8
// CHECK-1024-NEXT: @global_i16m8 ={{.*}} global <512 x i16> zeroinitializer, align 8
// CHECK-1024-NEXT: @global_i32m8 ={{.*}} global <256 x i32> zeroinitializer, align 8
// CHECK-1024-NEXT: @global_i64m8 ={{.*}} global <128 x i64> zeroinitializer, align 8
// CHECK-1024-NEXT: @global_u8m8 ={{.*}} global <1024 x i8> zeroinitializer, align 8
// CHECK-1024-NEXT: @global_u16m8 ={{.*}} global <512 x i16> zeroinitializer, align 8
// CHECK-1024-NEXT: @global_u32m8 ={{.*}} global <256 x i32> zeroinitializer, align 8
// CHECK-1024-NEXT: @global_u64m8 ={{.*}} global <128 x i64> zeroinitializer, align 8
// CHECK-1024-NEXT: @global_f32m8 ={{.*}} global <256 x float> zeroinitializer, align 8
// CHECK-1024-NEXT: @global_f64m8 ={{.*}} global <128 x double> zeroinitializer, align 8
// CHECK-1024-NEXT: @global_bool1 ={{.*}} global <128 x i8> zeroinitializer, align 8
// CHECK-1024-NEXT: @global_bool2 ={{.*}} global <64 x i8> zeroinitializer, align 8
// CHECK-1024-NEXT: @global_bool4 ={{.*}} global <32 x i8> zeroinitializer, align 8
// CHECK-1024-NEXT: @global_bool8 ={{.*}} global <16 x i8> zeroinitializer, align 8
// CHECK-1024-NEXT: @global_bool16 ={{.*}} global <8 x i8> zeroinitializer, align 8
// CHECK-1024-NEXT: @global_bool32 ={{.*}} global <4 x i8> zeroinitializer, align 4
// CHECK-1024-NEXT: @global_bool64 ={{.*}} global <2 x i8> zeroinitializer, align 2

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
// CHECK-64-NEXT: @global_arr_i8m2 ={{.*}} global [3 x <16 x i8>] zeroinitializer, align 8
// CHECK-64-NEXT: @global_arr_i16m2 ={{.*}} global [3 x <8 x i16>] zeroinitializer, align 8
// CHECK-64-NEXT: @global_arr_i32m2 ={{.*}} global [3 x <4 x i32>] zeroinitializer, align 8
// CHECK-64-NEXT: @global_arr_i64m2 ={{.*}} global [3 x <2 x i64>] zeroinitializer, align 8
// CHECK-64-NEXT: @global_arr_u8m2 ={{.*}} global [3 x <16 x i8>] zeroinitializer, align 8
// CHECK-64-NEXT: @global_arr_u16m2 ={{.*}} global [3 x <8 x i16>] zeroinitializer, align 8
// CHECK-64-NEXT: @global_arr_u32m2 ={{.*}} global [3 x <4 x i32>] zeroinitializer, align 8
// CHECK-64-NEXT: @global_arr_u64m2 ={{.*}} global [3 x <2 x i64>] zeroinitializer, align 8
// CHECK-64-NEXT: @global_arr_f32m2 ={{.*}} global [3 x <4 x float>] zeroinitializer, align 8
// CHECK-64-NEXT: @global_arr_f64m2 ={{.*}} global [3 x <2 x double>] zeroinitializer, align 8
// CHECK-64-NEXT: @global_arr_i8m4 ={{.*}} global [3 x <32 x i8>] zeroinitializer, align 8
// CHECK-64-NEXT: @global_arr_i16m4 ={{.*}} global [3 x <16 x i16>] zeroinitializer, align 8
// CHECK-64-NEXT: @global_arr_i32m4 ={{.*}} global [3 x <8 x i32>] zeroinitializer, align 8
// CHECK-64-NEXT: @global_arr_i64m4 ={{.*}} global [3 x <4 x i64>] zeroinitializer, align 8
// CHECK-64-NEXT: @global_arr_u8m4 ={{.*}} global [3 x <32 x i8>] zeroinitializer, align 8
// CHECK-64-NEXT: @global_arr_u16m4 ={{.*}} global [3 x <16 x i16>] zeroinitializer, align 8
// CHECK-64-NEXT: @global_arr_u32m4 ={{.*}} global [3 x <8 x i32>] zeroinitializer, align 8
// CHECK-64-NEXT: @global_arr_u64m4 ={{.*}} global [3 x <4 x i64>] zeroinitializer, align 8
// CHECK-64-NEXT: @global_arr_f32m4 ={{.*}} global [3 x <8 x float>] zeroinitializer, align 8
// CHECK-64-NEXT: @global_arr_f64m4 ={{.*}} global [3 x <4 x double>] zeroinitializer, align 8
// CHECK-64-NEXT: @global_arr_i8m8 ={{.*}} global [3 x <64 x i8>] zeroinitializer, align 8
// CHECK-64-NEXT: @global_arr_i16m8 ={{.*}} global [3 x <32 x i16>] zeroinitializer, align 8
// CHECK-64-NEXT: @global_arr_i32m8 ={{.*}} global [3 x <16 x i32>] zeroinitializer, align 8
// CHECK-64-NEXT: @global_arr_i64m8 ={{.*}} global [3 x <8 x i64>] zeroinitializer, align 8
// CHECK-64-NEXT: @global_arr_u8m8 ={{.*}} global [3 x <64 x i8>] zeroinitializer, align 8
// CHECK-64-NEXT: @global_arr_u16m8 ={{.*}} global [3 x <32 x i16>] zeroinitializer, align 8
// CHECK-64-NEXT: @global_arr_u32m8 ={{.*}} global [3 x <16 x i32>] zeroinitializer, align 8
// CHECK-64-NEXT: @global_arr_u64m8 ={{.*}} global [3 x <8 x i64>] zeroinitializer, align 8
// CHECK-64-NEXT: @global_arr_f32m8 ={{.*}} global [3 x <16 x float>] zeroinitializer, align 8
// CHECK-64-NEXT: @global_arr_f64m8 ={{.*}} global [3 x <8 x double>] zeroinitializer, align 8
// CHECK-64-NEXT: @global_arr_bool1 ={{.*}} global [3 x <8 x i8>] zeroinitializer, align 8
// CHECK-64-NEXT: @global_arr_bool2 ={{.*}} global [3 x <4 x i8>] zeroinitializer, align 4
// CHECK-64-NEXT: @global_arr_bool4 ={{.*}} global [3 x <2 x i8>] zeroinitializer, align 2
// CHECK-64-NEXT: @global_arr_bool8 ={{.*}} global [3 x <1 x i8>] zeroinitializer, align 1

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
// CHECK-128-NEXT: @global_arr_i8m2 ={{.*}} global [3 x <32 x i8>] zeroinitializer, align 8
// CHECK-128-NEXT: @global_arr_i16m2 ={{.*}} global [3 x <16 x i16>] zeroinitializer, align 8
// CHECK-128-NEXT: @global_arr_i32m2 ={{.*}} global [3 x <8 x i32>] zeroinitializer, align 8
// CHECK-128-NEXT: @global_arr_i64m2 ={{.*}} global [3 x <4 x i64>] zeroinitializer, align 8
// CHECK-128-NEXT: @global_arr_u8m2 ={{.*}} global [3 x <32 x i8>] zeroinitializer, align 8
// CHECK-128-NEXT: @global_arr_u16m2 ={{.*}} global [3 x <16 x i16>] zeroinitializer, align 8
// CHECK-128-NEXT: @global_arr_u32m2 ={{.*}} global [3 x <8 x i32>] zeroinitializer, align 8
// CHECK-128-NEXT: @global_arr_u64m2 ={{.*}} global [3 x <4 x i64>] zeroinitializer, align 8
// CHECK-128-NEXT: @global_arr_f32m2 ={{.*}} global [3 x <8 x float>] zeroinitializer, align 8
// CHECK-128-NEXT: @global_arr_f64m2 ={{.*}} global [3 x <4 x double>] zeroinitializer, align 8
// CHECK-128-NEXT: @global_arr_i8m4 ={{.*}} global [3 x <64 x i8>] zeroinitializer, align 8
// CHECK-128-NEXT: @global_arr_i16m4 ={{.*}} global [3 x <32 x i16>] zeroinitializer, align 8
// CHECK-128-NEXT: @global_arr_i32m4 ={{.*}} global [3 x <16 x i32>] zeroinitializer, align 8
// CHECK-128-NEXT: @global_arr_i64m4 ={{.*}} global [3 x <8 x i64>] zeroinitializer, align 8
// CHECK-128-NEXT: @global_arr_u8m4 ={{.*}} global [3 x <64 x i8>] zeroinitializer, align 8
// CHECK-128-NEXT: @global_arr_u16m4 ={{.*}} global [3 x <32 x i16>] zeroinitializer, align 8
// CHECK-128-NEXT: @global_arr_u32m4 ={{.*}} global [3 x <16 x i32>] zeroinitializer, align 8
// CHECK-128-NEXT: @global_arr_u64m4 ={{.*}} global [3 x <8 x i64>] zeroinitializer, align 8
// CHECK-128-NEXT: @global_arr_f32m4 ={{.*}} global [3 x <16 x float>] zeroinitializer, align 8
// CHECK-128-NEXT: @global_arr_f64m4 ={{.*}} global [3 x <8 x double>] zeroinitializer, align 8
// CHECK-128-NEXT: @global_arr_i8m8 ={{.*}} global [3 x <128 x i8>] zeroinitializer, align 8
// CHECK-128-NEXT: @global_arr_i16m8 ={{.*}} global [3 x <64 x i16>] zeroinitializer, align 8
// CHECK-128-NEXT: @global_arr_i32m8 ={{.*}} global [3 x <32 x i32>] zeroinitializer, align 8
// CHECK-128-NEXT: @global_arr_i64m8 ={{.*}} global [3 x <16 x i64>] zeroinitializer, align 8
// CHECK-128-NEXT: @global_arr_u8m8 ={{.*}} global [3 x <128 x i8>] zeroinitializer, align 8
// CHECK-128-NEXT: @global_arr_u16m8 ={{.*}} global [3 x <64 x i16>] zeroinitializer, align 8
// CHECK-128-NEXT: @global_arr_u32m8 ={{.*}} global [3 x <32 x i32>] zeroinitializer, align 8
// CHECK-128-NEXT: @global_arr_u64m8 ={{.*}} global [3 x <16 x i64>] zeroinitializer, align 8
// CHECK-128-NEXT: @global_arr_f32m8 ={{.*}} global [3 x <32 x float>] zeroinitializer, align 8
// CHECK-128-NEXT: @global_arr_f64m8 ={{.*}} global [3 x <16 x double>] zeroinitializer, align 8
// CHECK-128-NEXT: @global_arr_bool1 ={{.*}} global [3 x <16 x i8>] zeroinitializer, align 8
// CHECK-128-NEXT: @global_arr_bool2 ={{.*}} global [3 x <8 x i8>] zeroinitializer, align 8
// CHECK-128-NEXT: @global_arr_bool4 ={{.*}} global [3 x <4 x i8>] zeroinitializer, align 4
// CHECK-128-NEXT: @global_arr_bool8 ={{.*}} global [3 x <2 x i8>] zeroinitializer, align 2
// CHECK-128-NEXT: @global_arr_bool16 ={{.*}} global [3 x <1 x i8>] zeroinitializer, align 1

// CHECK-256:      @global_arr_i8 ={{.*}} global [3 x <32 x i8>] zeroinitializer, align 8
// CHECK-256-NEXT: @global_arr_i16 ={{.*}} global [3 x <16 x i16>] zeroinitializer, align 8
// CHECK-256-NEXT: @global_arr_i32 ={{.*}} global [3 x <8 x i32>] zeroinitializer, align 8
// CHECK-256-NEXT: @global_arr_i64 ={{.*}} global [3 x <4 x i64>] zeroinitializer, align 8
// CHECK-256-NEXT: @global_arr_u8 ={{.*}} global [3 x <32 x i8>] zeroinitializer, align 8
// CHECK-256-NEXT: @global_arr_u16 ={{.*}} global [3 x <16 x i16>] zeroinitializer, align 8
// CHECK-256-NEXT: @global_arr_u32 ={{.*}} global [3 x <8 x i32>] zeroinitializer, align 8
// CHECK-256-NEXT: @global_arr_u64 ={{.*}} global [3 x <4 x i64>] zeroinitializer, align 8
// CHECK-256-NEXT: @global_arr_f32 ={{.*}} global [3 x <8 x float>] zeroinitializer, align 8
// CHECK-256-NEXT: @global_arr_f64 ={{.*}} global [3 x <4 x double>] zeroinitializer, align 8
// CHECK-256-NEXT: @global_arr_i8m2 ={{.*}} global [3 x <64 x i8>] zeroinitializer, align 8
// CHECK-256-NEXT: @global_arr_i16m2 ={{.*}} global [3 x <32 x i16>] zeroinitializer, align 8
// CHECK-256-NEXT: @global_arr_i32m2 ={{.*}} global [3 x <16 x i32>] zeroinitializer, align 8
// CHECK-256-NEXT: @global_arr_i64m2 ={{.*}} global [3 x <8 x i64>] zeroinitializer, align 8
// CHECK-256-NEXT: @global_arr_u8m2 ={{.*}} global [3 x <64 x i8>] zeroinitializer, align 8
// CHECK-256-NEXT: @global_arr_u16m2 ={{.*}} global [3 x <32 x i16>] zeroinitializer, align 8
// CHECK-256-NEXT: @global_arr_u32m2 ={{.*}} global [3 x <16 x i32>] zeroinitializer, align 8
// CHECK-256-NEXT: @global_arr_u64m2 ={{.*}} global [3 x <8 x i64>] zeroinitializer, align 8
// CHECK-256-NEXT: @global_arr_f32m2 ={{.*}} global [3 x <16 x float>] zeroinitializer, align 8
// CHECK-256-NEXT: @global_arr_f64m2 ={{.*}} global [3 x <8 x double>] zeroinitializer, align 8
// CHECK-256-NEXT: @global_arr_i8m4 ={{.*}} global [3 x <128 x i8>] zeroinitializer, align 8
// CHECK-256-NEXT: @global_arr_i16m4 ={{.*}} global [3 x <64 x i16>] zeroinitializer, align 8
// CHECK-256-NEXT: @global_arr_i32m4 ={{.*}} global [3 x <32 x i32>] zeroinitializer, align 8
// CHECK-256-NEXT: @global_arr_i64m4 ={{.*}} global [3 x <16 x i64>] zeroinitializer, align 8
// CHECK-256-NEXT: @global_arr_u8m4 ={{.*}} global [3 x <128 x i8>] zeroinitializer, align 8
// CHECK-256-NEXT: @global_arr_u16m4 ={{.*}} global [3 x <64 x i16>] zeroinitializer, align 8
// CHECK-256-NEXT: @global_arr_u32m4 ={{.*}} global [3 x <32 x i32>] zeroinitializer, align 8
// CHECK-256-NEXT: @global_arr_u64m4 ={{.*}} global [3 x <16 x i64>] zeroinitializer, align 8
// CHECK-256-NEXT: @global_arr_f32m4 ={{.*}} global [3 x <32 x float>] zeroinitializer, align 8
// CHECK-256-NEXT: @global_arr_f64m4 ={{.*}} global [3 x <16 x double>] zeroinitializer, align 8
// CHECK-256-NEXT: @global_arr_i8m8 ={{.*}} global [3 x <256 x i8>] zeroinitializer, align 8
// CHECK-256-NEXT: @global_arr_i16m8 ={{.*}} global [3 x <128 x i16>] zeroinitializer, align 8
// CHECK-256-NEXT: @global_arr_i32m8 ={{.*}} global [3 x <64 x i32>] zeroinitializer, align 8
// CHECK-256-NEXT: @global_arr_i64m8 ={{.*}} global [3 x <32 x i64>] zeroinitializer, align 8
// CHECK-256-NEXT: @global_arr_u8m8 ={{.*}} global [3 x <256 x i8>] zeroinitializer, align 8
// CHECK-256-NEXT: @global_arr_u16m8 ={{.*}} global [3 x <128 x i16>] zeroinitializer, align 8
// CHECK-256-NEXT: @global_arr_u32m8 ={{.*}} global [3 x <64 x i32>] zeroinitializer, align 8
// CHECK-256-NEXT: @global_arr_u64m8 ={{.*}} global [3 x <32 x i64>] zeroinitializer, align 8
// CHECK-256-NEXT: @global_arr_f32m8 ={{.*}} global [3 x <64 x float>] zeroinitializer, align 8
// CHECK-256-NEXT: @global_arr_f64m8 ={{.*}} global [3 x <32 x double>] zeroinitializer, align 8
// CHECK-256-NEXT: @global_arr_bool1 ={{.*}} global [3 x <32 x i8>] zeroinitializer, align 8
// CHECK-256-NEXT: @global_arr_bool2 ={{.*}} global [3 x <16 x i8>] zeroinitializer, align 8
// CHECK-256-NEXT: @global_arr_bool4 ={{.*}} global [3 x <8 x i8>] zeroinitializer, align 8
// CHECK-256-NEXT: @global_arr_bool8 ={{.*}} global [3 x <4 x i8>] zeroinitializer, align 4
// CHECK-256-NEXT: @global_arr_bool16 ={{.*}} global [3 x <2 x i8>] zeroinitializer, align 2
// CHECK-256-NEXT: @global_arr_bool32 ={{.*}} global [3 x <1 x i8>] zeroinitializer, align 1

// CHECK-512:      @global_arr_i8 ={{.*}} global [3 x <64 x i8>] zeroinitializer, align 8
// CHECK-512-NEXT: @global_arr_i16 ={{.*}} global [3 x <32 x i16>] zeroinitializer, align 8
// CHECK-512-NEXT: @global_arr_i32 ={{.*}} global [3 x <16 x i32>] zeroinitializer, align 8
// CHECK-512-NEXT: @global_arr_i64 ={{.*}} global [3 x <8 x i64>] zeroinitializer, align 8
// CHECK-512-NEXT: @global_arr_u8 ={{.*}} global [3 x <64 x i8>] zeroinitializer, align 8
// CHECK-512-NEXT: @global_arr_u16 ={{.*}} global [3 x <32 x i16>] zeroinitializer, align 8
// CHECK-512-NEXT: @global_arr_u32 ={{.*}} global [3 x <16 x i32>] zeroinitializer, align 8
// CHECK-512-NEXT: @global_arr_u64 ={{.*}} global [3 x <8 x i64>] zeroinitializer, align 8
// CHECK-512-NEXT: @global_arr_f32 ={{.*}} global [3 x <16 x float>] zeroinitializer, align 8
// CHECK-512-NEXT: @global_arr_f64 ={{.*}} global [3 x <8 x double>] zeroinitializer, align 8
// CHECK-512-NEXT: @global_arr_i8m2 ={{.*}} global [3 x <128 x i8>] zeroinitializer, align 8
// CHECK-512-NEXT: @global_arr_i16m2 ={{.*}} global [3 x <64 x i16>] zeroinitializer, align 8
// CHECK-512-NEXT: @global_arr_i32m2 ={{.*}} global [3 x <32 x i32>] zeroinitializer, align 8
// CHECK-512-NEXT: @global_arr_i64m2 ={{.*}} global [3 x <16 x i64>] zeroinitializer, align 8
// CHECK-512-NEXT: @global_arr_u8m2 ={{.*}} global [3 x <128 x i8>] zeroinitializer, align 8
// CHECK-512-NEXT: @global_arr_u16m2 ={{.*}} global [3 x <64 x i16>] zeroinitializer, align 8
// CHECK-512-NEXT: @global_arr_u32m2 ={{.*}} global [3 x <32 x i32>] zeroinitializer, align 8
// CHECK-512-NEXT: @global_arr_u64m2 ={{.*}} global [3 x <16 x i64>] zeroinitializer, align 8
// CHECK-512-NEXT: @global_arr_f32m2 ={{.*}} global [3 x <32 x float>] zeroinitializer, align 8
// CHECK-512-NEXT: @global_arr_f64m2 ={{.*}} global [3 x <16 x double>] zeroinitializer, align 8
// CHECK-512-NEXT: @global_arr_i8m4 ={{.*}} global [3 x <256 x i8>] zeroinitializer, align 8
// CHECK-512-NEXT: @global_arr_i16m4 ={{.*}} global [3 x <128 x i16>] zeroinitializer, align 8
// CHECK-512-NEXT: @global_arr_i32m4 ={{.*}} global [3 x <64 x i32>] zeroinitializer, align 8
// CHECK-512-NEXT: @global_arr_i64m4 ={{.*}} global [3 x <32 x i64>] zeroinitializer, align 8
// CHECK-512-NEXT: @global_arr_u8m4 ={{.*}} global [3 x <256 x i8>] zeroinitializer, align 8
// CHECK-512-NEXT: @global_arr_u16m4 ={{.*}} global [3 x <128 x i16>] zeroinitializer, align 8
// CHECK-512-NEXT: @global_arr_u32m4 ={{.*}} global [3 x <64 x i32>] zeroinitializer, align 8
// CHECK-512-NEXT: @global_arr_u64m4 ={{.*}} global [3 x <32 x i64>] zeroinitializer, align 8
// CHECK-512-NEXT: @global_arr_f32m4 ={{.*}} global [3 x <64 x float>] zeroinitializer, align 8
// CHECK-512-NEXT: @global_arr_f64m4 ={{.*}} global [3 x <32 x double>] zeroinitializer, align 8
// CHECK-512-NEXT: @global_arr_i8m8 ={{.*}} global [3 x <512 x i8>] zeroinitializer, align 8
// CHECK-512-NEXT: @global_arr_i16m8 ={{.*}} global [3 x <256 x i16>] zeroinitializer, align 8
// CHECK-512-NEXT: @global_arr_i32m8 ={{.*}} global [3 x <128 x i32>] zeroinitializer, align 8
// CHECK-512-NEXT: @global_arr_i64m8 ={{.*}} global [3 x <64 x i64>] zeroinitializer, align 8
// CHECK-512-NEXT: @global_arr_u8m8 ={{.*}} global [3 x <512 x i8>] zeroinitializer, align 8
// CHECK-512-NEXT: @global_arr_u16m8 ={{.*}} global [3 x <256 x i16>] zeroinitializer, align 8
// CHECK-512-NEXT: @global_arr_u32m8 ={{.*}} global [3 x <128 x i32>] zeroinitializer, align 8
// CHECK-512-NEXT: @global_arr_u64m8 ={{.*}} global [3 x <64 x i64>] zeroinitializer, align 8
// CHECK-512-NEXT: @global_arr_f32m8 ={{.*}} global [3 x <128 x float>] zeroinitializer, align 8
// CHECK-512-NEXT: @global_arr_f64m8 ={{.*}} global [3 x <64 x double>] zeroinitializer, align 8
// CHECK-512-NEXT: @global_arr_bool1 ={{.*}} global [3 x <64 x i8>] zeroinitializer, align 8
// CHECK-512-NEXT: @global_arr_bool2 ={{.*}} global [3 x <32 x i8>] zeroinitializer, align 8
// CHECK-512-NEXT: @global_arr_bool4 ={{.*}} global [3 x <16 x i8>] zeroinitializer, align 8
// CHECK-512-NEXT: @global_arr_bool8 ={{.*}} global [3 x <8 x i8>] zeroinitializer, align 8
// CHECK-512-NEXT: @global_arr_bool16 ={{.*}} global [3 x <4 x i8>] zeroinitializer, align 4
// CHECK-512-NEXT: @global_arr_bool32 ={{.*}} global [3 x <2 x i8>] zeroinitializer, align 2
// CHECK-512-NEXT: @global_arr_bool64 ={{.*}} global [3 x <1 x i8>] zeroinitializer, align 1

// CHECK-1024:      @global_arr_i8 ={{.*}} global [3 x <128 x i8>] zeroinitializer, align 8
// CHECK-1024-NEXT: @global_arr_i16 ={{.*}} global [3 x <64 x i16>] zeroinitializer, align 8
// CHECK-1024-NEXT: @global_arr_i32 ={{.*}} global [3 x <32 x i32>] zeroinitializer, align 8
// CHECK-1024-NEXT: @global_arr_i64 ={{.*}} global [3 x <16 x i64>] zeroinitializer, align 8
// CHECK-1024-NEXT: @global_arr_u8 ={{.*}} global [3 x <128 x i8>] zeroinitializer, align 8
// CHECK-1024-NEXT: @global_arr_u16 ={{.*}} global [3 x <64 x i16>] zeroinitializer, align 8
// CHECK-1024-NEXT: @global_arr_u32 ={{.*}} global [3 x <32 x i32>] zeroinitializer, align 8
// CHECK-1024-NEXT: @global_arr_u64 ={{.*}} global [3 x <16 x i64>] zeroinitializer, align 8
// CHECK-1024-NEXT: @global_arr_f32 ={{.*}} global [3 x <32 x float>] zeroinitializer, align 8
// CHECK-1024-NEXT: @global_arr_f64 ={{.*}} global [3 x <16 x double>] zeroinitializer, align 8
// CHECK-1024-NEXT: @global_arr_i8m2 ={{.*}} global [3 x <256 x i8>] zeroinitializer, align 8
// CHECK-1024-NEXT: @global_arr_i16m2 ={{.*}} global [3 x <128 x i16>] zeroinitializer, align 8
// CHECK-1024-NEXT: @global_arr_i32m2 ={{.*}} global [3 x <64 x i32>] zeroinitializer, align 8
// CHECK-1024-NEXT: @global_arr_i64m2 ={{.*}} global [3 x <32 x i64>] zeroinitializer, align 8
// CHECK-1024-NEXT: @global_arr_u8m2 ={{.*}} global [3 x <256 x i8>] zeroinitializer, align 8
// CHECK-1024-NEXT: @global_arr_u16m2 ={{.*}} global [3 x <128 x i16>] zeroinitializer, align 8
// CHECK-1024-NEXT: @global_arr_u32m2 ={{.*}} global [3 x <64 x i32>] zeroinitializer, align 8
// CHECK-1024-NEXT: @global_arr_u64m2 ={{.*}} global [3 x <32 x i64>] zeroinitializer, align 8
// CHECK-1024-NEXT: @global_arr_f32m2 ={{.*}} global [3 x <64 x float>] zeroinitializer, align 8
// CHECK-1024-NEXT: @global_arr_f64m2 ={{.*}} global [3 x <32 x double>] zeroinitializer, align 8
// CHECK-1024-NEXT: @global_arr_i8m4 ={{.*}} global [3 x <512 x i8>] zeroinitializer, align 8
// CHECK-1024-NEXT: @global_arr_i16m4 ={{.*}} global [3 x <256 x i16>] zeroinitializer, align 8
// CHECK-1024-NEXT: @global_arr_i32m4 ={{.*}} global [3 x <128 x i32>] zeroinitializer, align 8
// CHECK-1024-NEXT: @global_arr_i64m4 ={{.*}} global [3 x <64 x i64>] zeroinitializer, align 8
// CHECK-1024-NEXT: @global_arr_u8m4 ={{.*}} global [3 x <512 x i8>] zeroinitializer, align 8
// CHECK-1024-NEXT: @global_arr_u16m4 ={{.*}} global [3 x <256 x i16>] zeroinitializer, align 8
// CHECK-1024-NEXT: @global_arr_u32m4 ={{.*}} global [3 x <128 x i32>] zeroinitializer, align 8
// CHECK-1024-NEXT: @global_arr_u64m4 ={{.*}} global [3 x <64 x i64>] zeroinitializer, align 8
// CHECK-1024-NEXT: @global_arr_f32m4 ={{.*}} global [3 x <128 x float>] zeroinitializer, align 8
// CHECK-1024-NEXT: @global_arr_f64m4 ={{.*}} global [3 x <64 x double>] zeroinitializer, align 8
// CHECK-1024-NEXT: @global_arr_i8m8 ={{.*}} global [3 x <1024 x i8>] zeroinitializer, align 8
// CHECK-1024-NEXT: @global_arr_i16m8 ={{.*}} global [3 x <512 x i16>] zeroinitializer, align 8
// CHECK-1024-NEXT: @global_arr_i32m8 ={{.*}} global [3 x <256 x i32>] zeroinitializer, align 8
// CHECK-1024-NEXT: @global_arr_i64m8 ={{.*}} global [3 x <128 x i64>] zeroinitializer, align 8
// CHECK-1024-NEXT: @global_arr_u8m8 ={{.*}} global [3 x <1024 x i8>] zeroinitializer, align 8
// CHECK-1024-NEXT: @global_arr_u16m8 ={{.*}} global [3 x <512 x i16>] zeroinitializer, align 8
// CHECK-1024-NEXT: @global_arr_u32m8 ={{.*}} global [3 x <256 x i32>] zeroinitializer, align 8
// CHECK-1024-NEXT: @global_arr_u64m8 ={{.*}} global [3 x <128 x i64>] zeroinitializer, align 8
// CHECK-1024-NEXT: @global_arr_f32m8 ={{.*}} global [3 x <256 x float>] zeroinitializer, align 8
// CHECK-1024-NEXT: @global_arr_f64m8 ={{.*}} global [3 x <128 x double>] zeroinitializer, align 8
// CHECK-1024-NEXT: @global_arr_bool1 ={{.*}} global [3 x <128 x i8>] zeroinitializer, align 8
// CHECK-1024-NEXT: @global_arr_bool2 ={{.*}} global [3 x <64 x i8>] zeroinitializer, align 8
// CHECK-1024-NEXT: @global_arr_bool4 ={{.*}} global [3 x <32 x i8>] zeroinitializer, align 8
// CHECK-1024-NEXT: @global_arr_bool8 ={{.*}} global [3 x <16 x i8>] zeroinitializer, align 8
// CHECK-1024-NEXT: @global_arr_bool16 ={{.*}} global [3 x <8 x i8>] zeroinitializer, align 8
// CHECK-1024-NEXT: @global_arr_bool32 ={{.*}} global [3 x <4 x i8>] zeroinitializer, align 4
// CHECK-1024-NEXT: @global_arr_bool64 ={{.*}} global [3 x <2 x i8>] zeroinitializer, align 2

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
// CHECK-64-NEXT: %local_i8m2 = alloca <16 x i8>, align 8
// CHECK-64-NEXT: %local_i16m2 = alloca <8 x i16>, align 8
// CHECK-64-NEXT: %local_i32m2 = alloca <4 x i32>, align 8
// CHECK-64-NEXT: %local_i64m2 = alloca <2 x i64>, align 8
// CHECK-64-NEXT: %local_u8m2 = alloca <16 x i8>, align 8
// CHECK-64-NEXT: %local_u16m2 = alloca <8 x i16>, align 8
// CHECK-64-NEXT: %local_u32m2 = alloca <4 x i32>, align 8
// CHECK-64-NEXT: %local_u64m2 = alloca <2 x i64>, align 8
// CHECK-64-NEXT: %local_f32m2 = alloca <4 x float>, align 8
// CHECK-64-NEXT: %local_f64m2 = alloca <2 x double>, align 8
// CHECK-64-NEXT: %local_i8m4 = alloca <32 x i8>, align 8
// CHECK-64-NEXT: %local_i16m4 = alloca <16 x i16>, align 8
// CHECK-64-NEXT: %local_i32m4 = alloca <8 x i32>, align 8
// CHECK-64-NEXT: %local_i64m4 = alloca <4 x i64>, align 8
// CHECK-64-NEXT: %local_u8m4 = alloca <32 x i8>, align 8
// CHECK-64-NEXT: %local_u16m4 = alloca <16 x i16>, align 8
// CHECK-64-NEXT: %local_u32m4 = alloca <8 x i32>, align 8
// CHECK-64-NEXT: %local_u64m4 = alloca <4 x i64>, align 8
// CHECK-64-NEXT: %local_f32m4 = alloca <8 x float>, align 8
// CHECK-64-NEXT: %local_f64m4 = alloca <4 x double>, align 8
// CHECK-64-NEXT: %local_i8m8 = alloca <64 x i8>, align 8
// CHECK-64-NEXT: %local_i16m8 = alloca <32 x i16>, align 8
// CHECK-64-NEXT: %local_i32m8 = alloca <16 x i32>, align 8
// CHECK-64-NEXT: %local_i64m8 = alloca <8 x i64>, align 8
// CHECK-64-NEXT: %local_u8m8 = alloca <64 x i8>, align 8
// CHECK-64-NEXT: %local_u16m8 = alloca <32 x i16>, align 8
// CHECK-64-NEXT: %local_u32m8 = alloca <16 x i32>, align 8
// CHECK-64-NEXT: %local_u64m8 = alloca <8 x i64>, align 8
// CHECK-64-NEXT: %local_f32m8 = alloca <16 x float>, align 8
// CHECK-64-NEXT: %local_f64m8 = alloca <8 x double>, align 8
// CHECK-64-NEXT: %local_bool1 = alloca <8 x i8>, align 8
// CHECK-64-NEXT: %local_bool2 = alloca <4 x i8>, align 4
// CHECK-64-NEXT: %local_bool4 = alloca <2 x i8>, align 2
// CHECK-64-NEXT: %local_bool8 = alloca <1 x i8>, align 1

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
// CHECK-128-NEXT: %local_i8m2 = alloca <32 x i8>, align 8
// CHECK-128-NEXT: %local_i16m2 = alloca <16 x i16>, align 8
// CHECK-128-NEXT: %local_i32m2 = alloca <8 x i32>, align 8
// CHECK-128-NEXT: %local_i64m2 = alloca <4 x i64>, align 8
// CHECK-128-NEXT: %local_u8m2 = alloca <32 x i8>, align 8
// CHECK-128-NEXT: %local_u16m2 = alloca <16 x i16>, align 8
// CHECK-128-NEXT: %local_u32m2 = alloca <8 x i32>, align 8
// CHECK-128-NEXT: %local_u64m2 = alloca <4 x i64>, align 8
// CHECK-128-NEXT: %local_f32m2 = alloca <8 x float>, align 8
// CHECK-128-NEXT: %local_f64m2 = alloca <4 x double>, align 8
// CHECK-128-NEXT: %local_i8m4 = alloca <64 x i8>, align 8
// CHECK-128-NEXT: %local_i16m4 = alloca <32 x i16>, align 8
// CHECK-128-NEXT: %local_i32m4 = alloca <16 x i32>, align 8
// CHECK-128-NEXT: %local_i64m4 = alloca <8 x i64>, align 8
// CHECK-128-NEXT: %local_u8m4 = alloca <64 x i8>, align 8
// CHECK-128-NEXT: %local_u16m4 = alloca <32 x i16>, align 8
// CHECK-128-NEXT: %local_u32m4 = alloca <16 x i32>, align 8
// CHECK-128-NEXT: %local_u64m4 = alloca <8 x i64>, align 8
// CHECK-128-NEXT: %local_f32m4 = alloca <16 x float>, align 8
// CHECK-128-NEXT: %local_f64m4 = alloca <8 x double>, align 8
// CHECK-128-NEXT: %local_i8m8 = alloca <128 x i8>, align 8
// CHECK-128-NEXT: %local_i16m8 = alloca <64 x i16>, align 8
// CHECK-128-NEXT: %local_i32m8 = alloca <32 x i32>, align 8
// CHECK-128-NEXT: %local_i64m8 = alloca <16 x i64>, align 8
// CHECK-128-NEXT: %local_u8m8 = alloca <128 x i8>, align 8
// CHECK-128-NEXT: %local_u16m8 = alloca <64 x i16>, align 8
// CHECK-128-NEXT: %local_u32m8 = alloca <32 x i32>, align 8
// CHECK-128-NEXT: %local_u64m8 = alloca <16 x i64>, align 8
// CHECK-128-NEXT: %local_f32m8 = alloca <32 x float>, align 8
// CHECK-128-NEXT: %local_f64m8 = alloca <16 x double>, align 8
// CHECK-128-NEXT: %local_bool1 = alloca <16 x i8>, align 8
// CHECK-128-NEXT: %local_bool2 = alloca <8 x i8>, align 8
// CHECK-128-NEXT: %local_bool4 = alloca <4 x i8>, align 4
// CHECK-128-NEXT: %local_bool8 = alloca <2 x i8>, align 2
// CHECK-128-NEXT: %local_bool16 = alloca <1 x i8>, align 1

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
// CHECK-256-NEXT: %local_i8m2 = alloca <64 x i8>, align 8
// CHECK-256-NEXT: %local_i16m2 = alloca <32 x i16>, align 8
// CHECK-256-NEXT: %local_i32m2 = alloca <16 x i32>, align 8
// CHECK-256-NEXT: %local_i64m2 = alloca <8 x i64>, align 8
// CHECK-256-NEXT: %local_u8m2 = alloca <64 x i8>, align 8
// CHECK-256-NEXT: %local_u16m2 = alloca <32 x i16>, align 8
// CHECK-256-NEXT: %local_u32m2 = alloca <16 x i32>, align 8
// CHECK-256-NEXT: %local_u64m2 = alloca <8 x i64>, align 8
// CHECK-256-NEXT: %local_f32m2 = alloca <16 x float>, align 8
// CHECK-256-NEXT: %local_f64m2 = alloca <8 x double>, align 8
// CHECK-256-NEXT: %local_i8m4 = alloca <128 x i8>, align 8
// CHECK-256-NEXT: %local_i16m4 = alloca <64 x i16>, align 8
// CHECK-256-NEXT: %local_i32m4 = alloca <32 x i32>, align 8
// CHECK-256-NEXT: %local_i64m4 = alloca <16 x i64>, align 8
// CHECK-256-NEXT: %local_u8m4 = alloca <128 x i8>, align 8
// CHECK-256-NEXT: %local_u16m4 = alloca <64 x i16>, align 8
// CHECK-256-NEXT: %local_u32m4 = alloca <32 x i32>, align 8
// CHECK-256-NEXT: %local_u64m4 = alloca <16 x i64>, align 8
// CHECK-256-NEXT: %local_f32m4 = alloca <32 x float>, align 8
// CHECK-256-NEXT: %local_f64m4 = alloca <16 x double>, align 8
// CHECK-256-NEXT: %local_i8m8 = alloca <256 x i8>, align 8
// CHECK-256-NEXT: %local_i16m8 = alloca <128 x i16>, align 8
// CHECK-256-NEXT: %local_i32m8 = alloca <64 x i32>, align 8
// CHECK-256-NEXT: %local_i64m8 = alloca <32 x i64>, align 8
// CHECK-256-NEXT: %local_u8m8 = alloca <256 x i8>, align 8
// CHECK-256-NEXT: %local_u16m8 = alloca <128 x i16>, align 8
// CHECK-256-NEXT: %local_u32m8 = alloca <64 x i32>, align 8
// CHECK-256-NEXT: %local_u64m8 = alloca <32 x i64>, align 8
// CHECK-256-NEXT: %local_f32m8 = alloca <64 x float>, align 8
// CHECK-256-NEXT: %local_f64m8 = alloca <32 x double>, align 8
// CHECK-256-NEXT: %local_bool1 = alloca <32 x i8>, align 8
// CHECK-256-NEXT: %local_bool2 = alloca <16 x i8>, align 8
// CHECK-256-NEXT: %local_bool4 = alloca <8 x i8>, align 8
// CHECK-256-NEXT: %local_bool8 = alloca <4 x i8>, align 4
// CHECK-256-NEXT: %local_bool16 = alloca <2 x i8>, align 2
// CHECK-256-NEXT: %local_bool32 = alloca <1 x i8>, align 1

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
// CHECK-512-NEXT: %local_i8m2 = alloca <128 x i8>, align 8
// CHECK-512-NEXT: %local_i16m2 = alloca <64 x i16>, align 8
// CHECK-512-NEXT: %local_i32m2 = alloca <32 x i32>, align 8
// CHECK-512-NEXT: %local_i64m2 = alloca <16 x i64>, align 8
// CHECK-512-NEXT: %local_u8m2 = alloca <128 x i8>, align 8
// CHECK-512-NEXT: %local_u16m2 = alloca <64 x i16>, align 8
// CHECK-512-NEXT: %local_u32m2 = alloca <32 x i32>, align 8
// CHECK-512-NEXT: %local_u64m2 = alloca <16 x i64>, align 8
// CHECK-512-NEXT: %local_f32m2 = alloca <32 x float>, align 8
// CHECK-512-NEXT: %local_f64m2 = alloca <16 x double>, align 8
// CHECK-512-NEXT: %local_i8m4 = alloca <256 x i8>, align 8
// CHECK-512-NEXT: %local_i16m4 = alloca <128 x i16>, align 8
// CHECK-512-NEXT: %local_i32m4 = alloca <64 x i32>, align 8
// CHECK-512-NEXT: %local_i64m4 = alloca <32 x i64>, align 8
// CHECK-512-NEXT: %local_u8m4 = alloca <256 x i8>, align 8
// CHECK-512-NEXT: %local_u16m4 = alloca <128 x i16>, align 8
// CHECK-512-NEXT: %local_u32m4 = alloca <64 x i32>, align 8
// CHECK-512-NEXT: %local_u64m4 = alloca <32 x i64>, align 8
// CHECK-512-NEXT: %local_f32m4 = alloca <64 x float>, align 8
// CHECK-512-NEXT: %local_f64m4 = alloca <32 x double>, align 8
// CHECK-512-NEXT: %local_i8m8 = alloca <512 x i8>, align 8
// CHECK-512-NEXT: %local_i16m8 = alloca <256 x i16>, align 8
// CHECK-512-NEXT: %local_i32m8 = alloca <128 x i32>, align 8
// CHECK-512-NEXT: %local_i64m8 = alloca <64 x i64>, align 8
// CHECK-512-NEXT: %local_u8m8 = alloca <512 x i8>, align 8
// CHECK-512-NEXT: %local_u16m8 = alloca <256 x i16>, align 8
// CHECK-512-NEXT: %local_u32m8 = alloca <128 x i32>, align 8
// CHECK-512-NEXT: %local_u64m8 = alloca <64 x i64>, align 8
// CHECK-512-NEXT: %local_f32m8 = alloca <128 x float>, align 8
// CHECK-512-NEXT: %local_f64m8 = alloca <64 x double>, align 8
// CHECK-512-NEXT: %local_bool1 = alloca <64 x i8>, align 8
// CHECK-512-NEXT: %local_bool2 = alloca <32 x i8>, align 8
// CHECK-512-NEXT: %local_bool4 = alloca <16 x i8>, align 8
// CHECK-512-NEXT: %local_bool8 = alloca <8 x i8>, align 8
// CHECK-512-NEXT: %local_bool16 = alloca <4 x i8>, align 4
// CHECK-512-NEXT: %local_bool32 = alloca <2 x i8>, align 2
// CHECK-512-NEXT: %local_bool64 = alloca <1 x i8>, align 1

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
// CHECK-1024-NEXT:  %local_i8m2 = alloca <256 x i8>, align 8
// CHECK-1024-NEXT:  %local_i16m2 = alloca <128 x i16>, align 8
// CHECK-1024-NEXT:  %local_i32m2 = alloca <64 x i32>, align 8
// CHECK-1024-NEXT:  %local_i64m2 = alloca <32 x i64>, align 8
// CHECK-1024-NEXT:  %local_u8m2 = alloca <256 x i8>, align 8
// CHECK-1024-NEXT:  %local_u16m2 = alloca <128 x i16>, align 8
// CHECK-1024-NEXT:  %local_u32m2 = alloca <64 x i32>, align 8
// CHECK-1024-NEXT:  %local_u64m2 = alloca <32 x i64>, align 8
// CHECK-1024-NEXT:  %local_f32m2 = alloca <64 x float>, align 8
// CHECK-1024-NEXT:  %local_f64m2 = alloca <32 x double>, align 8
// CHECK-1024-NEXT:  %local_i8m4 = alloca <512 x i8>, align 8
// CHECK-1024-NEXT:  %local_i16m4 = alloca <256 x i16>, align 8
// CHECK-1024-NEXT:  %local_i32m4 = alloca <128 x i32>, align 8
// CHECK-1024-NEXT:  %local_i64m4 = alloca <64 x i64>, align 8
// CHECK-1024-NEXT:  %local_u8m4 = alloca <512 x i8>, align 8
// CHECK-1024-NEXT:  %local_u16m4 = alloca <256 x i16>, align 8
// CHECK-1024-NEXT:  %local_u32m4 = alloca <128 x i32>, align 8
// CHECK-1024-NEXT:  %local_u64m4 = alloca <64 x i64>, align 8
// CHECK-1024-NEXT:  %local_f32m4 = alloca <128 x float>, align 8
// CHECK-1024-NEXT:  %local_f64m4 = alloca <64 x double>, align 8
// CHECK-1024-NEXT:  %local_i8m8 = alloca <1024 x i8>, align 8
// CHECK-1024-NEXT:  %local_i16m8 = alloca <512 x i16>, align 8
// CHECK-1024-NEXT:  %local_i32m8 = alloca <256 x i32>, align 8
// CHECK-1024-NEXT:  %local_i64m8 = alloca <128 x i64>, align 8
// CHECK-1024-NEXT:  %local_u8m8 = alloca <1024 x i8>, align 8
// CHECK-1024-NEXT:  %local_u16m8 = alloca <512 x i16>, align 8
// CHECK-1024-NEXT:  %local_u32m8 = alloca <256 x i32>, align 8
// CHECK-1024-NEXT:  %local_u64m8 = alloca <128 x i64>, align 8
// CHECK-1024-NEXT:  %local_f32m8 = alloca <256 x float>, align 8
// CHECK-1024-NEXT:  %local_f64m8 = alloca <128 x double>, align 8
// CHECK-1024-NEXT: %local_bool1 = alloca <128 x i8>, align 8
// CHECK-1024-NEXT: %local_bool2 = alloca <64 x i8>, align 8
// CHECK-1024-NEXT: %local_bool4 = alloca <32 x i8>, align 8
// CHECK-1024-NEXT: %local_bool8 = alloca <16 x i8>, align 8
// CHECK-1024-NEXT: %local_bool16 = alloca <8 x i8>, align 8
// CHECK-1024-NEXT: %local_bool32 = alloca <4 x i8>, align 4
// CHECK-1024-NEXT: %local_bool64 = alloca <2 x i8>, align 2

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
// CHECK-64-NEXT: %local_arr_i8m2 = alloca [3 x <16 x i8>], align 8
// CHECK-64-NEXT: %local_arr_i16m2 = alloca [3 x <8 x i16>], align 8
// CHECK-64-NEXT: %local_arr_i32m2 = alloca [3 x <4 x i32>], align 8
// CHECK-64-NEXT: %local_arr_i64m2 = alloca [3 x <2 x i64>], align 8
// CHECK-64-NEXT: %local_arr_u8m2 = alloca [3 x <16 x i8>], align 8
// CHECK-64-NEXT: %local_arr_u16m2 = alloca [3 x <8 x i16>], align 8
// CHECK-64-NEXT: %local_arr_u32m2 = alloca [3 x <4 x i32>], align 8
// CHECK-64-NEXT: %local_arr_u64m2 = alloca [3 x <2 x i64>], align 8
// CHECK-64-NEXT: %local_arr_f32m2 = alloca [3 x <4 x float>], align 8
// CHECK-64-NEXT: %local_arr_f64m2 = alloca [3 x <2 x double>], align 8
// CHECK-64-NEXT: %local_arr_i8m4 = alloca [3 x <32 x i8>], align 8
// CHECK-64-NEXT: %local_arr_i16m4 = alloca [3 x <16 x i16>], align 8
// CHECK-64-NEXT: %local_arr_i32m4 = alloca [3 x <8 x i32>], align 8
// CHECK-64-NEXT: %local_arr_i64m4 = alloca [3 x <4 x i64>], align 8
// CHECK-64-NEXT: %local_arr_u8m4 = alloca [3 x <32 x i8>], align 8
// CHECK-64-NEXT: %local_arr_u16m4 = alloca [3 x <16 x i16>], align 8
// CHECK-64-NEXT: %local_arr_u32m4 = alloca [3 x <8 x i32>], align 8
// CHECK-64-NEXT: %local_arr_u64m4 = alloca [3 x <4 x i64>], align 8
// CHECK-64-NEXT: %local_arr_f32m4 = alloca [3 x <8 x float>], align 8
// CHECK-64-NEXT: %local_arr_f64m4 = alloca [3 x <4 x double>], align 8
// CHECK-64-NEXT: %local_arr_i8m8 = alloca [3 x <64 x i8>], align 8
// CHECK-64-NEXT: %local_arr_i16m8 = alloca [3 x <32 x i16>], align 8
// CHECK-64-NEXT: %local_arr_i32m8 = alloca [3 x <16 x i32>], align 8
// CHECK-64-NEXT: %local_arr_i64m8 = alloca [3 x <8 x i64>], align 8
// CHECK-64-NEXT: %local_arr_u8m8 = alloca [3 x <64 x i8>], align 8
// CHECK-64-NEXT: %local_arr_u16m8 = alloca [3 x <32 x i16>], align 8
// CHECK-64-NEXT: %local_arr_u32m8 = alloca [3 x <16 x i32>], align 8
// CHECK-64-NEXT: %local_arr_u64m8 = alloca [3 x <8 x i64>], align 8
// CHECK-64-NEXT: %local_arr_f32m8 = alloca [3 x <16 x float>], align 8
// CHECK-64-NEXT: %local_arr_f64m8 = alloca [3 x <8 x double>], align 8
// CHECK-64-NEXT: %local_arr_i8mf2 = alloca [3 x <4 x i8>], align 4
// CHECK-64-NEXT: %local_arr_i16mf2 = alloca [3 x <2 x i16>], align 4
// CHECK-64-NEXT: %local_arr_i32mf2 = alloca [3 x <1 x i32>], align 4
// CHECK-64-NEXT: %local_arr_u8mf2 = alloca [3 x <4 x i8>], align 4
// CHECK-64-NEXT: %local_arr_u16mf2 = alloca [3 x <2 x i16>], align 4
// CHECK-64-NEXT: %local_arr_u32mf2 = alloca [3 x <1 x i32>], align 4
// CHECK-64-NEXT: %local_arr_f32mf2 = alloca [3 x <1 x float>], align 4
// CHECK-64-NEXT: %local_arr_i8mf4 = alloca [3 x <2 x i8>], align 2
// CHECK-64-NEXT: %local_arr_i16mf4 = alloca [3 x <1 x i16>], align 2
// CHECK-64-NEXT: %local_arr_u8mf4 = alloca [3 x <2 x i8>], align 2
// CHECK-64-NEXT: %local_arr_u16mf4 = alloca [3 x <1 x i16>], align 2
// CHECK-64-NEXT: %local_arr_i8mf8 = alloca [3 x <1 x i8>], align 1
// CHECK-64-NEXT: %local_arr_u8mf8 = alloca [3 x <1 x i8>], align 1
// CHECK-64-NEXT: %local_arr_bool1 = alloca [3 x <8 x i8>], align 8
// CHECK-64-NEXT: %local_arr_bool2 = alloca [3 x <4 x i8>], align 4
// CHECK-64-NEXT: %local_arr_bool4 = alloca [3 x <2 x i8>], align 2
// CHECK-64-NEXT: %local_arr_bool8 = alloca [3 x <1 x i8>], align 1

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
// CHECK-128-NEXT: %local_arr_i8m2 = alloca [3 x <32 x i8>], align 8
// CHECK-128-NEXT: %local_arr_i16m2 = alloca [3 x <16 x i16>], align 8
// CHECK-128-NEXT: %local_arr_i32m2 = alloca [3 x <8 x i32>], align 8
// CHECK-128-NEXT: %local_arr_i64m2 = alloca [3 x <4 x i64>], align 8
// CHECK-128-NEXT: %local_arr_u8m2 = alloca [3 x <32 x i8>], align 8
// CHECK-128-NEXT: %local_arr_u16m2 = alloca [3 x <16 x i16>], align 8
// CHECK-128-NEXT: %local_arr_u32m2 = alloca [3 x <8 x i32>], align 8
// CHECK-128-NEXT: %local_arr_u64m2 = alloca [3 x <4 x i64>], align 8
// CHECK-128-NEXT: %local_arr_f32m2 = alloca [3 x <8 x float>], align 8
// CHECK-128-NEXT: %local_arr_f64m2 = alloca [3 x <4 x double>], align 8
// CHECK-128-NEXT: %local_arr_i8m4 = alloca [3 x <64 x i8>], align 8
// CHECK-128-NEXT: %local_arr_i16m4 = alloca [3 x <32 x i16>], align 8
// CHECK-128-NEXT: %local_arr_i32m4 = alloca [3 x <16 x i32>], align 8
// CHECK-128-NEXT: %local_arr_i64m4 = alloca [3 x <8 x i64>], align 8
// CHECK-128-NEXT: %local_arr_u8m4 = alloca [3 x <64 x i8>], align 8
// CHECK-128-NEXT: %local_arr_u16m4 = alloca [3 x <32 x i16>], align 8
// CHECK-128-NEXT: %local_arr_u32m4 = alloca [3 x <16 x i32>], align 8
// CHECK-128-NEXT: %local_arr_u64m4 = alloca [3 x <8 x i64>], align 8
// CHECK-128-NEXT: %local_arr_f32m4 = alloca [3 x <16 x float>], align 8
// CHECK-128-NEXT: %local_arr_f64m4 = alloca [3 x <8 x double>], align 8
// CHECK-128-NEXT: %local_arr_i8m8 = alloca [3 x <128 x i8>], align 8
// CHECK-128-NEXT: %local_arr_i16m8 = alloca [3 x <64 x i16>], align 8
// CHECK-128-NEXT: %local_arr_i32m8 = alloca [3 x <32 x i32>], align 8
// CHECK-128-NEXT: %local_arr_i64m8 = alloca [3 x <16 x i64>], align 8
// CHECK-128-NEXT: %local_arr_u8m8 = alloca [3 x <128 x i8>], align 8
// CHECK-128-NEXT: %local_arr_u16m8 = alloca [3 x <64 x i16>], align 8
// CHECK-128-NEXT: %local_arr_u32m8 = alloca [3 x <32 x i32>], align 8
// CHECK-128-NEXT: %local_arr_u64m8 = alloca [3 x <16 x i64>], align 8
// CHECK-128-NEXT: %local_arr_f32m8 = alloca [3 x <32 x float>], align 8
// CHECK-128-NEXT: %local_arr_f64m8 = alloca [3 x <16 x double>], align 8
// CHECK-128-NEXT: %local_arr_i8mf2 = alloca [3 x <8 x i8>], align 8
// CHECK-128-NEXT: %local_arr_i16mf2 = alloca [3 x <4 x i16>], align 8
// CHECK-128-NEXT: %local_arr_i32mf2 = alloca [3 x <2 x i32>], align 8
// CHECK-128-NEXT: %local_arr_u8mf2 = alloca [3 x <8 x i8>], align 8
// CHECK-128-NEXT: %local_arr_u16mf2 = alloca [3 x <4 x i16>], align 8
// CHECK-128-NEXT: %local_arr_u32mf2 = alloca [3 x <2 x i32>], align 8
// CHECK-128-NEXT: %local_arr_f32mf2 = alloca [3 x <2 x float>], align 8
// CHECK-128-NEXT: %local_arr_i8mf4 = alloca [3 x <4 x i8>], align 4
// CHECK-128-NEXT: %local_arr_i16mf4 = alloca [3 x <2 x i16>], align 4
// CHECK-128-NEXT: %local_arr_u8mf4 = alloca [3 x <4 x i8>], align 4
// CHECK-128-NEXT: %local_arr_u16mf4 = alloca [3 x <2 x i16>], align 4
// CHECK-128-NEXT: %local_arr_i8mf8 = alloca [3 x <2 x i8>], align 2
// CHECK-128-NEXT: %local_arr_u8mf8 = alloca [3 x <2 x i8>], align 2
// CHECK-128-NEXT: %local_arr_bool1 = alloca [3 x <16 x i8>], align 8
// CHECK-128-NEXT: %local_arr_bool2 = alloca [3 x <8 x i8>], align 8
// CHECK-128-NEXT: %local_arr_bool4 = alloca [3 x <4 x i8>], align 4
// CHECK-128-NEXT: %local_arr_bool8 = alloca [3 x <2 x i8>], align 2
// CHECK-128-NEXT: %local_arr_bool16 = alloca [3 x <1 x i8>], align 1

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
// CHECK-256-NEXT: %local_arr_i8m2 = alloca [3 x <64 x i8>], align 8
// CHECK-256-NEXT: %local_arr_i16m2 = alloca [3 x <32 x i16>], align 8
// CHECK-256-NEXT: %local_arr_i32m2 = alloca [3 x <16 x i32>], align 8
// CHECK-256-NEXT: %local_arr_i64m2 = alloca [3 x <8 x i64>], align 8
// CHECK-256-NEXT: %local_arr_u8m2 = alloca [3 x <64 x i8>], align 8
// CHECK-256-NEXT: %local_arr_u16m2 = alloca [3 x <32 x i16>], align 8
// CHECK-256-NEXT: %local_arr_u32m2 = alloca [3 x <16 x i32>], align 8
// CHECK-256-NEXT: %local_arr_u64m2 = alloca [3 x <8 x i64>], align 8
// CHECK-256-NEXT: %local_arr_f32m2 = alloca [3 x <16 x float>], align 8
// CHECK-256-NEXT: %local_arr_f64m2 = alloca [3 x <8 x double>], align 8
// CHECK-256-NEXT: %local_arr_i8m4 = alloca [3 x <128 x i8>], align 8
// CHECK-256-NEXT: %local_arr_i16m4 = alloca [3 x <64 x i16>], align 8
// CHECK-256-NEXT: %local_arr_i32m4 = alloca [3 x <32 x i32>], align 8
// CHECK-256-NEXT: %local_arr_i64m4 = alloca [3 x <16 x i64>], align 8
// CHECK-256-NEXT: %local_arr_u8m4 = alloca [3 x <128 x i8>], align 8
// CHECK-256-NEXT: %local_arr_u16m4 = alloca [3 x <64 x i16>], align 8
// CHECK-256-NEXT: %local_arr_u32m4 = alloca [3 x <32 x i32>], align 8
// CHECK-256-NEXT: %local_arr_u64m4 = alloca [3 x <16 x i64>], align 8
// CHECK-256-NEXT: %local_arr_f32m4 = alloca [3 x <32 x float>], align 8
// CHECK-256-NEXT: %local_arr_f64m4 = alloca [3 x <16 x double>], align 8
// CHECK-256-NEXT: %local_arr_i8m8 = alloca [3 x <256 x i8>], align 8
// CHECK-256-NEXT: %local_arr_i16m8 = alloca [3 x <128 x i16>], align 8
// CHECK-256-NEXT: %local_arr_i32m8 = alloca [3 x <64 x i32>], align 8
// CHECK-256-NEXT: %local_arr_i64m8 = alloca [3 x <32 x i64>], align 8
// CHECK-256-NEXT: %local_arr_u8m8 = alloca [3 x <256 x i8>], align 8
// CHECK-256-NEXT: %local_arr_u16m8 = alloca [3 x <128 x i16>], align 8
// CHECK-256-NEXT: %local_arr_u32m8 = alloca [3 x <64 x i32>], align 8
// CHECK-256-NEXT: %local_arr_u64m8 = alloca [3 x <32 x i64>], align 8
// CHECK-256-NEXT: %local_arr_f32m8 = alloca [3 x <64 x float>], align 8
// CHECK-256-NEXT: %local_arr_f64m8 = alloca [3 x <32 x double>], align 8
// CHECK-256-NEXT: %local_arr_i8mf2 = alloca [3 x <16 x i8>], align 8
// CHECK-256-NEXT: %local_arr_i16mf2 = alloca [3 x <8 x i16>], align 8
// CHECK-256-NEXT: %local_arr_i32mf2 = alloca [3 x <4 x i32>], align 8
// CHECK-256-NEXT: %local_arr_u8mf2 = alloca [3 x <16 x i8>], align 8
// CHECK-256-NEXT: %local_arr_u16mf2 = alloca [3 x <8 x i16>], align 8
// CHECK-256-NEXT: %local_arr_u32mf2 = alloca [3 x <4 x i32>], align 8
// CHECK-256-NEXT: %local_arr_f32mf2 = alloca [3 x <4 x float>], align 8
// CHECK-256-NEXT: %local_arr_i8mf4 = alloca [3 x <8 x i8>], align 8
// CHECK-256-NEXT: %local_arr_i16mf4 = alloca [3 x <4 x i16>], align 8
// CHECK-256-NEXT: %local_arr_u8mf4 = alloca [3 x <8 x i8>], align 8
// CHECK-256-NEXT: %local_arr_u16mf4 = alloca [3 x <4 x i16>], align 8
// CHECK-256-NEXT: %local_arr_i8mf8 = alloca [3 x <4 x i8>], align 4
// CHECK-256-NEXT: %local_arr_u8mf8 = alloca [3 x <4 x i8>], align 4
// CHECK-256-NEXT: %local_arr_bool1 = alloca [3 x <32 x i8>], align 8
// CHECK-256-NEXT: %local_arr_bool2 = alloca [3 x <16 x i8>], align 8
// CHECK-256-NEXT: %local_arr_bool4 = alloca [3 x <8 x i8>], align 8
// CHECK-256-NEXT: %local_arr_bool8 = alloca [3 x <4 x i8>], align 4
// CHECK-256-NEXT: %local_arr_bool16 = alloca [3 x <2 x i8>], align 2
// CHECK-256-NEXT: %local_arr_bool32 = alloca [3 x <1 x i8>], align 1

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
// CHECK-512-NEXT: %local_arr_i8m2 = alloca [3 x <128 x i8>], align 8
// CHECK-512-NEXT: %local_arr_i16m2 = alloca [3 x <64 x i16>], align 8
// CHECK-512-NEXT: %local_arr_i32m2 = alloca [3 x <32 x i32>], align 8
// CHECK-512-NEXT: %local_arr_i64m2 = alloca [3 x <16 x i64>], align 8
// CHECK-512-NEXT: %local_arr_u8m2 = alloca [3 x <128 x i8>], align 8
// CHECK-512-NEXT: %local_arr_u16m2 = alloca [3 x <64 x i16>], align 8
// CHECK-512-NEXT: %local_arr_u32m2 = alloca [3 x <32 x i32>], align 8
// CHECK-512-NEXT: %local_arr_u64m2 = alloca [3 x <16 x i64>], align 8
// CHECK-512-NEXT: %local_arr_f32m2 = alloca [3 x <32 x float>], align 8
// CHECK-512-NEXT: %local_arr_f64m2 = alloca [3 x <16 x double>], align 8
// CHECK-512-NEXT: %local_arr_i8m4 = alloca [3 x <256 x i8>], align 8
// CHECK-512-NEXT: %local_arr_i16m4 = alloca [3 x <128 x i16>], align 8
// CHECK-512-NEXT: %local_arr_i32m4 = alloca [3 x <64 x i32>], align 8
// CHECK-512-NEXT: %local_arr_i64m4 = alloca [3 x <32 x i64>], align 8
// CHECK-512-NEXT: %local_arr_u8m4 = alloca [3 x <256 x i8>], align 8
// CHECK-512-NEXT: %local_arr_u16m4 = alloca [3 x <128 x i16>], align 8
// CHECK-512-NEXT: %local_arr_u32m4 = alloca [3 x <64 x i32>], align 8
// CHECK-512-NEXT: %local_arr_u64m4 = alloca [3 x <32 x i64>], align 8
// CHECK-512-NEXT: %local_arr_f32m4 = alloca [3 x <64 x float>], align 8
// CHECK-512-NEXT: %local_arr_f64m4 = alloca [3 x <32 x double>], align 8
// CHECK-512-NEXT: %local_arr_i8m8 = alloca [3 x <512 x i8>], align 8
// CHECK-512-NEXT: %local_arr_i16m8 = alloca [3 x <256 x i16>], align 8
// CHECK-512-NEXT: %local_arr_i32m8 = alloca [3 x <128 x i32>], align 8
// CHECK-512-NEXT: %local_arr_i64m8 = alloca [3 x <64 x i64>], align 8
// CHECK-512-NEXT: %local_arr_u8m8 = alloca [3 x <512 x i8>], align 8
// CHECK-512-NEXT: %local_arr_u16m8 = alloca [3 x <256 x i16>], align 8
// CHECK-512-NEXT: %local_arr_u32m8 = alloca [3 x <128 x i32>], align 8
// CHECK-512-NEXT: %local_arr_u64m8 = alloca [3 x <64 x i64>], align 8
// CHECK-512-NEXT: %local_arr_f32m8 = alloca [3 x <128 x float>], align 8
// CHECK-512-NEXT: %local_arr_f64m8 = alloca [3 x <64 x double>], align 8
// CHECK-512-NEXT: %local_arr_i8mf2 = alloca [3 x <32 x i8>], align 8
// CHECK-512-NEXT: %local_arr_i16mf2 = alloca [3 x <16 x i16>], align 8
// CHECK-512-NEXT: %local_arr_i32mf2 = alloca [3 x <8 x i32>], align 8
// CHECK-512-NEXT: %local_arr_u8mf2 = alloca [3 x <32 x i8>], align 8
// CHECK-512-NEXT: %local_arr_u16mf2 = alloca [3 x <16 x i16>], align 8
// CHECK-512-NEXT: %local_arr_u32mf2 = alloca [3 x <8 x i32>], align 8
// CHECK-512-NEXT: %local_arr_f32mf2 = alloca [3 x <8 x float>], align 8
// CHECK-512-NEXT: %local_arr_i8mf4 = alloca [3 x <16 x i8>], align 8
// CHECK-512-NEXT: %local_arr_i16mf4 = alloca [3 x <8 x i16>], align 8
// CHECK-512-NEXT: %local_arr_u8mf4 = alloca [3 x <16 x i8>], align 8
// CHECK-512-NEXT: %local_arr_u16mf4 = alloca [3 x <8 x i16>], align 8
// CHECK-512-NEXT: %local_arr_i8mf8 = alloca [3 x <8 x i8>], align 8
// CHECK-512-NEXT: %local_arr_u8mf8 = alloca [3 x <8 x i8>], align 8
// CHECK-512-NEXT: %local_arr_bool1 = alloca [3 x <64 x i8>], align 8
// CHECK-512-NEXT: %local_arr_bool2 = alloca [3 x <32 x i8>], align 8
// CHECK-512-NEXT: %local_arr_bool4 = alloca [3 x <16 x i8>], align 8
// CHECK-512-NEXT: %local_arr_bool8 = alloca [3 x <8 x i8>], align 8
// CHECK-512-NEXT: %local_arr_bool16 = alloca [3 x <4 x i8>], align 4
// CHECK-512-NEXT: %local_arr_bool32 = alloca [3 x <2 x i8>], align 2
// CHECK-512-NEXT: %local_arr_bool64 = alloca [3 x <1 x i8>], align 1

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
// CHECK-1024-NEXT:  %local_arr_i8m2 = alloca [3 x <256 x i8>], align 8
// CHECK-1024-NEXT:  %local_arr_i16m2 = alloca [3 x <128 x i16>], align 8
// CHECK-1024-NEXT:  %local_arr_i32m2 = alloca [3 x <64 x i32>], align 8
// CHECK-1024-NEXT:  %local_arr_i64m2 = alloca [3 x <32 x i64>], align 8
// CHECK-1024-NEXT:  %local_arr_u8m2 = alloca [3 x <256 x i8>], align 8
// CHECK-1024-NEXT:  %local_arr_u16m2 = alloca [3 x <128 x i16>], align 8
// CHECK-1024-NEXT:  %local_arr_u32m2 = alloca [3 x <64 x i32>], align 8
// CHECK-1024-NEXT:  %local_arr_u64m2 = alloca [3 x <32 x i64>], align 8
// CHECK-1024-NEXT:  %local_arr_f32m2 = alloca [3 x <64 x float>], align 8
// CHECK-1024-NEXT:  %local_arr_f64m2 = alloca [3 x <32 x double>], align 8
// CHECK-1024-NEXT:  %local_arr_i8m4 = alloca [3 x <512 x i8>], align 8
// CHECK-1024-NEXT:  %local_arr_i16m4 = alloca [3 x <256 x i16>], align 8
// CHECK-1024-NEXT:  %local_arr_i32m4 = alloca [3 x <128 x i32>], align 8
// CHECK-1024-NEXT:  %local_arr_i64m4 = alloca [3 x <64 x i64>], align 8
// CHECK-1024-NEXT:  %local_arr_u8m4 = alloca [3 x <512 x i8>], align 8
// CHECK-1024-NEXT:  %local_arr_u16m4 = alloca [3 x <256 x i16>], align 8
// CHECK-1024-NEXT:  %local_arr_u32m4 = alloca [3 x <128 x i32>], align 8
// CHECK-1024-NEXT:  %local_arr_u64m4 = alloca [3 x <64 x i64>], align 8
// CHECK-1024-NEXT:  %local_arr_f32m4 = alloca [3 x <128 x float>], align 8
// CHECK-1024-NEXT:  %local_arr_f64m4 = alloca [3 x <64 x double>], align 8
// CHECK-1024-NEXT:  %local_arr_i8m8 = alloca [3 x <1024 x i8>], align 8
// CHECK-1024-NEXT:  %local_arr_i16m8 = alloca [3 x <512 x i16>], align 8
// CHECK-1024-NEXT:  %local_arr_i32m8 = alloca [3 x <256 x i32>], align 8
// CHECK-1024-NEXT:  %local_arr_i64m8 = alloca [3 x <128 x i64>], align 8
// CHECK-1024-NEXT:  %local_arr_u8m8 = alloca [3 x <1024 x i8>], align 8
// CHECK-1024-NEXT:  %local_arr_u16m8 = alloca [3 x <512 x i16>], align 8
// CHECK-1024-NEXT:  %local_arr_u32m8 = alloca [3 x <256 x i32>], align 8
// CHECK-1024-NEXT:  %local_arr_u64m8 = alloca [3 x <128 x i64>], align 8
// CHECK-1024-NEXT:  %local_arr_f32m8 = alloca [3 x <256 x float>], align 8
// CHECK-1024-NEXT:  %local_arr_f64m8 = alloca [3 x <128 x double>], align 8
// CHECK-1024-NEXT: %local_arr_i8mf2 = alloca [3 x <64 x i8>], align 8
// CHECK-1024-NEXT: %local_arr_i16mf2 = alloca [3 x <32 x i16>], align 8
// CHECK-1024-NEXT: %local_arr_i32mf2 = alloca [3 x <16 x i32>], align 8
// CHECK-1024-NEXT: %local_arr_u8mf2 = alloca [3 x <64 x i8>], align 8
// CHECK-1024-NEXT: %local_arr_u16mf2 = alloca [3 x <32 x i16>], align 8
// CHECK-1024-NEXT: %local_arr_u32mf2 = alloca [3 x <16 x i32>], align 8
// CHECK-1024-NEXT: %local_arr_f32mf2 = alloca [3 x <16 x float>], align 8
// CHECK-1024-NEXT: %local_arr_i8mf4 = alloca [3 x <32 x i8>], align 8
// CHECK-1024-NEXT: %local_arr_i16mf4 = alloca [3 x <16 x i16>], align 8
// CHECK-1024-NEXT: %local_arr_u8mf4 = alloca [3 x <32 x i8>], align 8
// CHECK-1024-NEXT: %local_arr_u16mf4 = alloca [3 x <16 x i16>], align 8
// CHECK-1024-NEXT: %local_arr_i8mf8 = alloca [3 x <16 x i8>], align 8
// CHECK-1024-NEXT: %local_arr_u8mf8 = alloca [3 x <16 x i8>], align 8
// CHECK-1024-NEXT: %local_arr_bool1 = alloca [3 x <128 x i8>], align 8
// CHECK-1024-NEXT: %local_arr_bool2 = alloca [3 x <64 x i8>], align 8
// CHECK-1024-NEXT: %local_arr_bool4 = alloca [3 x <32 x i8>], align 8
// CHECK-1024-NEXT: %local_arr_bool8 = alloca [3 x <16 x i8>], align 8
// CHECK-1024-NEXT: %local_arr_bool16 = alloca [3 x <8 x i8>], align 8
// CHECK-1024-NEXT: %local_arr_bool32 = alloca [3 x <4 x i8>], align 4
// CHECK-1024-NEXT: %local_arr_bool64 = alloca [3 x <2 x i8>], align 2

// RUN: %clang_cc1 -triple riscv32-none-linux-gnu %s -emit-llvm -o - \
// RUN:   -target-feature +zve64d -target-feature +zvfhmin \
// RUN:   -target-feature +zvfbfmin -target-feature +experimental-zvfofp8min \
// RUN:   | FileCheck %s
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu %s -emit-llvm -o - \
// RUN:   -target-feature +zve64d -target-feature +zvfhmin \
// RUN:   -target-feature +zvfbfmin -target-feature +experimental-zvfofp8min \
// RUN:   | FileCheck %s

typedef __rvv_int8mf8_t  vint8mf8_t;
typedef __rvv_int8mf4_t  vint8mf4_t;
typedef __rvv_int8mf2_t  vint8mf2_t;
typedef __rvv_int8m1_t   vint8m1_t;
typedef __rvv_int8m2_t   vint8m2_t;
typedef __rvv_int8m4_t   vint8m4_t;
typedef __rvv_int8m8_t   vint8m8_t;

typedef __rvv_uint8mf8_t vuint8mf8_t;
typedef __rvv_uint8mf4_t vuint8mf4_t;
typedef __rvv_uint8mf2_t vuint8mf2_t;
typedef __rvv_uint8m1_t  vuint8m1_t;
typedef __rvv_uint8m2_t  vuint8m2_t;
typedef __rvv_uint8m4_t  vuint8m4_t;
typedef __rvv_uint8m8_t  vuint8m8_t;

typedef __rvv_int16mf4_t vint16mf4_t;
typedef __rvv_int16mf2_t vint16mf2_t;
typedef __rvv_int16m1_t  vint16m1_t;
typedef __rvv_int16m2_t  vint16m2_t;
typedef __rvv_int16m4_t  vint16m4_t;
typedef __rvv_int16m8_t  vint16m8_t;

typedef __rvv_uint16mf4_t vuint16mf4_t;
typedef __rvv_uint16mf2_t vuint16mf2_t;
typedef __rvv_uint16m1_t  vuint16m1_t;
typedef __rvv_uint16m2_t  vuint16m2_t;
typedef __rvv_uint16m4_t  vuint16m4_t;
typedef __rvv_uint16m8_t  vuint16m8_t;

typedef __rvv_int32mf2_t vint32mf2_t;
typedef __rvv_int32m1_t  vint32m1_t;
typedef __rvv_int32m2_t  vint32m2_t;
typedef __rvv_int32m4_t  vint32m4_t;
typedef __rvv_int32m8_t  vint32m8_t;

typedef __rvv_uint32mf2_t vuint32mf2_t;
typedef __rvv_uint32m1_t  vuint32m1_t;
typedef __rvv_uint32m2_t  vuint32m2_t;
typedef __rvv_uint32m4_t  vuint32m4_t;
typedef __rvv_uint32m8_t  vuint32m8_t;

typedef __rvv_int64m1_t vint64m1_t;
typedef __rvv_int64m2_t vint64m2_t;
typedef __rvv_int64m4_t vint64m4_t;
typedef __rvv_int64m8_t vint64m8_t;

typedef __rvv_uint64m1_t vuint64m1_t;
typedef __rvv_uint64m2_t vuint64m2_t;
typedef __rvv_uint64m4_t vuint64m4_t;
typedef __rvv_uint64m8_t vuint64m8_t;

typedef __rvv_float8e4m3mf8_t vfloat8e4m3mf8_t;
typedef __rvv_float8e4m3mf4_t vfloat8e4m3mf4_t;
typedef __rvv_float8e4m3mf2_t vfloat8e4m3mf2_t;
typedef __rvv_float8e4m3m1_t  vfloat8e4m3m1_t;
typedef __rvv_float8e4m3m2_t  vfloat8e4m3m2_t;
typedef __rvv_float8e4m3m4_t  vfloat8e4m3m4_t;
typedef __rvv_float8e4m3m8_t  vfloat8e4m3m8_t;

typedef __rvv_float8e5m2mf8_t vfloat8e5m2mf8_t;
typedef __rvv_float8e5m2mf4_t vfloat8e5m2mf4_t;
typedef __rvv_float8e5m2mf2_t vfloat8e5m2mf2_t;
typedef __rvv_float8e5m2m1_t  vfloat8e5m2m1_t;
typedef __rvv_float8e5m2m2_t  vfloat8e5m2m2_t;
typedef __rvv_float8e5m2m4_t  vfloat8e5m2m4_t;
typedef __rvv_float8e5m2m8_t  vfloat8e5m2m8_t;

typedef __rvv_float16mf4_t vfloat16mf4_t;
typedef __rvv_float16mf2_t vfloat16mf2_t;
typedef __rvv_float16m1_t  vfloat16m1_t;
typedef __rvv_float16m2_t  vfloat16m2_t;
typedef __rvv_float16m4_t  vfloat16m4_t;
typedef __rvv_float16m8_t  vfloat16m8_t;

typedef __rvv_bfloat16mf4_t vbfloat16mf4_t;
typedef __rvv_bfloat16mf2_t vbfloat16mf2_t;
typedef __rvv_bfloat16m1_t  vbfloat16m1_t;
typedef __rvv_bfloat16m2_t  vbfloat16m2_t;
typedef __rvv_bfloat16m4_t  vbfloat16m4_t;
typedef __rvv_bfloat16m8_t  vbfloat16m8_t;

typedef __rvv_float32mf2_t vfloat32mf2_t;
typedef __rvv_float32m1_t  vfloat32m1_t;
typedef __rvv_float32m2_t  vfloat32m2_t;
typedef __rvv_float32m4_t  vfloat32m4_t;
typedef __rvv_float32m8_t  vfloat32m8_t;

typedef __rvv_float64m1_t vfloat64m1_t;
typedef __rvv_float64m2_t vfloat64m2_t;
typedef __rvv_float64m4_t vfloat64m4_t;
typedef __rvv_float64m8_t vfloat64m8_t;

typedef __rvv_bool1_t  vbool1_t;
typedef __rvv_bool2_t  vbool2_t;
typedef __rvv_bool4_t  vbool4_t;
typedef __rvv_bool8_t  vbool8_t;
typedef __rvv_bool16_t vbool16_t;
typedef __rvv_bool32_t vbool32_t;
typedef __rvv_bool64_t vbool64_t;

// CHECK: _Z7f_i8mf8u15__rvv_int8mf8_t
void f_i8mf8(vint8mf8_t) {}
// CHECK: _Z7f_i8mf4u15__rvv_int8mf4_t
void f_i8mf4(vint8mf4_t) {}
// CHECK: _Z7f_i8mf2u15__rvv_int8mf2_t
void f_i8mf2(vint8mf2_t) {}
// CHECK: _Z6f_i8m1u14__rvv_int8m1_t
void f_i8m1(vint8m1_t) {}
// CHECK: _Z6f_i8m2u14__rvv_int8m2_t
void f_i8m2(vint8m2_t) {}
// CHECK: _Z6f_i8m4u14__rvv_int8m4_t
void f_i8m4(vint8m4_t) {}
// CHECK: _Z6f_i8m8u14__rvv_int8m8_t
void f_i8m8(vint8m8_t) {}

// CHECK: _Z7f_u8mf8u16__rvv_uint8mf8_t
void f_u8mf8(vuint8mf8_t) {}
// CHECK: _Z7f_u8mf4u16__rvv_uint8mf4_t
void f_u8mf4(vuint8mf4_t) {}
// CHECK: _Z7f_u8mf2u16__rvv_uint8mf2_t
void f_u8mf2(vuint8mf2_t) {}
// CHECK: _Z6f_u8m1u15__rvv_uint8m1_t
void f_u8m1(vuint8m1_t) {}
// CHECK: _Z6f_u8m2u15__rvv_uint8m2_t
void f_u8m2(vuint8m2_t) {}
// CHECK: _Z6f_u8m4u15__rvv_uint8m4_t
void f_u8m4(vuint8m4_t) {}
// CHECK: _Z6f_u8m8u15__rvv_uint8m8_t
void f_u8m8(vuint8m8_t) {}

// CHECK: _Z8f_i16mf4u16__rvv_int16mf4_t
void f_i16mf4(vint16mf4_t) {}
// CHECK: _Z8f_i16mf2u16__rvv_int16mf2_t
void f_i16mf2(vint16mf2_t) {}
// CHECK: _Z7f_i16m1u15__rvv_int16m1_t
void f_i16m1(vint16m1_t) {}
// CHECK: _Z7f_i16m2u15__rvv_int16m2_t
void f_i16m2(vint16m2_t) {}
// CHECK: _Z7f_i16m4u15__rvv_int16m4_t
void f_i16m4(vint16m4_t) {}
// CHECK: _Z7f_i16m8u15__rvv_int16m8_t
void f_i16m8(vint16m8_t) {}

// CHECK: _Z8f_u16mf4u17__rvv_uint16mf4_t
void f_u16mf4(vuint16mf4_t) {}
// CHECK: _Z8f_u16mf2u17__rvv_uint16mf2_t
void f_u16mf2(vuint16mf2_t) {}
// CHECK: _Z7f_u16m1u16__rvv_uint16m1_t
void f_u16m1(vuint16m1_t) {}
// CHECK: _Z7f_u16m2u16__rvv_uint16m2_t
void f_u16m2(vuint16m2_t) {}
// CHECK: _Z7f_u16m4u16__rvv_uint16m4_t
void f_u16m4(vuint16m4_t) {}
// CHECK: _Z7f_u16m8u16__rvv_uint16m8_t
void f_u16m8(vuint16m8_t) {}

// CHECK: _Z8f_i32mf2u16__rvv_int32mf2_t
void f_i32mf2(vint32mf2_t) {}
// CHECK: _Z7f_i32m1u15__rvv_int32m1_t
void f_i32m1(vint32m1_t) {}
// CHECK: _Z7f_i32m2u15__rvv_int32m2_t
void f_i32m2(vint32m2_t) {}
// CHECK: _Z7f_i32m4u15__rvv_int32m4_t
void f_i32m4(vint32m4_t) {}
// CHECK: _Z7f_i32m8u15__rvv_int32m8_t
void f_i32m8(vint32m8_t) {}

// CHECK: _Z8f_u32mf2u17__rvv_uint32mf2_t
void f_u32mf2(vuint32mf2_t) {}
// CHECK: _Z7f_u32m1u16__rvv_uint32m1_t
void f_u32m1(vuint32m1_t) {}
// CHECK: _Z7f_u32m2u16__rvv_uint32m2_t
void f_u32m2(vuint32m2_t) {}
// CHECK: _Z7f_u32m4u16__rvv_uint32m4_t
void f_u32m4(vuint32m4_t) {}
// CHECK: _Z7f_u32m8u16__rvv_uint32m8_t
void f_u32m8(vuint32m8_t) {}

// CHECK: _Z7f_i64m1u15__rvv_int64m1_t
void f_i64m1(vint64m1_t) {}
// CHECK: _Z7f_i64m2u15__rvv_int64m2_t
void f_i64m2(vint64m2_t) {}
// CHECK: _Z7f_i64m4u15__rvv_int64m4_t
void f_i64m4(vint64m4_t) {}
// CHECK: _Z7f_i64m8u15__rvv_int64m8_t
void f_i64m8(vint64m8_t) {}

// CHECK: _Z7f_u64m1u16__rvv_uint64m1_t
void f_u64m1(vuint64m1_t) {}
// CHECK: _Z7f_u64m2u16__rvv_uint64m2_t
void f_u64m2(vuint64m2_t) {}
// CHECK: _Z7f_u64m4u16__rvv_uint64m4_t
void f_u64m4(vuint64m4_t) {}
// CHECK: _Z7f_u64m8u16__rvv_uint64m8_t
void f_u64m8(vuint64m8_t) {}

// CHECK: _Z8fe4m3mf8u21__rvv_float8e4m3mf8_t
void fe4m3mf8(vfloat8e4m3mf8_t) {}

// CHECK: _Z8fe4m3mf4u21__rvv_float8e4m3mf4_t
void fe4m3mf4(vfloat8e4m3mf4_t) {}

// CHECK: _Z8fe4m3mf2u21__rvv_float8e4m3mf2_t
void fe4m3mf2(vfloat8e4m3mf2_t) {}

// CHECK: _Z7fe4m3m1u20__rvv_float8e4m3m1_t
void fe4m3m1(vfloat8e4m3m1_t) {}

// CHECK: _Z7fe4m3m2u20__rvv_float8e4m3m2_t
void fe4m3m2(vfloat8e4m3m2_t) {}

// CHECK: _Z7fe4m3m4u20__rvv_float8e4m3m4_t
void fe4m3m4(vfloat8e4m3m4_t) {}

// CHECK: _Z7fe4m3m8u20__rvv_float8e4m3m8_t
void fe4m3m8(vfloat8e4m3m8_t) {}

// CHECK: _Z8fe5m2mf8u21__rvv_float8e5m2mf8_t
void fe5m2mf8(vfloat8e5m2mf8_t) {}

// CHECK: _Z8fe5m2mf4u21__rvv_float8e5m2mf4_t
void fe5m2mf4(vfloat8e5m2mf4_t) {}

// CHECK: _Z8fe5m2mf2u21__rvv_float8e5m2mf2_t
void fe5m2mf2(vfloat8e5m2mf2_t) {}

// CHECK: _Z7fe5m2m1u20__rvv_float8e5m2m1_t
void fe5m2m1(vfloat8e5m2m1_t) {}

// CHECK: _Z7fe5m2m2u20__rvv_float8e5m2m2_t
void fe5m2m2(vfloat8e5m2m2_t) {}

// CHECK: _Z7fe5m2m4u20__rvv_float8e5m2m4_t
void fe5m2m4(vfloat8e5m2m4_t) {}

// CHECK: _Z7fe5m2m8u20__rvv_float8e5m2m8_t
void fe5m2m8(vfloat8e5m2m8_t) {}

// CHECK: _Z8f_f16mf4u18__rvv_float16mf4_t
void f_f16mf4(vfloat16mf4_t) {}
// CHECK: _Z8f_f16mf2u18__rvv_float16mf2_t
void f_f16mf2(vfloat16mf2_t) {}
// CHECK: _Z7f_f16m1u17__rvv_float16m1_t
void f_f16m1(vfloat16m1_t) {}
// CHECK: _Z7f_f16m2u17__rvv_float16m2_t
void f_f16m2(vfloat16m2_t) {}
// CHECK: _Z7f_f16m4u17__rvv_float16m4_t
void f_f16m4(vfloat16m4_t) {}
// CHECK: _Z7f_f16m8u17__rvv_float16m8_t
void f_f16m8(vfloat16m8_t) {}

// CHECK: _Z9f_bf16mf4u19__rvv_bfloat16mf4_t
void f_bf16mf4(vbfloat16mf4_t) {}
// CHECK: _Z9f_bf16mf2u19__rvv_bfloat16mf2_t
void f_bf16mf2(vbfloat16mf2_t) {}
// CHECK: _Z8f_bf16m1u18__rvv_bfloat16m1_t
void f_bf16m1(vbfloat16m1_t) {}
// CHECK: _Z8f_bf16m2u18__rvv_bfloat16m2_t
void f_bf16m2(vbfloat16m2_t) {}
// CHECK: _Z8f_bf16m4u18__rvv_bfloat16m4_t
void f_bf16m4(vbfloat16m4_t) {}
// CHECK: _Z8f_bf16m8u18__rvv_bfloat16m8_t
void f_bf16m8(vbfloat16m8_t) {}

// CHECK: _Z8f_f32mf2u18__rvv_float32mf2_t
void f_f32mf2(vfloat32mf2_t) {}
// CHECK: _Z7f_f32m1u17__rvv_float32m1_t
void f_f32m1(vfloat32m1_t) {}
// CHECK: _Z7f_f32m2u17__rvv_float32m2_t
void f_f32m2(vfloat32m2_t) {}
// CHECK: _Z7f_f32m4u17__rvv_float32m4_t
void f_f32m4(vfloat32m4_t) {}
// CHECK: _Z7f_f32m8u17__rvv_float32m8_t
void f_f32m8(vfloat32m8_t) {}

// CHECK: _Z7f_f64m1u17__rvv_float64m1_t
void f_f64m1(vfloat64m1_t) {}
// CHECK: _Z7f_f64m2u17__rvv_float64m2_t
void f_f64m2(vfloat64m2_t) {}
// CHECK: _Z7f_f64m4u17__rvv_float64m4_t
void f_f64m4(vfloat64m4_t) {}
// CHECK: _Z7f_f64m8u17__rvv_float64m8_t
void f_f64m8(vfloat64m8_t) {}

// CHECK: _Z4f_b1u13__rvv_bool1_t
void f_b1(vbool1_t) {}
// CHECK: _Z4f_b2u13__rvv_bool2_t
void f_b2(vbool2_t) {}
// CHECK: _Z4f_b4u13__rvv_bool4_t
void f_b4(vbool4_t) {}
// CHECK: _Z4f_b8u13__rvv_bool8_t
void f_b8(vbool8_t) {}
// CHECK: _Z5f_b16u14__rvv_bool16_t
void f_b16(vbool16_t) {}
// CHECK: _Z5f_b32u14__rvv_bool32_t
void f_b32(vbool32_t) {}
// CHECK: _Z5f_b64u14__rvv_bool64_t
void f_b64(vbool64_t) {}

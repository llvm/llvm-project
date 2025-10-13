// REQUIRES: webassembly-registered-target
// expected-no-diagnostics

// RUN: %clang %s -O2 -S -o - -target wasm32-unknown-unknown \
// RUN: -msimd128 -mrelaxed-simd -mfp16 -Wcast-qual -Werror | FileCheck %s

#include <wasm_simd128.h>

// CHECK-LABEL: test_v128_load:
// CHECK: v128.load 0:p2align=0{{$}}
v128_t test_v128_load(const void *mem) { return wasm_v128_load(mem); }

// CHECK-LABEL: test_v128_load8_splat:
// CHECK: v128.load8_splat 0{{$}}
v128_t test_v128_load8_splat(const void *mem) {
  return wasm_v128_load8_splat(mem);
}

// CHECK-LABEL: test_v128_load16_splat:
// CHECK: v128.load16_splat 0:p2align=0{{$}}
v128_t test_v128_load16_splat(const void *mem) {
  return wasm_v128_load16_splat(mem);
}

// CHECK-LABEL: test_v128_load32_splat:
// CHECK: v128.load32_splat 0:p2align=0{{$}}
v128_t test_v128_load32_splat(const void *mem) {
  return wasm_v128_load32_splat(mem);
}

// CHECK-LABEL: test_v128_load64_splat:
// CHECK: v128.load64_splat 0:p2align=0{{$}}
v128_t test_v128_load64_splat(const void *mem) {
  return wasm_v128_load64_splat(mem);
}

// CHECK-LABEL: test_i16x8_load8x8:
// CHECK: i16x8.load8x8_s 0:p2align=0{{$}}
v128_t test_i16x8_load8x8(const void *mem) { return wasm_i16x8_load8x8(mem); }

// CHECK-LABEL: test_u16x8_load8x8:
// CHECK: i16x8.load8x8_u 0:p2align=0{{$}}
v128_t test_u16x8_load8x8(const void *mem) { return wasm_u16x8_load8x8(mem); }

// CHECK-LABEL: test_i32x4_load16x4:
// CHECK: i32x4.load16x4_s 0:p2align=0{{$}}
v128_t test_i32x4_load16x4(const void *mem) { return wasm_i32x4_load16x4(mem); }

// CHECK-LABEL: test_u32x4_load16x4:
// CHECK: i32x4.load16x4_u 0:p2align=0{{$}}
v128_t test_u32x4_load16x4(const void *mem) { return wasm_u32x4_load16x4(mem); }

// CHECK-LABEL: test_i64x2_load32x2:
// CHECK: i64x2.load32x2_s 0:p2align=0{{$}}
v128_t test_i64x2_load32x2(const void *mem) { return wasm_i64x2_load32x2(mem); }

// CHECK-LABEL: test_u64x2_load32x2:
// CHECK: i64x2.load32x2_u 0:p2align=0{{$}}
v128_t test_u64x2_load32x2(const void *mem) { return wasm_u64x2_load32x2(mem); }

// CHECK-LABEL: test_v128_load32_zero:
// CHECK: v128.load32_zero 0:p2align=0{{$}}
v128_t test_v128_load32_zero(const void *mem) {
  return wasm_v128_load32_zero(mem);
}

// CHECK-LABEL: test_v128_load64_zero:
// CHECK: v128.load64_zero 0:p2align=0{{$}}
v128_t test_v128_load64_zero(const void *mem) {
  return wasm_v128_load64_zero(mem);
}

// CHECK-LABEL: test_v128_load8_lane:
// CHECK: v128.load8_lane 0, 15{{$}}
v128_t test_v128_load8_lane(uint8_t *ptr, v128_t vec) {
  return wasm_v128_load8_lane(ptr, vec, 15);
}

// CHECK-LABEL: test_v128_load16_lane:
// CHECK: v128.load16_lane 0:p2align=0, 7{{$}}
v128_t test_v128_load16_lane(uint16_t *ptr, v128_t vec) {
  return wasm_v128_load16_lane(ptr, vec, 7);
}

// CHECK-LABEL: test_v128_load32_lane:
// CHECK: v128.load32_lane 0:p2align=0, 3{{$}}
v128_t test_v128_load32_lane(uint32_t *ptr, v128_t vec) {
  return wasm_v128_load32_lane(ptr, vec, 3);
}

// CHECK-LABEL: test_v128_load64_lane:
// CHECK: v128.load64_lane 0:p2align=0, 1{{$}}
v128_t test_v128_load64_lane(uint64_t *ptr, v128_t vec) {
  return wasm_v128_load64_lane(ptr, vec, 1);
}

// CHECK-LABEL: test_v128_store:
// CHECK: v128.store 0:p2align=0{{$}}
void test_v128_store(void *mem, v128_t a) { return wasm_v128_store(mem, a); }

// CHECK-LABEL: test_v128_store8_lane:
// CHECK: v128.store8_lane 0, 15{{$}}
void test_v128_store8_lane(uint8_t *ptr, v128_t vec) {
  return wasm_v128_store8_lane(ptr, vec, 15);
}

// CHECK-LABEL: test_v128_store16_lane:
// CHECK: v128.store16_lane 0:p2align=0, 7{{$}}
void test_v128_store16_lane(uint16_t *ptr, v128_t vec) {
  return wasm_v128_store16_lane(ptr, vec, 7);
}

// CHECK-LABEL: test_v128_store32_lane:
// CHECK: v128.store32_lane 0:p2align=0, 3{{$}}
void test_v128_store32_lane(uint32_t *ptr, v128_t vec) {
  return wasm_v128_store32_lane(ptr, vec, 3);
}

// CHECK-LABEL: test_v128_store64_lane:
// CHECK: v128.store64_lane 0:p2align=0, 1{{$}}
void test_v128_store64_lane(uint64_t *ptr, v128_t vec) {
  return wasm_v128_store64_lane(ptr, vec, 1);
}

// CHECK-LABEL: test_i8x16_make:
// CHECK:      local.get 0{{$}}
// CHECK-NEXT: i8x16.splat{{$}}
// CHECK-NEXT: local.get 1{{$}}
// CHECK-NEXT: i8x16.replace_lane 1{{$}}
// CHECK-NEXT: local.get 2{{$}}
// CHECK-NEXT: i8x16.replace_lane 2{{$}}
// CHECK-NEXT: local.get 3{{$}}
// CHECK-NEXT: i8x16.replace_lane 3{{$}}
// CHECK-NEXT: local.get 4{{$}}
// CHECK-NEXT: i8x16.replace_lane 4{{$}}
// CHECK-NEXT: local.get 5{{$}}
// CHECK-NEXT: i8x16.replace_lane 5{{$}}
// CHECK-NEXT: local.get 6{{$}}
// CHECK-NEXT: i8x16.replace_lane 6{{$}}
// CHECK-NEXT: local.get 7{{$}}
// CHECK-NEXT: i8x16.replace_lane 7{{$}}
// CHECK-NEXT: local.get 8{{$}}
// CHECK-NEXT: i8x16.replace_lane 8{{$}}
// CHECK-NEXT: local.get 9{{$}}
// CHECK-NEXT: i8x16.replace_lane 9{{$}}
// CHECK-NEXT: local.get 10{{$}}
// CHECK-NEXT: i8x16.replace_lane 10{{$}}
// CHECK-NEXT: local.get 11{{$}}
// CHECK-NEXT: i8x16.replace_lane 11{{$}}
// CHECK-NEXT: local.get 12{{$}}
// CHECK-NEXT: i8x16.replace_lane 12{{$}}
// CHECK-NEXT: local.get 13{{$}}
// CHECK-NEXT: i8x16.replace_lane 13{{$}}
// CHECK-NEXT: local.get 14{{$}}
// CHECK-NEXT: i8x16.replace_lane 14{{$}}
// CHECK-NEXT: local.get 15{{$}}
// CHECK-NEXT: i8x16.replace_lane 15{{$}}
v128_t test_i8x16_make(int8_t c0, int8_t c1, int8_t c2, int8_t c3, int8_t c4,
                       int8_t c5, int8_t c6, int8_t c7, int8_t c8, int8_t c9,
                       int8_t c10, int8_t c11, int8_t c12, int8_t c13,
                       int8_t c14, int8_t c15) {
  return wasm_i8x16_make(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12,
                         c13, c14, c15);
}

// CHECK-LABEL: test_i16x8_make:
// CHECK:      local.get 0{{$}}
// CHECK-NEXT: i16x8.splat{{$}}
// CHECK-NEXT: local.get 1{{$}}
// CHECK-NEXT: i16x8.replace_lane 1{{$}}
// CHECK-NEXT: local.get 2{{$}}
// CHECK-NEXT: i16x8.replace_lane 2{{$}}
// CHECK-NEXT: local.get 3{{$}}
// CHECK-NEXT: i16x8.replace_lane 3{{$}}
// CHECK-NEXT: local.get 4{{$}}
// CHECK-NEXT: i16x8.replace_lane 4{{$}}
// CHECK-NEXT: local.get 5{{$}}
// CHECK-NEXT: i16x8.replace_lane 5{{$}}
// CHECK-NEXT: local.get 6{{$}}
// CHECK-NEXT: i16x8.replace_lane 6{{$}}
// CHECK-NEXT: local.get 7{{$}}
// CHECK-NEXT: i16x8.replace_lane 7{{$}}
v128_t test_i16x8_make(int16_t c0, int16_t c1, int16_t c2, int16_t c3,
                       int16_t c4, int16_t c5, int16_t c6, int16_t c7) {
  return wasm_i16x8_make(c0, c1, c2, c3, c4, c5, c6, c7);
}

// CHECK-LABEL: test_i32x4_make:
// CHECK:      local.get 0{{$}}
// CHECK-NEXT: i32x4.splat{{$}}
// CHECK-NEXT: local.get 1{{$}}
// CHECK-NEXT: i32x4.replace_lane 1{{$}}
// CHECK-NEXT: local.get 2{{$}}
// CHECK-NEXT: i32x4.replace_lane 2{{$}}
// CHECK-NEXT: local.get 3{{$}}
// CHECK-NEXT: i32x4.replace_lane 3{{$}}
v128_t test_i32x4_make(int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
  return wasm_i32x4_make(c0, c1, c2, c3);
}

// CHECK-LABEL: test_i64x2_make:
// CHECK:      local.get 0{{$}}
// CHECK-NEXT: i64x2.splat{{$}}
// CHECK-NEXT: local.get 1{{$}}
// CHECK-NEXT: i64x2.replace_lane 1{{$}}
v128_t test_i64x2_make(int64_t c0, int64_t c1) {
  return wasm_i64x2_make(c0, c1);
}

// CHECK-LABEL: test_f32x4_make:
// CHECK:      local.get 0{{$}}
// CHECK-NEXT: f32x4.splat{{$}}
// CHECK-NEXT: local.get 1{{$}}
// CHECK-NEXT: f32x4.replace_lane 1{{$}}
// CHECK-NEXT: local.get 2{{$}}
// CHECK-NEXT: f32x4.replace_lane 2{{$}}
// CHECK-NEXT: local.get 3{{$}}
// CHECK-NEXT: f32x4.replace_lane 3{{$}}
v128_t test_f32x4_make(float c0, float c1, float c2, float c3) {
  return wasm_f32x4_make(c0, c1, c2, c3);
}

// CHECK-LABEL: test_f64x2_make:
// CHECK:      local.get 0{{$}}
// CHECK-NEXT: f64x2.splat{{$}}
// CHECK-NEXT: local.get 1{{$}}
// CHECK-NEXT: f64x2.replace_lane 1{{$}}
v128_t test_f64x2_make(double c0, double c1) { return wasm_f64x2_make(c0, c1); }

// CHECK-LABEL: test_i8x16_const:
// CHECK: v128.const 50462976, 117835012, 185207048, 252579084{{$}}
v128_t test_i8x16_const() {
  return wasm_i8x16_const(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
}

// CHECK-LABEL: test_i16x8_const:
// CHECK: v128.const 65536, 196610, 327684, 458758{{$}}
v128_t test_i16x8_const() { return wasm_i16x8_const(0, 1, 2, 3, 4, 5, 6, 7); }

// CHECK-LABEL: test_i32x4_const:
// CHECK: v128.const 0, 1, 2, 3{{$}}
v128_t test_i32x4_const() { return wasm_i32x4_const(0, 1, 2, 3); }

// CHECK-LABEL: test_i64x2_const:
// CHECK: v128.const 0, 0, 1, 0{{$}}
v128_t test_i64x2_const() { return wasm_i64x2_const(0, 1); }

// CHECK-LABEL: test_f32x4_const:
// CHECK: v128.const 0, 1065353216, 1073741824, 1077936128{{$}}
v128_t test_f32x4_const() { return wasm_f32x4_const(0, 1, 2, 3); }

// CHECK-LABEL: test_f64x2_const:
// CHECK: v128.const 0, 0, 0, 1072693248{{$}}
v128_t test_f64x2_const() { return wasm_f64x2_const(0, 1); }

// CHECK-LABEL: test_i8x16_splat:
// CHECK: i8x16.splat{{$}}
v128_t test_i8x16_splat(int8_t a) { return wasm_i8x16_splat(a); }

// CHECK-LABEL: test_i8x16_extract_lane:
// CHECK: i8x16.extract_lane_s 15{{$}}
int8_t test_i8x16_extract_lane(v128_t a) {
  return wasm_i8x16_extract_lane(a, 15);
}

// CHECK-LABEL: test_u8x16_extract_lane:
// CHECK: i8x16.extract_lane_u 15{{$}}
uint8_t test_u8x16_extract_lane(v128_t a) {
  return wasm_u8x16_extract_lane(a, 15);
}

// CHECK-LABEL: test_i8x16_replace_lane:
// CHECK: i8x16.replace_lane 15{{$}}
v128_t test_i8x16_replace_lane(v128_t a, int8_t b) {
  return wasm_i8x16_replace_lane(a, 15, b);
}

// CHECK-LABEL: test_i16x8_splat:
// CHECK: i16x8.splat{{$}}
v128_t test_i16x8_splat(int16_t a) { return wasm_i16x8_splat(a); }

// CHECK-LABEL: test_i16x8_extract_lane:
// CHECK: i16x8.extract_lane_s 7{{$}}
int16_t test_i16x8_extract_lane(v128_t a) {
  return wasm_i16x8_extract_lane(a, 7);
}

// CHECK-LABEL: test_u16x8_extract_lane:
// CHECK: i16x8.extract_lane_u 7{{$}}
uint16_t test_u16x8_extract_lane(v128_t a) {
  return wasm_u16x8_extract_lane(a, 7);
}

// CHECK-LABEL: test_i16x8_replace_lane:
// CHECK: i16x8.replace_lane 7{{$}}
v128_t test_i16x8_replace_lane(v128_t a, int16_t b) {
  return wasm_i16x8_replace_lane(a, 7, b);
}

// CHECK-LABEL: test_i32x4_splat:
// CHECK: i32x4.splat{{$}}
v128_t test_i32x4_splat(int32_t a) { return wasm_i32x4_splat(a); }

// CHECK-LABEL: test_i32x4_extract_lane:
// CHECK: i32x4.extract_lane 3{{$}}
int32_t test_i32x4_extract_lane(v128_t a) {
  return wasm_i32x4_extract_lane(a, 3);
}

// CHECK-LABEL: test_i32x4_replace_lane:
// CHECK: i32x4.replace_lane 3{{$}}
v128_t test_i32x4_replace_lane(v128_t a, int32_t b) {
  return wasm_i32x4_replace_lane(a, 3, b);
}

// CHECK-LABEL: test_i64x2_splat:
// CHECK: i64x2.splat{{$}}
v128_t test_i64x2_splat(int64_t a) { return wasm_i64x2_splat(a); }

// CHECK-LABEL: test_i64x2_extract_lane:
// CHECK: i64x2.extract_lane 1{{$}}
int64_t test_i64x2_extract_lane(v128_t a) {
  return wasm_i64x2_extract_lane(a, 1);
}

// CHECK-LABEL: test_i64x2_replace_lane:
// CHECK: i64x2.replace_lane 1{{$}}
v128_t test_i64x2_replace_lane(v128_t a, int64_t b) {
  return wasm_i64x2_replace_lane(a, 1, b);
}

// CHECK-LABEL: test_f32x4_splat:
// CHECK: f32x4.splat{{$}}
v128_t test_f32x4_splat(float a) { return wasm_f32x4_splat(a); }

// CHECK-LABEL: test_f32x4_extract_lane:
// CHECK: f32x4.extract_lane 3{{$}}
float test_f32x4_extract_lane(v128_t a) {
  return wasm_f32x4_extract_lane(a, 3);
}

// CHECK-LABEL: test_f32x4_replace_lane:
// CHECK: f32x4.replace_lane 3{{$}}
v128_t test_f32x4_replace_lane(v128_t a, float b) {
  return wasm_f32x4_replace_lane(a, 3, b);
}

// CHECK-LABEL: test_f64x2_splat:
// CHECK: f64x2.splat{{$}}
v128_t test_f64x2_splat(double a) { return wasm_f64x2_splat(a); }

// CHECK-LABEL: test_f64x2_extract_lane:
// CHECK: f64x2.extract_lane 1{{$}}
double test_f64x2_extract_lane(v128_t a) {
  return wasm_f64x2_extract_lane(a, 1);
}

// CHECK-LABEL: test_f64x2_replace_lane:
// CHECK: f64x2.replace_lane 1{{$}}
v128_t test_f64x2_replace_lane(v128_t a, double b) {
  return wasm_f64x2_replace_lane(a, 1, b);
}

// CHECK-LABEL: test_i8x16_eq:
// CHECK: i8x16.eq{{$}}
v128_t test_i8x16_eq(v128_t a, v128_t b) { return wasm_i8x16_eq(a, b); }

// CHECK-LABEL: test_i8x16_ne:
// CHECK: i8x16.ne{{$}}
v128_t test_i8x16_ne(v128_t a, v128_t b) { return wasm_i8x16_ne(a, b); }

// CHECK-LABEL: test_i8x16_lt:
// CHECK: i8x16.lt_s{{$}}
v128_t test_i8x16_lt(v128_t a, v128_t b) { return wasm_i8x16_lt(a, b); }

// CHECK-LABEL: test_u8x16_lt:
// CHECK: i8x16.lt_u{{$}}
v128_t test_u8x16_lt(v128_t a, v128_t b) { return wasm_u8x16_lt(a, b); }

// CHECK-LABEL: test_i8x16_gt:
// CHECK: i8x16.gt_s{{$}}
v128_t test_i8x16_gt(v128_t a, v128_t b) { return wasm_i8x16_gt(a, b); }

// CHECK-LABEL: test_u8x16_gt:
// CHECK: i8x16.gt_u{{$}}
v128_t test_u8x16_gt(v128_t a, v128_t b) { return wasm_u8x16_gt(a, b); }

// CHECK-LABEL: test_i8x16_le:
// CHECK: i8x16.le_s{{$}}
v128_t test_i8x16_le(v128_t a, v128_t b) { return wasm_i8x16_le(a, b); }

// CHECK-LABEL: test_u8x16_le:
// CHECK: i8x16.le_u{{$}}
v128_t test_u8x16_le(v128_t a, v128_t b) { return wasm_u8x16_le(a, b); }

// CHECK-LABEL: test_i8x16_ge:
// CHECK: i8x16.ge_s{{$}}
v128_t test_i8x16_ge(v128_t a, v128_t b) { return wasm_i8x16_ge(a, b); }

// CHECK-LABEL: test_u8x16_ge:
// CHECK: i8x16.ge_u{{$}}
v128_t test_u8x16_ge(v128_t a, v128_t b) { return wasm_u8x16_ge(a, b); }

// CHECK-LABEL: test_i16x8_eq:
// CHECK: i16x8.eq{{$}}
v128_t test_i16x8_eq(v128_t a, v128_t b) { return wasm_i16x8_eq(a, b); }

// CHECK-LABEL: test_i16x8_ne:
// CHECK: i16x8.ne{{$}}
v128_t test_i16x8_ne(v128_t a, v128_t b) { return wasm_i16x8_ne(a, b); }

// CHECK-LABEL: test_i16x8_lt:
// CHECK: i16x8.lt_s{{$}}
v128_t test_i16x8_lt(v128_t a, v128_t b) { return wasm_i16x8_lt(a, b); }

// CHECK-LABEL: test_u16x8_lt:
// CHECK: i16x8.lt_u{{$}}
v128_t test_u16x8_lt(v128_t a, v128_t b) { return wasm_u16x8_lt(a, b); }

// CHECK-LABEL: test_i16x8_gt:
// CHECK: i16x8.gt_s{{$}}
v128_t test_i16x8_gt(v128_t a, v128_t b) { return wasm_i16x8_gt(a, b); }

// CHECK-LABEL: test_u16x8_gt:
// CHECK: i16x8.gt_u{{$}}
v128_t test_u16x8_gt(v128_t a, v128_t b) { return wasm_u16x8_gt(a, b); }

// CHECK-LABEL: test_i16x8_le:
// CHECK: i16x8.le_s{{$}}
v128_t test_i16x8_le(v128_t a, v128_t b) { return wasm_i16x8_le(a, b); }

// CHECK-LABEL: test_u16x8_le:
// CHECK: i16x8.le_u{{$}}
v128_t test_u16x8_le(v128_t a, v128_t b) { return wasm_u16x8_le(a, b); }

// CHECK-LABEL: test_i16x8_ge:
// CHECK: i16x8.ge_s{{$}}
v128_t test_i16x8_ge(v128_t a, v128_t b) { return wasm_i16x8_ge(a, b); }

// CHECK-LABEL: test_u16x8_ge:
// CHECK: i16x8.ge_u{{$}}
v128_t test_u16x8_ge(v128_t a, v128_t b) { return wasm_u16x8_ge(a, b); }

// CHECK-LABEL: test_i32x4_eq:
// CHECK: i32x4.eq{{$}}
v128_t test_i32x4_eq(v128_t a, v128_t b) { return wasm_i32x4_eq(a, b); }

// CHECK-LABEL: test_i32x4_ne:
// CHECK: i32x4.ne{{$}}
v128_t test_i32x4_ne(v128_t a, v128_t b) { return wasm_i32x4_ne(a, b); }

// CHECK-LABEL: test_i32x4_lt:
// CHECK: i32x4.lt_s{{$}}
v128_t test_i32x4_lt(v128_t a, v128_t b) { return wasm_i32x4_lt(a, b); }

// CHECK-LABEL: test_u32x4_lt:
// CHECK: i32x4.lt_u{{$}}
v128_t test_u32x4_lt(v128_t a, v128_t b) { return wasm_u32x4_lt(a, b); }

// CHECK-LABEL: test_i32x4_gt:
// CHECK: i32x4.gt_s{{$}}
v128_t test_i32x4_gt(v128_t a, v128_t b) { return wasm_i32x4_gt(a, b); }

// CHECK-LABEL: test_u32x4_gt:
// CHECK: i32x4.gt_u{{$}}
v128_t test_u32x4_gt(v128_t a, v128_t b) { return wasm_u32x4_gt(a, b); }

// CHECK-LABEL: test_i32x4_le:
// CHECK: i32x4.le_s{{$}}
v128_t test_i32x4_le(v128_t a, v128_t b) { return wasm_i32x4_le(a, b); }

// CHECK-LABEL: test_u32x4_le:
// CHECK: i32x4.le_u{{$}}
v128_t test_u32x4_le(v128_t a, v128_t b) { return wasm_u32x4_le(a, b); }

// CHECK-LABEL: test_i32x4_ge:
// CHECK: i32x4.ge_s{{$}}
v128_t test_i32x4_ge(v128_t a, v128_t b) { return wasm_i32x4_ge(a, b); }

// CHECK-LABEL: test_u32x4_ge:
// CHECK: i32x4.ge_u{{$}}
v128_t test_u32x4_ge(v128_t a, v128_t b) { return wasm_u32x4_ge(a, b); }

// CHECK-LABEL: test_i64x2_eq:
// CHECK: i64x2.eq{{$}}
v128_t test_i64x2_eq(v128_t a, v128_t b) { return wasm_i64x2_eq(a, b); }

// CHECK-LABEL: test_i64x2_ne:
// CHECK: i64x2.ne{{$}}
v128_t test_i64x2_ne(v128_t a, v128_t b) { return wasm_i64x2_ne(a, b); }

// CHECK-LABEL: test_i64x2_lt:
// CHECK: i64x2.lt_s{{$}}
v128_t test_i64x2_lt(v128_t a, v128_t b) { return wasm_i64x2_lt(a, b); }

// CHECK-LABEL: test_i64x2_gt:
// CHECK: i64x2.gt_s{{$}}
v128_t test_i64x2_gt(v128_t a, v128_t b) { return wasm_i64x2_gt(a, b); }

// CHECK-LABEL: test_i64x2_le:
// CHECK: i64x2.le_s{{$}}
v128_t test_i64x2_le(v128_t a, v128_t b) { return wasm_i64x2_le(a, b); }

// CHECK-LABEL: test_i64x2_ge:
// CHECK: i64x2.ge_s{{$}}
v128_t test_i64x2_ge(v128_t a, v128_t b) { return wasm_i64x2_ge(a, b); }

// CHECK-LABEL: test_f32x4_eq:
// CHECK: f32x4.eq{{$}}
v128_t test_f32x4_eq(v128_t a, v128_t b) { return wasm_f32x4_eq(a, b); }

// CHECK-LABEL: test_f32x4_ne:
// CHECK: f32x4.ne{{$}}
v128_t test_f32x4_ne(v128_t a, v128_t b) { return wasm_f32x4_ne(a, b); }

// CHECK-LABEL: test_f32x4_lt:
// CHECK: f32x4.lt{{$}}
v128_t test_f32x4_lt(v128_t a, v128_t b) { return wasm_f32x4_lt(a, b); }

// CHECK-LABEL: test_f32x4_gt:
// CHECK: f32x4.gt{{$}}
v128_t test_f32x4_gt(v128_t a, v128_t b) { return wasm_f32x4_gt(a, b); }

// CHECK-LABEL: test_f32x4_le:
// CHECK: f32x4.le{{$}}
v128_t test_f32x4_le(v128_t a, v128_t b) { return wasm_f32x4_le(a, b); }

// CHECK-LABEL: test_f32x4_ge:
// CHECK: f32x4.ge{{$}}
v128_t test_f32x4_ge(v128_t a, v128_t b) { return wasm_f32x4_ge(a, b); }

// CHECK-LABEL: test_f64x2_eq:
// CHECK: f64x2.eq{{$}}
v128_t test_f64x2_eq(v128_t a, v128_t b) { return wasm_f64x2_eq(a, b); }

// CHECK-LABEL: test_f64x2_ne:
// CHECK: f64x2.ne{{$}}
v128_t test_f64x2_ne(v128_t a, v128_t b) { return wasm_f64x2_ne(a, b); }

// CHECK-LABEL: test_f64x2_lt:
// CHECK: f64x2.lt{{$}}
v128_t test_f64x2_lt(v128_t a, v128_t b) { return wasm_f64x2_lt(a, b); }

// CHECK-LABEL: test_f64x2_gt:
// CHECK: f64x2.gt{{$}}
v128_t test_f64x2_gt(v128_t a, v128_t b) { return wasm_f64x2_gt(a, b); }

// CHECK-LABEL: test_f64x2_le:
// CHECK: f64x2.le{{$}}
v128_t test_f64x2_le(v128_t a, v128_t b) { return wasm_f64x2_le(a, b); }

// CHECK-LABEL: test_f64x2_ge:
// CHECK: f64x2.ge{{$}}
v128_t test_f64x2_ge(v128_t a, v128_t b) { return wasm_f64x2_ge(a, b); }

// CHECK-LABEL: test_v128_not:
// CHECK: v128.not{{$}}
v128_t test_v128_not(v128_t a) { return wasm_v128_not(a); }

// CHECK-LABEL: test_v128_and:
// CHECK: v128.and{{$}}
v128_t test_v128_and(v128_t a, v128_t b) { return wasm_v128_and(a, b); }

// CHECK-LABEL: test_v128_or:
// CHECK: v128.or{{$}}
v128_t test_v128_or(v128_t a, v128_t b) { return wasm_v128_or(a, b); }

// CHECK-LABEL: test_v128_xor:
// CHECK: v128.xor{{$}}
v128_t test_v128_xor(v128_t a, v128_t b) { return wasm_v128_xor(a, b); }

// CHECK-LABEL: test_v128_andnot:
// CHECK: v128.andnot{{$}}
v128_t test_v128_andnot(v128_t a, v128_t b) { return wasm_v128_andnot(a, b); }

// CHECK-LABEL: test_v128_any_true:
// CHECK: v128.any_true{{$}}
bool test_v128_any_true(v128_t a) { return wasm_v128_any_true(a); }

// CHECK-LABEL: test_v128_bitselect:
// CHECK: v128.bitselect{{$}}
v128_t test_v128_bitselect(v128_t a, v128_t b, v128_t mask) {
  return wasm_v128_bitselect(a, b, mask);
}

// CHECK-LABEL: test_i8x16_abs:
// CHECK: i8x16.abs{{$}}
v128_t test_i8x16_abs(v128_t a) { return wasm_i8x16_abs(a); }

// CHECK-LABEL: test_i8x16_neg:
// CHECK: i8x16.neg{{$}}
v128_t test_i8x16_neg(v128_t a) { return wasm_i8x16_neg(a); }

// CHECK-LABEL: test_i8x16_all_true:
// CHECK: i8x16.all_true{{$}}
bool test_i8x16_all_true(v128_t a) { return wasm_i8x16_all_true(a); }

// CHECK-LABEL: test_i8x16_bitmask:
// CHECK: i8x16.bitmask{{$}}
int32_t test_i8x16_bitmask(v128_t a) { return wasm_i8x16_bitmask(a); }

// CHECK-LABEL: test_i8x16_popcnt:
// CHECK: i8x16.popcnt{{$}}
v128_t test_i8x16_popcnt(v128_t a) { return wasm_i8x16_popcnt(a); }

// CHECK-LABEL: test_i8x16_shl:
// CHECK: i8x16.shl{{$}}
v128_t test_i8x16_shl(v128_t a, int32_t b) { return wasm_i8x16_shl(a, b); }

// CHECK-LABEL: test_i8x16_shr:
// CHECK: i8x16.shr_s{{$}}
v128_t test_i8x16_shr(v128_t a, int32_t b) { return wasm_i8x16_shr(a, b); }

// CHECK-LABEL: test_u8x16_shr:
// CHECK: i8x16.shr_u{{$}}
v128_t test_u8x16_shr(v128_t a, int32_t b) { return wasm_u8x16_shr(a, b); }

// CHECK-LABEL: test_i8x16_add:
// CHECK: i8x16.add{{$}}
v128_t test_i8x16_add(v128_t a, v128_t b) { return wasm_i8x16_add(a, b); }

// CHECK-LABEL: test_i8x16_add_sat:
// CHECK: i8x16.add_sat_s{{$}}
v128_t test_i8x16_add_sat(v128_t a, v128_t b) {
  return wasm_i8x16_add_sat(a, b);
}

// CHECK-LABEL: test_u8x16_add_sat:
// CHECK: i8x16.add_sat_u{{$}}
v128_t test_u8x16_add_sat(v128_t a, v128_t b) {
  return wasm_u8x16_add_sat(a, b);
}

// CHECK-LABEL: test_i8x16_sub:
// CHECK: i8x16.sub{{$}}
v128_t test_i8x16_sub(v128_t a, v128_t b) { return wasm_i8x16_sub(a, b); }

// CHECK-LABEL: test_i8x16_sub_sat:
// CHECK: i8x16.sub_sat_s{{$}}
v128_t test_i8x16_sub_sat(v128_t a, v128_t b) {
  return wasm_i8x16_sub_sat(a, b);
}

// CHECK-LABEL: test_u8x16_sub_sat:
// CHECK: i8x16.sub_sat_u{{$}}
v128_t test_u8x16_sub_sat(v128_t a, v128_t b) {
  return wasm_u8x16_sub_sat(a, b);
}

// CHECK-LABEL: test_i8x16_min:
// CHECK: i8x16.min_s{{$}}
v128_t test_i8x16_min(v128_t a, v128_t b) { return wasm_i8x16_min(a, b); }

// CHECK-LABEL: test_u8x16_min:
// CHECK: i8x16.min_u{{$}}
v128_t test_u8x16_min(v128_t a, v128_t b) { return wasm_u8x16_min(a, b); }

// CHECK-LABEL: test_i8x16_max:
// CHECK: i8x16.max_s{{$}}
v128_t test_i8x16_max(v128_t a, v128_t b) { return wasm_i8x16_max(a, b); }

// CHECK-LABEL: test_u8x16_max:
// CHECK: i8x16.max_u{{$}}
v128_t test_u8x16_max(v128_t a, v128_t b) { return wasm_u8x16_max(a, b); }

// CHECK-LABEL: test_u8x16_avgr:
// CHECK: i8x16.avgr_u{{$}}
v128_t test_u8x16_avgr(v128_t a, v128_t b) { return wasm_u8x16_avgr(a, b); }

// CHECK-LABEL: test_i16x8_abs:
// CHECK: i16x8.abs{{$}}
v128_t test_i16x8_abs(v128_t a) { return wasm_i16x8_abs(a); }

// CHECK-LABEL: test_i16x8_neg:
// CHECK: i16x8.neg{{$}}
v128_t test_i16x8_neg(v128_t a) { return wasm_i16x8_neg(a); }

// CHECK-LABEL: test_i16x8_all_true:
// CHECK: i16x8.all_true{{$}}
bool test_i16x8_all_true(v128_t a) { return wasm_i16x8_all_true(a); }

// CHECK-LABEL: test_i16x8_bitmask:
// CHECK: i16x8.bitmask{{$}}
int32_t test_i16x8_bitmask(v128_t a) { return wasm_i16x8_bitmask(a); }

// CHECK-LABEL: test_i16x8_shl:
// CHECK: i16x8.shl{{$}}
v128_t test_i16x8_shl(v128_t a, int32_t b) { return wasm_i16x8_shl(a, b); }

// CHECK-LABEL: test_i16x8_shr:
// CHECK: i16x8.shr_s{{$}}
v128_t test_i16x8_shr(v128_t a, int32_t b) { return wasm_i16x8_shr(a, b); }

// CHECK-LABEL: test_u16x8_shr:
// CHECK: i16x8.shr_u{{$}}
v128_t test_u16x8_shr(v128_t a, int32_t b) { return wasm_u16x8_shr(a, b); }

// CHECK-LABEL: test_i16x8_add:
// CHECK: i16x8.add{{$}}
v128_t test_i16x8_add(v128_t a, v128_t b) { return wasm_i16x8_add(a, b); }

// CHECK-LABEL: test_i16x8_add_sat:
// CHECK: i16x8.add_sat_s{{$}}
v128_t test_i16x8_add_sat(v128_t a, v128_t b) {
  return wasm_i16x8_add_sat(a, b);
}

// CHECK-LABEL: test_u16x8_add_sat:
// CHECK: i16x8.add_sat_u{{$}}
v128_t test_u16x8_add_sat(v128_t a, v128_t b) {
  return wasm_u16x8_add_sat(a, b);
}

// CHECK-LABEL: test_i16x8_sub:
// CHECK: i16x8.sub{{$}}
v128_t test_i16x8_sub(v128_t a, v128_t b) { return wasm_i16x8_sub(a, b); }

// CHECK-LABEL: test_i16x8_sub_sat:
// CHECK: i16x8.sub_sat_s{{$}}
v128_t test_i16x8_sub_sat(v128_t a, v128_t b) {
  return wasm_i16x8_sub_sat(a, b);
}

// CHECK-LABEL: test_u16x8_sub_sat:
// CHECK: i16x8.sub_sat_u{{$}}
v128_t test_u16x8_sub_sat(v128_t a, v128_t b) {
  return wasm_u16x8_sub_sat(a, b);
}

// CHECK-LABEL: test_i16x8_mul:
// CHECK: i16x8.mul{{$}}
v128_t test_i16x8_mul(v128_t a, v128_t b) { return wasm_i16x8_mul(a, b); }

// CHECK-LABEL: test_i16x8_min:
// CHECK: i16x8.min_s{{$}}
v128_t test_i16x8_min(v128_t a, v128_t b) { return wasm_i16x8_min(a, b); }

// CHECK-LABEL: test_u16x8_min:
// CHECK: i16x8.min_u{{$}}
v128_t test_u16x8_min(v128_t a, v128_t b) { return wasm_u16x8_min(a, b); }

// CHECK-LABEL: test_i16x8_max:
// CHECK: i16x8.max_s{{$}}
v128_t test_i16x8_max(v128_t a, v128_t b) { return wasm_i16x8_max(a, b); }

// CHECK-LABEL: test_u16x8_max:
// CHECK: i16x8.max_u{{$}}
v128_t test_u16x8_max(v128_t a, v128_t b) { return wasm_u16x8_max(a, b); }

// CHECK-LABEL: test_u16x8_avgr:
// CHECK: i16x8.avgr_u{{$}}
v128_t test_u16x8_avgr(v128_t a, v128_t b) { return wasm_u16x8_avgr(a, b); }

// CHECK-LABEL: test_i32x4_abs:
// CHECK: i32x4.abs{{$}}
v128_t test_i32x4_abs(v128_t a) { return wasm_i32x4_abs(a); }

// CHECK-LABEL: test_i32x4_neg:
// CHECK: i32x4.neg{{$}}
v128_t test_i32x4_neg(v128_t a) { return wasm_i32x4_neg(a); }

// CHECK-LABEL: test_i32x4_all_true:
// CHECK: i32x4.all_true{{$}}
bool test_i32x4_all_true(v128_t a) { return wasm_i32x4_all_true(a); }

// CHECK-LABEL: test_i32x4_bitmask:
// CHECK: i32x4.bitmask{{$}}
int32_t test_i32x4_bitmask(v128_t a) { return wasm_i32x4_bitmask(a); }

// CHECK-LABEL: test_i32x4_shl:
// CHECK: i32x4.shl{{$}}
v128_t test_i32x4_shl(v128_t a, int32_t b) { return wasm_i32x4_shl(a, b); }

// CHECK-LABEL: test_i32x4_shr:
// CHECK: i32x4.shr_s{{$}}
v128_t test_i32x4_shr(v128_t a, int32_t b) { return wasm_i32x4_shr(a, b); }

// CHECK-LABEL: test_u32x4_shr:
// CHECK: i32x4.shr_u{{$}}
v128_t test_u32x4_shr(v128_t a, int32_t b) { return wasm_u32x4_shr(a, b); }

// CHECK-LABEL: test_i32x4_add:
// CHECK: i32x4.add{{$}}
v128_t test_i32x4_add(v128_t a, v128_t b) { return wasm_i32x4_add(a, b); }

// CHECK-LABEL: test_i32x4_sub:
// CHECK: i32x4.sub{{$}}
v128_t test_i32x4_sub(v128_t a, v128_t b) { return wasm_i32x4_sub(a, b); }

// CHECK-LABEL: test_i32x4_mul:
// CHECK: i32x4.mul{{$}}
v128_t test_i32x4_mul(v128_t a, v128_t b) { return wasm_i32x4_mul(a, b); }

// CHECK-LABEL: test_i32x4_min:
// CHECK: i32x4.min_s{{$}}
v128_t test_i32x4_min(v128_t a, v128_t b) { return wasm_i32x4_min(a, b); }

// CHECK-LABEL: test_u32x4_min:
// CHECK: i32x4.min_u{{$}}
v128_t test_u32x4_min(v128_t a, v128_t b) { return wasm_u32x4_min(a, b); }

// CHECK-LABEL: test_i32x4_max:
// CHECK: i32x4.max_s{{$}}
v128_t test_i32x4_max(v128_t a, v128_t b) { return wasm_i32x4_max(a, b); }

// CHECK-LABEL: test_u32x4_max:
// CHECK: i32x4.max_u{{$}}
v128_t test_u32x4_max(v128_t a, v128_t b) { return wasm_u32x4_max(a, b); }

// CHECK-LABEL: test_i32x4_dot_i16x8:
// CHECK: i32x4.dot_i16x8_s{{$}}
v128_t test_i32x4_dot_i16x8(v128_t a, v128_t b) {
  return wasm_i32x4_dot_i16x8(a, b);
}

// CHECK-LABEL: test_i64x2_abs:
// CHECK: i64x2.abs{{$}}
v128_t test_i64x2_abs(v128_t a) { return wasm_i64x2_abs(a); }

// CHECK-LABEL: test_i64x2_neg:
// CHECK: i64x2.neg{{$}}
v128_t test_i64x2_neg(v128_t a) { return wasm_i64x2_neg(a); }

// CHECK-LABEL: test_i64x2_all_true:
// CHECK: i64x2.all_true{{$}}
bool test_i64x2_all_true(v128_t a) { return wasm_i64x2_all_true(a); }

// CHECK-LABEL: test_i64x2_bitmask:
// CHECK: i64x2.bitmask{{$}}
int32_t test_i64x2_bitmask(v128_t a) { return wasm_i64x2_bitmask(a); }

// CHECK-LABEL: test_i64x2_shl:
// CHECK: i64x2.shl{{$}}
v128_t test_i64x2_shl(v128_t a, int32_t b) { return wasm_i64x2_shl(a, b); }

// CHECK-LABEL: test_i64x2_shr:
// CHECK: i64x2.shr_s{{$}}
v128_t test_i64x2_shr(v128_t a, int32_t b) { return wasm_i64x2_shr(a, b); }

// CHECK-LABEL: test_u64x2_shr:
// CHECK: i64x2.shr_u{{$}}
v128_t test_u64x2_shr(v128_t a, int32_t b) { return wasm_u64x2_shr(a, b); }

// CHECK-LABEL: test_i64x2_add:
// CHECK: i64x2.add{{$}}
v128_t test_i64x2_add(v128_t a, v128_t b) { return wasm_i64x2_add(a, b); }

// CHECK-LABEL: test_i64x2_sub:
// CHECK: i64x2.sub{{$}}
v128_t test_i64x2_sub(v128_t a, v128_t b) { return wasm_i64x2_sub(a, b); }

// CHECK-LABEL: test_i64x2_mul:
// CHECK: i64x2.mul{{$}}
v128_t test_i64x2_mul(v128_t a, v128_t b) { return wasm_i64x2_mul(a, b); }

// CHECK-LABEL: test_f32x4_abs:
// CHECK: f32x4.abs{{$}}
v128_t test_f32x4_abs(v128_t a) { return wasm_f32x4_abs(a); }

// CHECK-LABEL: test_f32x4_neg:
// CHECK: f32x4.neg{{$}}
v128_t test_f32x4_neg(v128_t a) { return wasm_f32x4_neg(a); }

// CHECK-LABEL: test_f32x4_sqrt:
// CHECK: f32x4.sqrt{{$}}
v128_t test_f32x4_sqrt(v128_t a) { return wasm_f32x4_sqrt(a); }

// CHECK-LABEL: test_f32x4_ceil:
// CHECK: f32x4.ceil{{$}}
v128_t test_f32x4_ceil(v128_t a) { return wasm_f32x4_ceil(a); }

// CHECK-LABEL: test_f32x4_floor:
// CHECK: f32x4.floor{{$}}
v128_t test_f32x4_floor(v128_t a) { return wasm_f32x4_floor(a); }

// CHECK-LABEL: test_f32x4_trunc:
// CHECK: f32x4.trunc{{$}}
v128_t test_f32x4_trunc(v128_t a) { return wasm_f32x4_trunc(a); }

// CHECK-LABEL: test_f32x4_nearest:
// CHECK: f32x4.nearest{{$}}
v128_t test_f32x4_nearest(v128_t a) { return wasm_f32x4_nearest(a); }

// CHECK-LABEL: test_f32x4_add:
// CHECK: f32x4.add{{$}}
v128_t test_f32x4_add(v128_t a, v128_t b) { return wasm_f32x4_add(a, b); }

// CHECK-LABEL: test_f32x4_sub:
// CHECK: f32x4.sub{{$}}
v128_t test_f32x4_sub(v128_t a, v128_t b) { return wasm_f32x4_sub(a, b); }

// CHECK-LABEL: test_f32x4_mul:
// CHECK: f32x4.mul{{$}}
v128_t test_f32x4_mul(v128_t a, v128_t b) { return wasm_f32x4_mul(a, b); }

// CHECK-LABEL: test_f32x4_div:
// CHECK: f32x4.div{{$}}
v128_t test_f32x4_div(v128_t a, v128_t b) { return wasm_f32x4_div(a, b); }

// CHECK-LABEL: test_f32x4_min:
// CHECK: f32x4.min{{$}}
v128_t test_f32x4_min(v128_t a, v128_t b) { return wasm_f32x4_min(a, b); }

// CHECK-LABEL: test_f32x4_max:
// CHECK: f32x4.max{{$}}
v128_t test_f32x4_max(v128_t a, v128_t b) { return wasm_f32x4_max(a, b); }

// CHECK-LABEL: test_f32x4_pmin:
// CHECK: f32x4.pmin{{$}}
v128_t test_f32x4_pmin(v128_t a, v128_t b) { return wasm_f32x4_pmin(a, b); }

// CHECK-LABEL: test_f32x4_pmax:
// CHECK: f32x4.pmax{{$}}
v128_t test_f32x4_pmax(v128_t a, v128_t b) { return wasm_f32x4_pmax(a, b); }

// CHECK-LABEL: test_f64x2_abs:
// CHECK: f64x2.abs{{$}}
v128_t test_f64x2_abs(v128_t a) { return wasm_f64x2_abs(a); }

// CHECK-LABEL: test_f64x2_neg:
// CHECK: f64x2.neg{{$}}
v128_t test_f64x2_neg(v128_t a) { return wasm_f64x2_neg(a); }

// CHECK-LABEL: test_f64x2_sqrt:
// CHECK: f64x2.sqrt{{$}}
v128_t test_f64x2_sqrt(v128_t a) { return wasm_f64x2_sqrt(a); }

// CHECK-LABEL: test_f64x2_ceil:
// CHECK: f64x2.ceil{{$}}
v128_t test_f64x2_ceil(v128_t a) { return wasm_f64x2_ceil(a); }

// CHECK-LABEL: test_f64x2_floor:
// CHECK: f64x2.floor{{$}}
v128_t test_f64x2_floor(v128_t a) { return wasm_f64x2_floor(a); }

// CHECK-LABEL: test_f64x2_trunc:
// CHECK: f64x2.trunc{{$}}
v128_t test_f64x2_trunc(v128_t a) { return wasm_f64x2_trunc(a); }

// CHECK-LABEL: test_f64x2_nearest:
// CHECK: f64x2.nearest{{$}}
v128_t test_f64x2_nearest(v128_t a) { return wasm_f64x2_nearest(a); }

// CHECK-LABEL: test_f64x2_add:
// CHECK: f64x2.add{{$}}
v128_t test_f64x2_add(v128_t a, v128_t b) { return wasm_f64x2_add(a, b); }

// CHECK-LABEL: test_f64x2_sub:
// CHECK: f64x2.sub{{$}}
v128_t test_f64x2_sub(v128_t a, v128_t b) { return wasm_f64x2_sub(a, b); }

// CHECK-LABEL: test_f64x2_mul:
// CHECK: f64x2.mul{{$}}
v128_t test_f64x2_mul(v128_t a, v128_t b) { return wasm_f64x2_mul(a, b); }

// CHECK-LABEL: test_f64x2_div:
// CHECK: f64x2.div{{$}}
v128_t test_f64x2_div(v128_t a, v128_t b) { return wasm_f64x2_div(a, b); }

// CHECK-LABEL: test_f64x2_min:
// CHECK: f64x2.min{{$}}
v128_t test_f64x2_min(v128_t a, v128_t b) { return wasm_f64x2_min(a, b); }

// CHECK-LABEL: test_f64x2_max:
// CHECK: f64x2.max{{$}}
v128_t test_f64x2_max(v128_t a, v128_t b) { return wasm_f64x2_max(a, b); }

// CHECK-LABEL: test_f64x2_pmin:
// CHECK: f64x2.pmin{{$}}
v128_t test_f64x2_pmin(v128_t a, v128_t b) { return wasm_f64x2_pmin(a, b); }

// CHECK-LABEL: test_f64x2_pmax:
// CHECK: f64x2.pmax{{$}}
v128_t test_f64x2_pmax(v128_t a, v128_t b) { return wasm_f64x2_pmax(a, b); }

// CHECK-LABEL: test_i32x4_trunc_sat_f32x4:
// CHECK: i32x4.trunc_sat_f32x4_s{{$}}
v128_t test_i32x4_trunc_sat_f32x4(v128_t a) {
  return wasm_i32x4_trunc_sat_f32x4(a);
}

// CHECK-LABEL: test_u32x4_trunc_sat_f32x4:
// CHECK: i32x4.trunc_sat_f32x4_u{{$}}
v128_t test_u32x4_trunc_sat_f32x4(v128_t a) {
  return wasm_u32x4_trunc_sat_f32x4(a);
}

// CHECK-LABEL: test_f32x4_convert_i32x4:
// CHECK: f32x4.convert_i32x4_s{{$}}
v128_t test_f32x4_convert_i32x4(v128_t a) {
  return wasm_f32x4_convert_i32x4(a);
}

// CHECK-LABEL: test_f32x4_convert_u32x4:
// CHECK: f32x4.convert_i32x4_u{{$}}
v128_t test_f32x4_convert_u32x4(v128_t a) {
  return wasm_f32x4_convert_u32x4(a);
}

// CHECK-LABEL: test_f64x2_convert_low_i32x4:
// CHECK: f64x2.convert_low_i32x4_s{{$}}
v128_t test_f64x2_convert_low_i32x4(v128_t a) {
  return wasm_f64x2_convert_low_i32x4(a);
}

// CHECK-LABEL: test_f64x2_convert_low_u32x4:
// CHECK: f64x2.convert_low_i32x4_u{{$}}
v128_t test_f64x2_convert_low_u32x4(v128_t a) {
  return wasm_f64x2_convert_low_u32x4(a);
}

// CHECK-LABEL: test_i32x4_trunc_sat_f64x2_zero:
// CHECK: i32x4.trunc_sat_f64x2_s_zero{{$}}
v128_t test_i32x4_trunc_sat_f64x2_zero(v128_t a) {
  return wasm_i32x4_trunc_sat_f64x2_zero(a);
}

// CHECK-LABEL: test_u32x4_trunc_sat_f64x2_zero:
// CHECK: i32x4.trunc_sat_f64x2_u_zero{{$}}
v128_t test_u32x4_trunc_sat_f64x2_zero(v128_t a) {
  return wasm_u32x4_trunc_sat_f64x2_zero(a);
}

// CHECK-LABEL: test_f32x4_demote_f64x2_zero:
// CHECK: f32x4.demote_f64x2_zero{{$}}
v128_t test_f32x4_demote_f64x2_zero(v128_t a) {
  return wasm_f32x4_demote_f64x2_zero(a);
}

// CHECK-LABEL: test_f64x2_promote_low_f32x4:
// CHECK: f64x2.promote_low_f32x4{{$}}
v128_t test_f64x2_promote_low_f32x4(v128_t a) {
  return wasm_f64x2_promote_low_f32x4(a);
}

// CHECK-LABEL: test_i8x16_shuffle:
// CHECK: i8x16.shuffle 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
// 0{{$}}
v128_t test_i8x16_shuffle(v128_t a, v128_t b) {
  return wasm_i8x16_shuffle(a, b, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3,
                            2, 1, 0);
}

// CHECK-LABEL: test_i16x8_shuffle:
// CHECK: i8x16.shuffle 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0,
// 1{{$}}
v128_t test_i16x8_shuffle(v128_t a, v128_t b) {
  return wasm_i16x8_shuffle(a, b, 7, 6, 5, 4, 3, 2, 1, 0);
}

// CHECK-LABEL: test_i32x4_shuffle:
// CHECK: i8x16.shuffle 12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2,
// 3{{$}}
v128_t test_i32x4_shuffle(v128_t a, v128_t b) {
  return wasm_i32x4_shuffle(a, b, 3, 2, 1, 0);
}

// CHECK-LABEL: test_i64x2_shuffle:
// CHECK: i8x16.shuffle 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6,
// 7{{$}}
v128_t test_i64x2_shuffle(v128_t a, v128_t b) {
  return wasm_i64x2_shuffle(a, b, 1, 0);
}

// CHECK-LABEL: test_i8x16_swizzle:
// CHECK: i8x16.swizzle{{$}}
v128_t test_i8x16_swizzle(v128_t a, v128_t b) {
  return wasm_i8x16_swizzle(a, b);
}

// CHECK-LABEL: test_i8x16_narrow_i16x8:
// CHECK: i8x16.narrow_i16x8_s{{$}}
v128_t test_i8x16_narrow_i16x8(v128_t a, v128_t b) {
  return wasm_i8x16_narrow_i16x8(a, b);
}

// CHECK-LABEL: test_u8x16_narrow_i16x8:
// CHECK: i8x16.narrow_i16x8_u{{$}}
v128_t test_u8x16_narrow_i16x8(v128_t a, v128_t b) {
  return wasm_u8x16_narrow_i16x8(a, b);
}

// CHECK-LABEL: test_i16x8_narrow_i32x4:
// CHECK: i16x8.narrow_i32x4_s{{$}}
v128_t test_i16x8_narrow_i32x4(v128_t a, v128_t b) {
  return wasm_i16x8_narrow_i32x4(a, b);
}

// CHECK-LABEL: test_u16x8_narrow_i32x4:
// CHECK: i16x8.narrow_i32x4_u{{$}}
v128_t test_u16x8_narrow_i32x4(v128_t a, v128_t b) {
  return wasm_u16x8_narrow_i32x4(a, b);
}

// CHECK-LABEL: test_i16x8_extend_low_i8x16:
// CHECK: i16x8.extend_low_i8x16_s{{$}}
v128_t test_i16x8_extend_low_i8x16(v128_t a) {
  return wasm_i16x8_extend_low_i8x16(a);
}

// CHECK-LABEL: test_i16x8_extend_high_i8x16:
// CHECK: i16x8.extend_high_i8x16_s{{$}}
v128_t test_i16x8_extend_high_i8x16(v128_t a) {
  return wasm_i16x8_extend_high_i8x16(a);
}

// CHECK-LABEL: test_u16x8_extend_low_u8x16:
// CHECK: i16x8.extend_low_i8x16_u{{$}}
v128_t test_u16x8_extend_low_u8x16(v128_t a) {
  return wasm_u16x8_extend_low_u8x16(a);
}

// CHECK-LABEL: test_u16x8_extend_high_u8x16:
// CHECK: i16x8.extend_high_i8x16_u{{$}}
v128_t test_u16x8_extend_high_u8x16(v128_t a) {
  return wasm_u16x8_extend_high_u8x16(a);
}

// CHECK-LABEL: test_i32x4_extend_low_i16x8:
// CHECK: i32x4.extend_low_i16x8_s{{$}}
v128_t test_i32x4_extend_low_i16x8(v128_t a) {
  return wasm_i32x4_extend_low_i16x8(a);
}

// CHECK-LABEL: test_i32x4_extend_high_i16x8:
// CHECK: i32x4.extend_high_i16x8_s{{$}}
v128_t test_i32x4_extend_high_i16x8(v128_t a) {
  return wasm_i32x4_extend_high_i16x8(a);
}

// CHECK-LABEL: test_u32x4_extend_low_u16x8:
// CHECK: i32x4.extend_low_i16x8_u{{$}}
v128_t test_u32x4_extend_low_u16x8(v128_t a) {
  return wasm_u32x4_extend_low_u16x8(a);
}

// CHECK-LABEL: test_u32x4_extend_high_u16x8:
// CHECK: i32x4.extend_high_i16x8_u{{$}}
v128_t test_u32x4_extend_high_u16x8(v128_t a) {
  return wasm_u32x4_extend_high_u16x8(a);
}

// CHECK-LABEL: test_i64x2_extend_low_i32x4:
// CHECK: i64x2.extend_low_i32x4_s{{$}}
v128_t test_i64x2_extend_low_i32x4(v128_t a) {
  return wasm_i64x2_extend_low_i32x4(a);
}

// CHECK-LABEL: test_i64x2_extend_high_i32x4:
// CHECK: i64x2.extend_high_i32x4_s{{$}}
v128_t test_i64x2_extend_high_i32x4(v128_t a) {
  return wasm_i64x2_extend_high_i32x4(a);
}

// CHECK-LABEL: test_u64x2_extend_low_u32x4:
// CHECK: i64x2.extend_low_i32x4_u{{$}}
v128_t test_u64x2_extend_low_u32x4(v128_t a) {
  return wasm_u64x2_extend_low_u32x4(a);
}

// CHECK-LABEL: test_u64x2_extend_high_u32x4:
// CHECK: i64x2.extend_high_i32x4_u{{$}}
v128_t test_u64x2_extend_high_u32x4(v128_t a) {
  return wasm_u64x2_extend_high_u32x4(a);
}

// CHECK-LABEL: test_i16x8_extadd_pairwise_i8x16:
// CHECK: i16x8.extadd_pairwise_i8x16_s{{$}}
v128_t test_i16x8_extadd_pairwise_i8x16(v128_t a) {
  return wasm_i16x8_extadd_pairwise_i8x16(a);
}

// CHECK-LABEL: test_u16x8_extadd_pairwise_u8x16:
// CHECK: i16x8.extadd_pairwise_i8x16_u{{$}}
v128_t test_u16x8_extadd_pairwise_u8x16(v128_t a) {
  return wasm_u16x8_extadd_pairwise_u8x16(a);
}

// CHECK-LABEL: test_i32x4_extadd_pairwise_i16x8:
// CHECK: i32x4.extadd_pairwise_i16x8_s{{$}}
v128_t test_i32x4_extadd_pairwise_i16x8(v128_t a) {
  return wasm_i32x4_extadd_pairwise_i16x8(a);
}

// CHECK-LABEL: test_u32x4_extadd_pairwise_u16x8:
// CHECK: i32x4.extadd_pairwise_i16x8_u{{$}}
v128_t test_u32x4_extadd_pairwise_u16x8(v128_t a) {
  return wasm_u32x4_extadd_pairwise_u16x8(a);
}

// CHECK-LABEL: test_i16x8_extmul_low_i8x16:
// CHECK: i16x8.extmul_low_i8x16_s{{$}}
v128_t test_i16x8_extmul_low_i8x16(v128_t a, v128_t b) {
  return wasm_i16x8_extmul_low_i8x16(a, b);
}

// CHECK-LABEL: test_i16x8_extmul_high_i8x16:
// CHECK: i16x8.extmul_high_i8x16_s{{$}}
v128_t test_i16x8_extmul_high_i8x16(v128_t a, v128_t b) {
  return wasm_i16x8_extmul_high_i8x16(a, b);
}

// CHECK-LABEL: test_u16x8_extmul_low_u8x16:
// CHECK: i16x8.extmul_low_i8x16_u{{$}}
v128_t test_u16x8_extmul_low_u8x16(v128_t a, v128_t b) {
  return wasm_u16x8_extmul_low_u8x16(a, b);
}

// CHECK-LABEL: test_u16x8_extmul_high_u8x16:
// CHECK: i16x8.extmul_high_i8x16_u{{$}}
v128_t test_u16x8_extmul_high_u8x16(v128_t a, v128_t b) {
  return wasm_u16x8_extmul_high_u8x16(a, b);
}

// CHECK-LABEL: test_i32x4_extmul_low_i16x8:
// CHECK: i32x4.extmul_low_i16x8_s{{$}}
v128_t test_i32x4_extmul_low_i16x8(v128_t a, v128_t b) {
  return wasm_i32x4_extmul_low_i16x8(a, b);
}

// CHECK-LABEL: test_i32x4_extmul_high_i16x8:
// CHECK: i32x4.extmul_high_i16x8_s{{$}}
v128_t test_i32x4_extmul_high_i16x8(v128_t a, v128_t b) {
  return wasm_i32x4_extmul_high_i16x8(a, b);
}

// CHECK-LABEL: test_u32x4_extmul_low_u16x8:
// CHECK: i32x4.extmul_low_i16x8_u{{$}}
v128_t test_u32x4_extmul_low_u16x8(v128_t a, v128_t b) {
  return wasm_u32x4_extmul_low_u16x8(a, b);
}

// CHECK-LABEL: test_u32x4_extmul_high_u16x8:
// CHECK: i32x4.extmul_high_i16x8_u{{$}}
v128_t test_u32x4_extmul_high_u16x8(v128_t a, v128_t b) {
  return wasm_u32x4_extmul_high_u16x8(a, b);
}

// CHECK-LABEL: test_i64x2_extmul_low_i32x4:
// CHECK: i64x2.extmul_low_i32x4_s{{$}}
v128_t test_i64x2_extmul_low_i32x4(v128_t a, v128_t b) {
  return wasm_i64x2_extmul_low_i32x4(a, b);
}

// CHECK-LABEL: test_i64x2_extmul_high_i32x4:
// CHECK: i64x2.extmul_high_i32x4_s{{$}}
v128_t test_i64x2_extmul_high_i32x4(v128_t a, v128_t b) {
  return wasm_i64x2_extmul_high_i32x4(a, b);
}

// CHECK-LABEL: test_u64x2_extmul_low_u32x4:
// CHECK: i64x2.extmul_low_i32x4_u{{$}}
v128_t test_u64x2_extmul_low_u32x4(v128_t a, v128_t b) {
  return wasm_u64x2_extmul_low_u32x4(a, b);
}

// CHECK-LABEL: test_u64x2_extmul_high_u32x4:
// CHECK: i64x2.extmul_high_i32x4_u{{$}}
v128_t test_u64x2_extmul_high_u32x4(v128_t a, v128_t b) {
  return wasm_u64x2_extmul_high_u32x4(a, b);
}

// CHECK-LABEL: test_i16x8_q15mulr_sat:
// CHECK: i16x8.q15mulr_sat_s{{$}}
v128_t test_i16x8_q15mulr_sat(v128_t a, v128_t b) {
  return wasm_i16x8_q15mulr_sat(a, b);
}

// CHECK-LABEL: test_f32x4_relaxed_madd:
// CHECK: f32x4.relaxed_madd{{$}}
v128_t test_f32x4_relaxed_madd(v128_t a, v128_t b, v128_t c) {
  return wasm_f32x4_relaxed_madd(a, b, c);
}

// CHECK-LABEL: test_f32x4_relaxed_nmadd:
// CHECK: f32x4.relaxed_nmadd{{$}}
v128_t test_f32x4_relaxed_nmadd(v128_t a, v128_t b, v128_t c) {
  return wasm_f32x4_relaxed_nmadd(a, b, c);
}

// CHECK-LABEL: test_f64x2_relaxed_madd:
// CHECK: f64x2.relaxed_madd{{$}}
v128_t test_f64x2_relaxed_madd(v128_t a, v128_t b, v128_t c) {
  return wasm_f64x2_relaxed_madd(a, b, c);
}

// CHECK-LABEL: test_f64x2_relaxed_nmadd:
// CHECK: f64x2.relaxed_nmadd{{$}}
v128_t test_f64x2_relaxed_nmadd(v128_t a, v128_t b, v128_t c) {
  return wasm_f64x2_relaxed_nmadd(a, b, c);
}

// CHECK-LABEL: test_i8x16_relaxed_laneselect:
// CHECK: i8x16.relaxed_laneselect{{$}}
v128_t test_i8x16_relaxed_laneselect(v128_t a, v128_t b, v128_t m) {
  return wasm_i8x16_relaxed_laneselect(a, b, m);
}

// CHECK-LABEL: test_i16x8_relaxed_laneselect:
// CHECK: i16x8.relaxed_laneselect{{$}}
v128_t test_i16x8_relaxed_laneselect(v128_t a, v128_t b, v128_t m) {
  return wasm_i16x8_relaxed_laneselect(a, b, m);
}

// CHECK-LABEL: test_i32x4_relaxed_laneselect:
// CHECK: i32x4.relaxed_laneselect{{$}}
v128_t test_i32x4_relaxed_laneselect(v128_t a, v128_t b, v128_t m) {
  return wasm_i32x4_relaxed_laneselect(a, b, m);
}

// CHECK-LABEL: test_i64x2_relaxed_laneselect:
// CHECK: i64x2.relaxed_laneselect{{$}}
v128_t test_i64x2_relaxed_laneselect(v128_t a, v128_t b, v128_t m) {
  return wasm_i64x2_relaxed_laneselect(a, b, m);
}

// CHECK-LABEL: test_i8x16_relaxed_swizzle:
// CHECK: i8x16.relaxed_swizzle{{$}}
v128_t test_i8x16_relaxed_swizzle(v128_t a, v128_t s) {
  return wasm_i8x16_relaxed_swizzle(a, s);
}

// CHECK-LABEL: test_f32x4_relaxed_min:
// CHECK: f32x4.relaxed_min{{$}}
v128_t test_f32x4_relaxed_min(v128_t a, v128_t b) {
  return wasm_f32x4_relaxed_min(a, b);
}

// CHECK-LABEL: test_f32x4_relaxed_max:
// CHECK: f32x4.relaxed_max{{$}}
v128_t test_f32x4_relaxed_max(v128_t a, v128_t b) {
  return wasm_f32x4_relaxed_max(a, b);
}

// CHECK-LABEL: test_f64x2_relaxed_min:
// CHECK: f64x2.relaxed_min{{$}}
v128_t test_f64x2_relaxed_min(v128_t a, v128_t b) {
  return wasm_f64x2_relaxed_min(a, b);
}

// CHECK-LABEL: test_f64x2_relaxed_max:
// CHECK: f64x2.relaxed_max
v128_t test_f64x2_relaxed_max(v128_t a, v128_t b) {
  return wasm_f64x2_relaxed_max(a, b);
}

// CHECK-LABEL: test_i32x4_relaxed_trunc_f32x4:
// CHECK: i32x4.relaxed_trunc_f32x4_s{{$}}
v128_t test_i32x4_relaxed_trunc_f32x4(v128_t a) {
  return wasm_i32x4_relaxed_trunc_f32x4(a);
}

// CHECK-LABEL: test_u32x4_relaxed_trunc_f32x4:
// CHECK: i32x4.relaxed_trunc_f32x4_u{{$}}
v128_t test_u32x4_relaxed_trunc_f32x4(v128_t a) {
  return wasm_u32x4_relaxed_trunc_f32x4(a);
}

// CHECK-LABEL: test_i32x4_relaxed_trunc_f64x2_zero:
// CHECK: i32x4.relaxed_trunc_f64x2_s_zero{{$}}
v128_t test_i32x4_relaxed_trunc_f64x2_zero(v128_t a) {
  return wasm_i32x4_relaxed_trunc_f64x2_zero(a);
}

// CHECK-LABEL: test_u32x4_relaxed_trunc_f64x2_zero:
// CHECK: i32x4.relaxed_trunc_f64x2_u_zero{{$}}
v128_t test_u32x4_relaxed_trunc_f64x2_zero(v128_t a) {
  return wasm_u32x4_relaxed_trunc_f64x2_zero(a);
}

// CHECK-LABEL: test_i16x8_relaxed_q15mulr:
// CHECK: i16x8.relaxed_q15mulr_s{{$}}
v128_t test_i16x8_relaxed_q15mulr(v128_t a, v128_t b) {
  return wasm_i16x8_relaxed_q15mulr(a, b);
}

// CHECK-LABEL: test_i16x8_relaxed_dot_i8x16_i7x16:
// CHECK: i16x8.relaxed_dot_i8x16_i7x16_s{{$}}
v128_t test_i16x8_relaxed_dot_i8x16_i7x16(v128_t a, v128_t b) {
  return wasm_i16x8_relaxed_dot_i8x16_i7x16(a, b);
}

// CHECK-LABEL: test_i32x4_relaxed_dot_i8x16_i7x16_add:
// CHECK: i32x4.relaxed_dot_i8x16_i7x16_add_s{{$}}
v128_t test_i32x4_relaxed_dot_i8x16_i7x16_add(v128_t a, v128_t b, v128_t c) {
  return wasm_i32x4_relaxed_dot_i8x16_i7x16_add(a, b, c);
}

// CHECK-LABEL: test_f16x8_splat:
// CHECK: f16x8.splat{{$}}
v128_t test_f16x8_splat(float a) { return wasm_f16x8_splat(a); }

// CHECK-LABEL: test_f16x8_extract_lane:
// CHECK: f16x8.extract_lane 7{{$}}
int16_t test_f16x8_extract_lane(v128_t a) {
  return wasm_f16x8_extract_lane(a, 7);
}

// CHECK-LABEL: test_f16x8_replace_lane:
// CHECK: f16x8.replace_lane 7{{$}}
v128_t test_f16x8_replace_lane(v128_t a, float b) {
  return wasm_f16x8_replace_lane(a, 7, b);
}

// CHECK-LABEL: test_f16x8_abs:
// CHECK: f16x8.abs{{$}}
v128_t test_f16x8_abs(v128_t a) { return wasm_f16x8_abs(a); }

// CHECK-LABEL: test_f16x8_neg:
// CHECK: f16x8.neg{{$}}
v128_t test_f16x8_neg(v128_t a) { return wasm_f16x8_neg(a); }

// CHECK-LABEL: test_f16x8_sqrt:
// CHECK: f16x8.sqrt{{$}}
v128_t test_f16x8_sqrt(v128_t a) { return wasm_f16x8_sqrt(a); }

// CHECK-LABEL: test_f16x8_ceil:
// CHECK: f16x8.ceil{{$}}
v128_t test_f16x8_ceil(v128_t a) { return wasm_f16x8_ceil(a); }

// CHECK-LABEL: test_f16x8_floor:
// CHECK: f16x8.floor{{$}}
v128_t test_f16x8_floor(v128_t a) { return wasm_f16x8_floor(a); }

// CHECK-LABEL: test_f16x8_trunc:
// CHECK: f16x8.trunc{{$}}
v128_t test_f16x8_trunc(v128_t a) { return wasm_f16x8_trunc(a); }

// CHECK-LABEL: test_f16x8_nearest:
// CHECK: f16x8.nearest{{$}}
v128_t test_f16x8_nearest(v128_t a) { return wasm_f16x8_nearest(a); }

// CHECK-LABEL: test_f16x8_add:
// CHECK: f16x8.add{{$}}
v128_t test_f16x8_add(v128_t a, v128_t b) { return wasm_f16x8_add(a, b); }

// CHECK-LABEL: test_f16x8_sub:
// CHECK: f16x8.sub{{$}}
v128_t test_f16x8_sub(v128_t a, v128_t b) { return wasm_f16x8_sub(a, b); }

// CHECK-LABEL: test_f16x8_mul:
// CHECK: f16x8.mul{{$}}
v128_t test_f16x8_mul(v128_t a, v128_t b) { return wasm_f16x8_mul(a, b); }

// CHECK-LABEL: test_f16x8_div:
// CHECK: f16x8.div{{$}}
v128_t test_f16x8_div(v128_t a, v128_t b) { return wasm_f16x8_div(a, b); }

// CHECK-LABEL: test_f16x8_min:
// CHECK: f16x8.min{{$}}
v128_t test_f16x8_min(v128_t a, v128_t b) { return wasm_f16x8_min(a, b); }

// CHECK-LABEL: test_f16x8_max:
// CHECK: f16x8.max{{$}}
v128_t test_f16x8_max(v128_t a, v128_t b) { return wasm_f16x8_max(a, b); }

// CHECK-LABEL: test_f16x8_pmin:
// CHECK: f16x8.pmin{{$}}
v128_t test_f16x8_pmin(v128_t a, v128_t b) { return wasm_f16x8_pmin(a, b); }

// CHECK-LABEL: test_f16x8_pmax:
// CHECK: f16x8.pmax{{$}}
v128_t test_f16x8_pmax(v128_t a, v128_t b) { return wasm_f16x8_pmax(a, b); }

// CHECK-LABEL: test_f16x8_eq:
// CHECK: f16x8.eq{{$}}
v128_t test_f16x8_eq(v128_t a, v128_t b) { return wasm_f16x8_eq(a, b); }

// CHECK-LABEL: test_f16x8_ne:
// CHECK: f16x8.ne{{$}}
v128_t test_f16x8_ne(v128_t a, v128_t b) { return wasm_f16x8_ne(a, b); }

// CHECK-LABEL: test_f16x8_lt:
// CHECK: f16x8.lt{{$}}
v128_t test_f16x8_lt(v128_t a, v128_t b) { return wasm_f16x8_lt(a, b); }

// CHECK-LABEL: test_f16x8_gt:
// CHECK: f16x8.gt{{$}}
v128_t test_f16x8_gt(v128_t a, v128_t b) { return wasm_f16x8_gt(a, b); }

// CHECK-LABEL: test_f16x8_le:
// CHECK: f16x8.le{{$}}
v128_t test_f16x8_le(v128_t a, v128_t b) { return wasm_f16x8_le(a, b); }

// CHECK-LABEL: test_f16x8_ge:
// CHECK: f16x8.ge{{$}}
v128_t test_f16x8_ge(v128_t a, v128_t b) { return wasm_f16x8_ge(a, b); }

// CHECK-LABEL: test_i16x8_trunc_sat_f16x8:
// CHECK: i16x8.trunc_sat_f16x8_s{{$}}
v128_t test_i16x8_trunc_sat_f16x8(v128_t a) {
  return wasm_i16x8_trunc_sat_f16x8(a);
}

// CHECK-LABEL: test_u16x8_trunc_sat_f16x8:
// CHECK: i16x8.trunc_sat_f16x8_u{{$}}
v128_t test_u16x8_trunc_sat_f16x8(v128_t a) {
  return wasm_u16x8_trunc_sat_f16x8(a);
}

// CHECK-LABEL: test_f16x8_convert_i16x8:
// CHECK: f16x8.convert_i16x8_s{{$}}
v128_t test_f16x8_convert_i16x8(v128_t a) {
  return wasm_f16x8_convert_i16x8(a);
}

// CHECK-LABEL: test_f16x8_convert_u16x8:
// CHECK: f16x8.convert_i16x8_u{{$}}
v128_t test_f16x8_convert_u16x8(v128_t a) {
  return wasm_f16x8_convert_u16x8(a);
}

// CHECK-LABEL: test_f16x8_relaxed_madd:
// CHECK: f16x8.madd{{$}}
v128_t test_f16x8_relaxed_madd(v128_t a, v128_t b, v128_t c) {
  return wasm_f16x8_relaxed_madd(a, b, c);
}

// CHECK-LABEL: test_f16x8_relaxed_nmadd:
// CHECK: f16x8.nmadd{{$}}
v128_t test_f16x8_relaxed_nmadd(v128_t a, v128_t b, v128_t c) {
  return wasm_f16x8_relaxed_nmadd(a, b, c);
}

// RUN: %clang_cc1 -triple riscv64 -target-feature +experimental-v -ast-print %s \
// RUN:    | FileCheck %s

void bar(void) {
  // CHECK: __rvv_int64m1_t x0;
  __rvv_int64m1_t x0;

  // CHECK: __rvv_float64m1_t x1;
  __rvv_float64m1_t x1;

  // CHECK: __rvv_int64m2_t x2;
  __rvv_int64m2_t x2;

  // CHECK: __rvv_float64m2_t x3;
  __rvv_float64m2_t x3;

  // CHECK: __rvv_int64m4_t x4;
  __rvv_int64m4_t x4;

  // CHECK: __rvv_float64m4_t x5;
  __rvv_float64m4_t x5;

  // CHECK: __rvv_int64m8_t x6;
  __rvv_int64m8_t x6;

  // CHECK: __rvv_float64m8_t x7;
  __rvv_float64m8_t x7;

  // CHECK: __rvv_int32m1_t x8;
  __rvv_int32m1_t x8;

  // CHECK: __rvv_float32m1_t x9;
  __rvv_float32m1_t x9;

  // CHECK: __rvv_int32m2_t x10;
  __rvv_int32m2_t x10;

  // CHECK: __rvv_float32m2_t x11;
  __rvv_float32m2_t x11;

  // CHECK: __rvv_int32m4_t x12;
  __rvv_int32m4_t x12;

  // CHECK: __rvv_float32m4_t x13;
  __rvv_float32m4_t x13;

  // CHECK: __rvv_int32m8_t x14;
  __rvv_int32m8_t x14;

  // CHECK: __rvv_float32m8_t x15;
  __rvv_float32m8_t x15;

  // CHECK: __rvv_int16m1_t x16;
  __rvv_int16m1_t x16;

  // CHECK: __rvv_float16m1_t x17;
  __rvv_float16m1_t x17;

  // CHECK: __rvv_int16m2_t x18;
  __rvv_int16m2_t x18;

  // CHECK: __rvv_float16m2_t x19;
  __rvv_float16m2_t x19;

  // CHECK: __rvv_int16m4_t x20;
  __rvv_int16m4_t x20;

  // CHECK: __rvv_float16m4_t x21;
  __rvv_float16m4_t x21;

  // CHECK: __rvv_int16m8_t x22;
  __rvv_int16m8_t x22;

  // CHECK: __rvv_float16m8_t x23;
  __rvv_float16m8_t x23;

  // CHECK: __rvv_int8m1_t x24;
  __rvv_int8m1_t x24;

  // CHECK: __rvv_int8m2_t x25;
  __rvv_int8m2_t x25;

  // CHECK: __rvv_int8m4_t x26;
  __rvv_int8m4_t x26;

  // CHECK: __rvv_int8m8_t x27;
  __rvv_int8m8_t x27;

  // CHECK: __rvv_bool64_t x28;
  __rvv_bool64_t x28;

  // CHECK: __rvv_bool32_t x29;
  __rvv_bool32_t x29;

  // CHECK: __rvv_bool16_t x30;
  __rvv_bool16_t x30;

  // CHECK: __rvv_bool8_t x31;
  __rvv_bool8_t x31;

  // CHECK: __rvv_bool8_t x32;
  __rvv_bool8_t x32;

  // CHECK: __rvv_bool8_t x33;
  __rvv_bool8_t x33;

  // CHECK: __rvv_bool8_t x34;
  __rvv_bool8_t x34;

  // CHECK: __rvv_int32mf2_t x35;
  __rvv_int32mf2_t x35;

  // CHECK: __rvv_float32mf2_t x36;
  __rvv_float32mf2_t x36;

  // CHECK: __rvv_int16mf4_t x37;
  __rvv_int16mf4_t x37;

  // CHECK: __rvv_float16mf4_t x38;
  __rvv_float16mf4_t x38;

  // CHECK: __rvv_int16mf2_t x39;
  __rvv_int16mf2_t x39;

  // CHECK: __rvv_float16mf2_t x40;
  __rvv_float16mf2_t x40;

  // CHECK: __rvv_int8mf8_t x41;
  __rvv_int8mf8_t x41;

  // CHECK: __rvv_int8mf4_t x42;
  __rvv_int8mf4_t x42;

  // CHECK: __rvv_int8mf2_t x43;
  __rvv_int8mf2_t x43;
}

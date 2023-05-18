// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -D__HLSL_ENABLE_16_BIT \
// RUN:   -emit-llvm -disable-llvm-passes -O3 -o - | FileCheck %s

// FIXME: add test for char/int8_t/uint8_t when these types are supported in HLSL.
//  See https://github.com/llvm/llvm-project/issues/58453.

// Make sure generate i16 add.
// CHECK: add nsw i16 %
int16_t add(int16_t a, int16_t b) {
  return a + b;
}
// CHECK: define noundef <2 x i16> @
// CHECK: add <2 x i16>
int16_t2 add(int16_t2 a, int16_t2 b) {
  return a + b;
}
// CHECK: define noundef <3 x i16> @
// CHECK: add <3 x i16>
int16_t3 add(int16_t3 a, int16_t3 b) {
  return a + b;
}
// CHECK: define noundef <4 x i16> @
// CHECK: add <4 x i16>
int16_t4 add(int16_t4 a, int16_t4 b) {
  return a + b;
}
// CHECK: define noundef i16 @
// CHECK: add i16 %
uint16_t add(uint16_t a, uint16_t b) {
  return a + b;
}
// CHECK: define noundef <2 x i16> @
// CHECK: add <2 x i16>
uint16_t2 add(uint16_t2 a, uint16_t2 b) {
  return a + b;
}
// CHECK: define noundef <3 x i16> @
// CHECK: add <3 x i16>
uint16_t3 add(uint16_t3 a, uint16_t3 b) {
  return a + b;
}
// CHECK: define noundef <4 x i16> @
// CHECK: add <4 x i16>
uint16_t4 add(uint16_t4 a, uint16_t4 b) {
  return a + b;
}

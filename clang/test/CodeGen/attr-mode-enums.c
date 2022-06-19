// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s

// Test checks that 'mode' attribute is handled correctly with enums, i. e. code
//   1. "typedef enum { A } __attribute__((mode(HI))) T;" is accepted,
//   2. "enum X __attribute__((mode(QI))) var;" forms a complete integer type.

int main(void) {
  // CHECK: [[X1:%.+]] = alloca i8
  enum { A1, B1 } __attribute__((mode(QI))) x1 = A1;

  // CHECK: [[X2:%.+]] = alloca i16
  enum { A2, B2 } x2 __attribute__((mode(HI))) = B2;

  // CHECK: [[X3:%.+]] = alloca i32
  typedef enum { A3, B3 } __attribute__((mode(SI))) T3;
  T3 x3 = A3;

  // CHECK: [[X4:%.+]] = alloca i64
  typedef enum { A4, B4 } T4 __attribute__((mode(DI)));
  T4 x4 = B4;

  // CHECK: [[X5:%.+]] = alloca i8
  typedef enum __attribute__((mode(QI))) { A5, B5 } T5;
  T5 x5 = A5;

  // CHECK: [[X6:%.+]] = alloca i8
  typedef enum X __attribute__((mode(QI))) T6;
  T6 x6;

  // CHECK: [[X7:%.+]] = alloca i128
  enum { A7, B7 } __attribute__((mode(TI))) x7 = A7;

  // CHECK: [[X8:%.+]] = alloca i8
  enum __attribute__((mode(QI))) { A8, B8 } x8 = B8;

  // CHECK: store i8 0, i8* [[X1]]
  // CHECK: store i16 1, i16* [[X2]]
  // CHECK: store i32 0, i32* [[X3]]
  // CHECK: store i64 1, i64* [[X4]]
  // CHECK: store i8 0, i8* [[X5]]
  // CHECK: store i128 0, i128* [[X7]]
  // CHECK: store i8 1, i8* [[X8]]

  return x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8;
}

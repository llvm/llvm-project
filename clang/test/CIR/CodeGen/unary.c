// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -Wno-unused-value -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

void valueNegation(int i, short s, long l, float f, double d) {
// CHECK: cir.func @valueNegation(
  !i;
  // CHECK: %[[#INT:]] = cir.load %{{[0-9]+}} : cir.ptr <!s32i>, !s32i
  // CHECK: %[[#INT_TO_BOOL:]] = cir.cast(int_to_bool, %[[#INT]] : !s32i), !cir.bool
  // CHECK: = cir.unary(not, %[[#INT_TO_BOOL]]) : !cir.bool, !cir.bool
  !s;
  // CHECK: %[[#SHORT:]] = cir.load %{{[0-9]+}} : cir.ptr <!s16i>, !s16i
  // CHECK: %[[#SHORT_TO_BOOL:]] = cir.cast(int_to_bool, %[[#SHORT]] : !s16i), !cir.bool
  // CHECK: = cir.unary(not, %[[#SHORT_TO_BOOL]]) : !cir.bool, !cir.bool
  !l;
  // CHECK: %[[#LONG:]] = cir.load %{{[0-9]+}} : cir.ptr <!s64i>, !s64i
  // CHECK: %[[#LONG_TO_BOOL:]] = cir.cast(int_to_bool, %[[#LONG]] : !s64i), !cir.bool
  // CHECK: = cir.unary(not, %[[#LONG_TO_BOOL]]) : !cir.bool, !cir.bool
  !f;
  // CHECK: %[[#FLOAT:]] = cir.load %{{[0-9]+}} : cir.ptr <!cir.float>, !cir.float
  // CHECK: %[[#FLOAT_TO_BOOL:]] = cir.cast(float_to_bool, %[[#FLOAT]] : !cir.float), !cir.bool
  // CHECK: %[[#FLOAT_NOT:]] = cir.unary(not, %[[#FLOAT_TO_BOOL]]) : !cir.bool, !cir.bool
  // CHECK: = cir.cast(bool_to_int, %[[#FLOAT_NOT]] : !cir.bool), !s32i
  !d;
  // CHECK: %[[#DOUBLE:]] = cir.load %{{[0-9]+}} : cir.ptr <!cir.double>, !cir.double
  // CHECK: %[[#DOUBLE_TO_BOOL:]] = cir.cast(float_to_bool, %[[#DOUBLE]] : !cir.double), !cir.bool
  // CHECK: %[[#DOUBLE_NOT:]] = cir.unary(not, %[[#DOUBLE_TO_BOOL]]) : !cir.bool, !cir.bool
  // CHECK: = cir.cast(bool_to_int, %[[#DOUBLE_NOT]] : !cir.bool), !s32i
}

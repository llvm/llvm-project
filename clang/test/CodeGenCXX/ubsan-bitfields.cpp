// RUN: %clang_cc1 -std=c++11 -triple x86_64-apple-darwin10 -emit-llvm -o - %s -fsanitize=enum | FileCheck %s

enum E {
  a = 1,
  b = 2,
  c = 3
};

struct S {
  E e1 : 10;
};

// CHECK-LABEL: define i32 @_Z4loadP1S
E load(S *s) {
  // CHECK: [[LOAD:%.*]] = load i16, i16* {{.*}}
  // CHECK: [[CLEAR:%.*]] = and i16 [[LOAD]], 1023
  // CHECK: [[CAST:%.*]] = zext i16 [[CLEAR]] to i32
  // CHECK: icmp ule i32 [[CAST]], 3, !nosanitize
  // CHECK: call void @__ubsan_handle_load_invalid_value
  return s->e1;
}

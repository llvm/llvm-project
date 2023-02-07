// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -Wno-unused-value -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

unsigned up0() {
  unsigned a = 1;
  return +a;
}

// CHECK: cir.func @_Z3up0v() -> i32 {
// CHECK: %[[#RET:]] = cir.alloca i32, cir.ptr <i32>, ["__retval"]
// CHECK: %[[#A:]] = cir.alloca i32, cir.ptr <i32>, ["a", init]
// CHECK: %[[#INPUT:]] = cir.load %[[#A]]
// CHECK: %[[#OUTPUT:]] = cir.unary(plus, %[[#INPUT]])
// CHECK: cir.store %[[#OUTPUT]], %[[#RET]]

unsigned um0() {
  unsigned a = 1;
  return -a;
}

// CHECK: cir.func @_Z3um0v() -> i32 {
// CHECK: %[[#RET:]] = cir.alloca i32, cir.ptr <i32>, ["__retval"]
// CHECK: %[[#A:]] = cir.alloca i32, cir.ptr <i32>, ["a", init]
// CHECK: %[[#INPUT:]] = cir.load %[[#A]]
// CHECK: %[[#OUTPUT:]] = cir.unary(minus, %[[#INPUT]])
// CHECK: cir.store %[[#OUTPUT]], %[[#RET]]

unsigned un0() {
  unsigned a = 1;
  return ~a; // a ^ -1 , not
}

// CHECK: cir.func @_Z3un0v() -> i32 {
// CHECK: %[[#RET:]] = cir.alloca i32, cir.ptr <i32>, ["__retval"]
// CHECK: %[[#A:]] = cir.alloca i32, cir.ptr <i32>, ["a", init]
// CHECK: %[[#INPUT:]] = cir.load %[[#A]]
// CHECK: %[[#OUTPUT:]] = cir.unary(not, %[[#INPUT]])
// CHECK: cir.store %[[#OUTPUT]], %[[#RET]]

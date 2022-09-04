// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -Wno-unused-value -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

int c0() {
    int a = 1;
    int b = 2;
    return b + 1, a;
}

// CHECK: cir.func @_Z2c0v() -> i32 {
// CHECK: %[[#RET:]] = cir.alloca i32, cir.ptr <i32>, ["__retval", uninitialized]
// CHECK: %[[#A:]] = cir.alloca i32, cir.ptr <i32>, ["a", cinit]
// CHECK: %[[#B:]] = cir.alloca i32, cir.ptr <i32>, ["b", cinit]
// CHECK: %[[#LOADED_B:]] = cir.load %[[#B]] : cir.ptr <i32>, i32
// CHECK: %[[#]] = cir.binop(add, %[[#LOADED_B]], %[[#]]) : i32
// CHECK: %[[#LOADED_A:]] = cir.load %[[#A]] : cir.ptr <i32>, i32
// CHECK: cir.store %[[#LOADED_A]], %[[#RET]] : i32, cir.ptr <i32>

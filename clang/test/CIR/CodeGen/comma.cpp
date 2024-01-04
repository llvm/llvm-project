// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -Wno-unused-value -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

int c0() {
    int a = 1;
    int b = 2;
    return b + 1, a;
}

// CHECK: cir.func @_Z2c0v() -> !s32i
// CHECK: %[[#RET:]] = cir.alloca !s32i, cir.ptr <!s32i>, ["__retval"]
// CHECK: %[[#A:]] = cir.alloca !s32i, cir.ptr <!s32i>, ["a", init]
// CHECK: %[[#B:]] = cir.alloca !s32i, cir.ptr <!s32i>, ["b", init]
// CHECK: %[[#LOADED_B:]] = cir.load %[[#B]] : cir.ptr <!s32i>, !s32i
// CHECK: %[[#]] = cir.binop(add, %[[#LOADED_B]], %[[#]]) : !s32i
// CHECK: %[[#LOADED_A:]] = cir.load %[[#A]] : cir.ptr <!s32i>, !s32i
// CHECK: cir.store %[[#LOADED_A]], %[[#RET]] : !s32i, cir.ptr <!s32i>

int &foo1();
int &foo2();

void c1() {
    int &x = (foo1(), foo2());
}

// CHECK: cir.func @_Z2c1v()
// CHECK: %0 = cir.alloca !cir.ptr<!s32i>, cir.ptr <!cir.ptr<!s32i>>
// CHECK: %1 = cir.call @_Z4foo1v() : () -> !cir.ptr<!s32i>
// CHECK: %2 = cir.call @_Z4foo2v() : () -> !cir.ptr<!s32i>
// CHECK: cir.store %2, %0 : !cir.ptr<!s32i>, cir.ptr <!cir.ptr<!s32i>>

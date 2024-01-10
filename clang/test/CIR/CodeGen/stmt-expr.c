// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// Yields void.
void test1() { ({ }); }
// CHECK: @test1
//     CHECK: cir.scope {
// CHECK-NOT:   cir.yield
//     CHECK: }

// Yields an out-of-scope scalar.
void test2() { ({int x = 3; x; }); }
// CHECK: @test2
// CHECK: %[[#RETVAL:]] = cir.alloca !s32i, cir.ptr <!s32i>
// CHECK: cir.scope {
// CHECK:   %[[#VAR:]] = cir.alloca !s32i, cir.ptr <!s32i>, ["x", init]
//          [...]
// CHECK:   %[[#TMP:]] = cir.load %[[#VAR]] : cir.ptr <!s32i>, !s32i
// CHECK:   cir.store %[[#TMP]], %[[#RETVAL]] : !s32i, cir.ptr <!s32i>
// CHECK: }
// CHECK: %{{.+}} = cir.load %[[#RETVAL]] : cir.ptr <!s32i>, !s32i

// Yields an aggregate.
struct S { int x; };
int test3() { return ({ struct S s = {1}; s; }).x; }
// CHECK: @test3
// CHECK: %[[#RETVAL:]] = cir.alloca !ty_22S22, cir.ptr <!ty_22S22>
// CHECK: cir.scope {
// CHECK:   %[[#VAR:]] = cir.alloca !ty_22S22, cir.ptr <!ty_22S22>
//          [...]
// CHECK:   cir.copy %[[#VAR]] to %[[#RETVAL]] : !cir.ptr<!ty_22S22>
// CHECK: }
// CHECK: %[[#RETADDR:]] = cir.get_member %1[0] {name = "x"} : !cir.ptr<!ty_22S22> -> !cir.ptr<!s32i>
// CHECK: %{{.+}} = cir.load %[[#RETADDR]] : cir.ptr <!s32i>, !s32i

// Expression is wrapped in an expression attribute (just ensure it does not crash).
void test4(int x) { ({[[gsl::suppress("foo")]] x;}); }
// CHECK: @test4

// TODO(cir): Missing label support.
// // Expression is wrapped in a label.
// // void test5(int x) { x = ({ label: x; }); }

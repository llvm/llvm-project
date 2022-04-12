// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// XFAIL: *

void l0() {
  for (;;) {
  }
}

// CHECK: func @l0
// CHECK: cir.loop(cond :  {
// CHECK-NEXT:   %0 = cir.cst(true) : !cir.bool
// CHECK-NEXT:   cir.yield loopcondition %0 : !cir.bool
// CHECK-NEXT: }, step :  {
// CHECK-NEXT:   cir.yield
// CHECK-NEXT: })  {
// CHECK-NEXT:   cir.yield
// CHECK-NEXT: }

void l1() {
  int x = 0;
  for (int i = 0; i < 10; i = i + 1) {
    x = x + 1;
  }
}

// CHECK: func @l1
// CHECK: cir.loop(cond :  {
// CHECK-NEXT:   %4 = cir.load %2 : cir.ptr <i32>, i32
// CHECK-NEXT:   %5 = cir.cst(10 : i32) : i32
// CHECK-NEXT:   %6 = cir.cmp(lt, %4, %5) : i32, !cir.bool
// CHECK-NEXT:   cir.yield loopcondition %6 : !cir.bool
// CHECK-NEXT: }, step :  {
// CHECK-NEXT:   %4 = cir.load %2 : cir.ptr <i32>, i32
// CHECK-NEXT:   %5 = cir.cst(1 : i32) : i32
// CHECK-NEXT:   %6 = cir.binop(add, %4, %5) : i32
// CHECK-NEXT:   cir.store %6, %2 : i32, cir.ptr <i32>
// CHECK-NEXT:   cir.yield
// CHECK-NEXT: })  {
// CHECK-NEXT:   %4 = cir.load %0 : cir.ptr <i32>, i32
// CHECK-NEXT:   %5 = cir.cst(1 : i32) : i32
// CHECK-NEXT:   %6 = cir.binop(add, %4, %5) : i32
// CHECK-NEXT:   cir.store %6, %0 : i32, cir.ptr <i32>
// CHECK-NEXT:   cir.yield
// CHECK-NEXT: }

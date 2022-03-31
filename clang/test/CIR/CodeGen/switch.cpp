// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// XFAIL: *

void sw1(int a) {
  switch (int b = 1; a) {
  case 0:
    b = b + 1;
    break;
  case 1:
    break;
  }
}

// CHECK: func @sw1
// CHECK: cir.switch (%3 : i32) [
// CHECK-NEXT: case (equal, 0 : i32)  {
// CHECK-NEXT:   %4 = cir.load %1 : cir.ptr <i32>, i32
// CHECK-NEXT:   %5 = cir.cst(1 : i32) : i32
// CHECK-NEXT:   %6 = cir.binop(add, %4, %5) : i32
// CHECK-NEXT:   cir.store %6, %1 : i32, cir.ptr <i32>
// CHECK-NEXT:   cir.yield
// CHECK-NEXT: },
// CHECK-NEXT: case (equal, 1 : i32)  {
// CHECK-NEXT:   cir.yield
// CHECK-NEXT: }

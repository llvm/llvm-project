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
  case 2: {
    b = b + 1;
    int yolo = 100;
    break;
  }
  }
}

// CHECK: func @sw1
// CHECK: cir.switch (%3 : i32) [
// CHECK-NEXT: case (equal, 0 : i32)  {
// CHECK-NEXT:   %4 = cir.load %1 : cir.ptr <i32>, i32
// CHECK-NEXT:   %5 = cir.cst(1 : i32) : i32
// CHECK-NEXT:   %6 = cir.binop(add, %4, %5) : i32
// CHECK-NEXT:   cir.store %6, %1 : i32, cir.ptr <i32>
// CHECK-NEXT:   cir.yield break
// CHECK-NEXT: },
// CHECK-NEXT: case (equal, 1 : i32)  {
// CHECK-NEXT:   cir.yield break
// CHECK-NEXT: },
// CHECK-NEXT: case (equal, 2 : i32)  {
// CHECK-NEXT:   cir.scope {
// CHECK-NEXT:       %4 = cir.alloca i32, cir.ptr <i32>, ["yolo", cinit]
// CHECK-NEXT:       %5 = cir.load %1 : cir.ptr <i32>, i32
// CHECK-NEXT:       %6 = cir.cst(1 : i32) : i32
// CHECK-NEXT:       %7 = cir.binop(add, %5, %6) : i32
// CHECK-NEXT:       cir.store %7, %1 : i32, cir.ptr <i32>
// CHECK-NEXT:       %8 = cir.cst(100 : i32) : i32
// CHECK-NEXT:       cir.store %8, %4 : i32, cir.ptr <i32>
// CHECK-NEXT:       cir.yield break
// CHECK-NEXT:     }
// CHECK-NEXT:     cir.yield fallthrough
// CHECK-NEXT:   }

void sw2(int a) {
  switch (int yolo = 2; a) {
  case 3:
    // "fomo" has the same lifetime as "yolo"
    int fomo = 0;
    yolo = yolo + fomo;
    break;
  }
}

// CHECK: func @sw2
// CHECK: cir.scope {
// CHECK-NEXT:   %1 = cir.alloca i32, cir.ptr <i32>, ["yolo", cinit]
// CHECK-NEXT:   %2 = cir.alloca i32, cir.ptr <i32>, ["fomo", cinit]
// CHECK:        cir.switch (%4 : i32) [
// CHECK-NEXT:   case (equal, 3 : i32)  {
// CHECK-NEXT:     %5 = cir.cst(0 : i32) : i32
// CHECK-NEXT:     cir.store %5, %2 : i32, cir.ptr <i32>